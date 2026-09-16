# Reporting Cache

Phase 1.5 of `docs/PRODUCT_ROADMAP.md` — a pure serving/infrastructure optimization over the frozen Semi-HMM's Reporting API. **The scientific model, its parameters, its checkpoint, and every historical result are untouched** — see `docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md` for the full integrity verification.

## Why this exists

`SemiHMMModel.next_phase_distribution()` costs **~333 seconds for a single 527-window Val trajectory** (measured directly). Root cause (diagnosed, not fixed — fixing it would mean touching the frozen model, out of scope for this phase): it is a `T×K×Dmax` nested Python loop that calls `duration_models[j].log_prob(a)`/`.log_survival(a)` fresh on every one of its ~2.1M inner iterations. Its sibling method, `filtering()` (via `_explicit_duration_forward_optimized`), already documents and applies the fix for exactly this pattern — precompute a `(K, Dmax)` table of `log_prob`/`log_survival` once per model instead of once per `(t,j,a)` triple — but that optimization was never extended to `next_phase_distribution()`. `filtering()` costs ~1.7s for the same trajectory; `next_phase_distribution()` costs ~195x more.

## Architecture

```
Frozen Semi-HMM (unmodified)
        │
        ▼
Full video pass: filtering() + next_phase_distribution()   ← the expensive step, ~333s
        │
        ▼
Cache.py: atomic write to disk, keyed by
(model_name, model_version, split, video_name)
        │
   ┌────┴────┐
   ▼         ▼
L1 (in-process   L2 (disk, Cache/reporting/,
 dict, per        survives process restarts,
 process)         shared across processes)
        │
        ▼
Reporting API (inference_service.infer / trajectory_summary)
        │
        ▼
Web App / RAG / LLM (all future consumers, unaware this cache exists)
```

## Cache key

`(model_name="semi_hmm", model_version, split, video_name)`. `model_version` is `model_loader.model_version_string()` — already existed, a stable fingerprint of every hyperparameter that affects `filtering()`/`next_phase_distribution()`'s output (`dmax`, `duration_family`, `transition_smoothing_alpha`, `transition_decay_rho`, `logreg_class_weight`). A cache entry is **never** read across a version, video, or split mismatch — and additionally never read if the live trajectory's `window_starts` list doesn't match what was cached (guards against a future re-extracted embeddings cache silently reusing a stale entry).

## Stored artifacts

Only two arrays per video: `filtering()`'s `(T,K)` posterior and `next_phase_distribution()`'s `(T,K)` distribution, plus `window_starts` (for the consistency check above) and provenance (`model_version`, `computed_at`, `compute_time_seconds`). **Embeddings are never stored in the cache** — the cached arrays are pure derived output, an order of magnitude smaller than the embeddings that produced them.

### What the future Web App needs, and what's actually cached

Every output `docs/WEBAPP_DATA_REQUIREMENTS.md`/`docs/INFERENCE_SCHEMA.md` identifies, checked against what this cache stores:

| Output | Cached here? | Why |
|---|---|---|
| Phase prediction (`current_phase`) | No — **derived** at read time (`argmax` of the cached posterior row) | O(K)=15 comparisons, trivial; storing it too would be redundant with the posterior |
| Phase probabilities (`phase_probabilities`) | **Yes** — this is exactly `filtering()`'s cached `(T,K)` array | The expensive-to-recompute quantity (well, `filtering()` itself is cheap at ~1.7s/video — cached anyway since it's produced by the same call as the truly expensive one) |
| Entropy | No — **derived** at read time (`HMMModel.entropy()` on the cached posterior row) | O(K), trivial; never re-implemented, reuses the existing static method |
| Next-phase distribution (`next_phase_distribution`) | **Yes** — `next_phase_distribution()`'s cached `(T,K)` array | **This is the ~333-357s bottleneck** — the entire reason this cache exists |
| Duration information (`expected_duration`, `duration_quantiles`) | No — **not trajectory-dependent at all**, a property of the fitted model and a phase index only (`model.duration_models[phase_idx]`) | Measured at 0.0016s per lookup — already below any caching threshold; caching it would add complexity for zero latency benefit |
| Timestamps (`window_start_time` etc.) | No — served from `metadata_with_time.csv` via `trajectory_service` | Already fast (pandas row lookup); a separate, pre-existing concern, not part of the model's forward pass |
| Ground truth / metadata (`sample`, `window`, `model` info) | No — served live from `Trajectory`/`model_loader` | Cheap, and some of it (`inference_timestamp`) is *supposed* to vary per request — caching it would be wrong, not just unnecessary |
| Embeddings | **Never** | Not a Reporting API output at all; caching them here would duplicate `Embeddings/resnet18/` for no benefit |

Net: **exactly the two arrays that are expensive to produce and identical for every window of a given video** are persisted; everything else is either already cheap or must vary per-request by definition.

## Location

`Cache/reporting/<model_name>/<model_version_slug>/<split>/<video_name>/inference.json` — a new top-level, gitignored directory (`Cache/`, added to `.gitignore` this phase), parallel to `Data/`/`Embeddings/`/`Results/`/`Models/`. Deliberately **not** under `Results/evaluation/` — this is serving infrastructure, not a scientific artifact, and mixing the two would blur a distinction this project has otherwise maintained carefully throughout.

## Precompute strategy

**Lazy-cache-then-persist (B), with a thin reusable driver for pre-warming (making A available for free).** A single function, `cache.get_or_compute_forward_pass()`, is the only way any code populates or reads a cache entry — called synchronously inside a real request (lazy: first request pays the cost, every later request for that video is fast) or in a loop from a standalone script (pre-warm: `Training/reporting/warm_cache.py`, not built this phase — the same function reused, not a separate compute path). This was chosen over a dedicated batch-precompute system because it is the simplest mechanism that satisfies both use cases without duplicating logic — directly matching the phase's own stated preference for "la solution la plus simple, reproductible et observable."

## Invalidation

Automatic, per-read, never silent: a cache entry is only ever used if `model_name`, `model_version`, `video_name`, `split`, and `window_starts` all match the live request exactly. Any mismatch — or a corrupted/truncated/malformed JSON file, or a file missing a required field — is treated as a cache miss: logged, recomputed, and the cache entry is atomically overwritten with a fresh, valid one. No manual invalidation command exists or is needed for the current single-model deployment; if a second model version is ever served, its entries simply live alongside the first's under a different `model_version` directory, never colliding.

## Concurrency

A per-cache-key `fcntl.flock()` (POSIX advisory lock, standard library, no new dependency) around the compute-and-write critical section — works across both threads and processes on the same machine. Two simultaneous requests for the same uncached video: the second blocks on the lock until the first finishes, then re-checks the cache (now warm) and reads it, rather than redundantly recomputing. This is a single-machine mechanism, not a distributed lock — sufficient for the current one-box deployment, explicitly not over-engineered into a distributed system per the phase's own constraint. Atomic write (temp file + `os.replace`) additionally guarantees a reader can never observe a partially-written file, and a crash mid-write leaves only an orphaned `.tmp` file, never a corrupted `inference.json`.

## Benchmark and numerical equivalence

See `docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md` §10-11 for the full measured before/after table and the explicit equivalence check (raw model call vs. cache-miss vs. cache-hit, compared array-for-array with `np.array_equal`).

## Limitations

- Cache is local-disk-only — a multi-machine deployment would need a shared filesystem or a different backend (not needed at current scale, flagged for later).
- No TTL/expiry — a cache entry lives until its `(model_version, video, split)` key is invalidated by one of the mismatches above; this is correct as long as `model_version` genuinely changes whenever the served model does (true today, since `model_loader` only ever serves one frozen checkpoint per process).
- Pre-warming all of Val (106 videos × ~333s) is ~9.7 hours run sequentially — parallelizable (each video is independent) but not implemented this phase; `warm_cache.py` is designed but not built (see `docs/PRODUCT_ROADMAP.md` Phase 1.5).
- This cache does not, and is not intended to, address the separate, distinct calibration finding (Brier≈0.96 on the raw posterior) — see Phase 10 discipline below.

## Explicitly out of scope (Phase 10 discipline)

The known calibration problem (`Results/evaluation/global_model_comparison/PRAUC_REPORT.md`: multiclass Brier≈0.96, worse than uniform, on the raw `phase_probabilities` posterior) is a **separate, distinct scientific question**, not addressed, not touched, not recalibrated by this phase. The cache stores and returns exactly what `filtering()`/`next_phase_distribution()` already compute — a pure storage/execution optimization, never a modification of the model's output. The future Web App must continue to treat this limitation exactly as documented in `docs/REPORTING_API.md`'s Known Limitations — the cache changes nothing about how trustworthy these numbers are, only how fast they're served the second time.

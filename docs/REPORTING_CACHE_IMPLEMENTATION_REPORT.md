# Reporting Cache Implementation Report

## 1. Problem

`SemiHMMModel.next_phase_distribution()` costs ~330-360s per video, cold — incompatible with any synchronous use from the Web App, the future RAG, or LLM tool calling (`docs/PRODUCT_ARCHITECTURE.md` §1, §9). This phase closes that gap through infrastructure only — the frozen scientific model, its parameters, checkpoint, and every historical result are untouched.

## 2. Root cause of latency

Diagnosed by reading the real code (`Training/evaluation/models/semi_hmm.py`), not assumed:

**CURRENT COST** (`Patient_319`, 527 windows, measured directly this session):

| Operation | Time |
|---|---:|
| load model | 0.006s |
| load trajectory (one-time embeddings load, shared across all videos in a process) | 4.86s |
| filtering | 1.56s |
| **next_phase_distribution** | **356.93s** |
| duration (single phase lookup) | 0.0016s |

`next_phase_distribution()` is a `T×K×Dmax` (≈2.1M for this trajectory, `Dmax=268`) nested Python loop that calls `duration_models[j].log_prob(a)`/`.log_survival(a)` fresh on every iteration. Its sibling, `filtering()` (via `_explicit_duration_forward_optimized`), already precomputes a `(K,Dmax)` table of these values once per model — its own docstring documents this exact fix and calls the repeated-method-call pattern "the dominant real cost in practice." That optimization was never extended to `next_phase_distribution()`. Everything else measured (model load, trajectory load, filtering, duration lookup) is already fast and needed no caching work.

**What's invariant per trajectory**: the entire `(T,K)` posterior and `(T,K)` next-phase distribution — computed once for the whole video regardless of which window is later requested. **What depends only on the requested window**: nothing in the expensive path — window selection is just a row index into an already-computed matrix. **What was actually being recomputed**: everything, on every process restart, because the pre-existing in-process cache (`inference_service._POSTERIOR_CACHE`/`_NEXT_PHASE_CACHE`) is volatile.

## 3. Architecture before

`inference_service._forward_pass()` → in-process dict cache only → on miss, calls `model.filtering()`/`model.next_phase_distribution()` directly. Fast for repeated requests within one process; every process restart (deploy, crash) pays the full cost again for every video.

## 4. Architecture after

Same call site, now routed through `cache.get_or_compute_forward_pass()` — a disk-backed L2 cache behind the unchanged in-process L1. See `docs/REPORTING_CACHE.md` for the full diagram, cache key, storage format, and design rationale (not repeated here).

## 5. Cache design

Lazy-compute-then-persist, reused as a pre-warm driver (`Training/reporting/warm_cache.py`) via the same function — one mechanism, two invocation modes, per `docs/REPORTING_CACHE.md`'s Precompute strategy section.

## 6. Cache key

`(model_name, model_version, split, video_name)`, plus a `window_starts` consistency check at read time. Full detail: `docs/REPORTING_CACHE.md`.

## 7. Stored artifacts

`window_starts`, `phase_probabilities` (T,K), `next_phase_distribution` (T,K), provenance fields. Duration, timestamps, and ground truth deliberately not cached (all already cheap and/or not trajectory-dependent — see `docs/REPORTING_CACHE.md`). Embeddings never stored.

## 8. Invalidation

Automatic on any mismatch (version/video/split/`window_starts`) or a corrupted/incomplete cache file — always a clean recompute-and-overwrite, never a silent reuse. Verified directly (§12).

## 9. Concurrency

Per-cache-key `fcntl.flock()` around the compute-and-write critical section (blocks a second concurrent request rather than duplicating the ~330s work); atomic write (temp file + `os.replace`) so a reader never sees a partial file and a crash never corrupts one. Verified directly (§12) — lock release confirmed via a non-blocking re-acquire immediately after a completed call.

## 10. Before/after benchmark

Measured directly on the server, `Patient_319` (527 windows), one clean run (cache cleared beforehand). Model/trajectory loading are one-time-per-process costs, unaffected by this cache (included for completeness, not because they were ever the bottleneck):

| Operation | Before | Cache miss | Cache hit |
|---|---:|---:|---:|
| Model loading | 0.0060s | 0.0060s (unchanged — not cached, already negligible) | 0.0060s (unchanged) |
| Trajectory loading | 4.8612s | 4.8612s (unchanged — not cached, one-time per process) | 4.8612s (unchanged) |
| Filtering | 1.5589s | included in compute_time below | — (served from the same cache entry) |
| Next-phase distribution | 356.9283s | included in compute_time below | — (served from the same cache entry) |
| **Forward-pass total (filtering + next-phase)** | **358.4872s** | **354.2661s** (wall) | **0.0093s** (wall) |

**Speedup, before → cache hit: 38,373.9x.** The cache-miss total (354.27s) closely matches the raw "before" total (358.49s) — confirming the caching machinery itself adds negligible overhead on the expensive path (the lock + JSON write cost is a few milliseconds against a 350+ second computation). Model/trajectory loading rows are identical across all three columns by construction — this cache only ever touches the forward-pass step, never those two. Only one trajectory benchmarked, per the phase's own "ne pas lancer de benchmark massif" instruction — the result is unambiguous enough that additional trajectories would not change the conclusion, and the mechanism (a generic disk cache) has no reason to behave differently for other videos of similar size.

## 11. Numerical equivalence

Compared array-for-array (`np.array_equal`, exact equality, not just tolerance-based) across three independent computations of the same trajectory:

| Comparison | Result |
|---|---|
| raw (uncached) vs. cache-miss posterior | **True** (exact) |
| raw vs. cache-miss next_phase_distribution | **True** (exact) |
| cache-miss vs. cache-hit posterior | **True** (exact) |
| cache-miss vs. cache-hit next_phase_distribution | **True** (exact) |

The cache is a strict passthrough — bit-identical output whether served fresh or from disk. End-to-end sanity check (`infer()` on a warm cache): `current_phase=tPNf`, `entropy=0.467151`, `duration_available=True` — all internally consistent, no NaN, no anomaly.

## 12. Tests

**existing tests = 287/287, new tests = 12/12, total = 299/299** (confirmed by a full clean run on the server, 6m02s — down from ~24 min pre-cache, since the warm cache from the benchmark run also speeds up the test suite's own real-data integration test). New tests, `Tests/reporting/test_cache.py`: cache absent (`test_cache_absent_initially_is_a_miss`), cache hit on repeat (`test_second_request_is_a_cache_hit_and_does_not_recompute`), numerical identity (`test_cached_result_is_numerically_identical_to_computed_result`), incompatible model version (`test_different_model_version_is_not_reused`), split isolation (`test_different_split_is_not_reused`), `window_starts` invalidation (`test_changed_window_starts_invalidates_cache`), corrupted cache (`test_corrupted_cache_file_is_treated_as_a_miss_not_a_crash`), missing field (`test_cache_missing_required_field_is_treated_as_a_miss`), video isolation (`test_different_video_names_are_independent`), concurrency/lock release (`test_lock_is_released_after_a_completed_call`), atomic-write cleanliness (`test_atomic_write_leaves_no_tmp_file_on_success`), stats reporting (`test_cache_stats_report_load_and_compute_times`) — 12 tests, all against a `FakeModel` that counts real method invocations, isolating cache *mechanics* from model *correctness* (already covered elsewhere). "Missing video" is covered by the pre-existing `trajectory_service`/`inference_service` tests (`KeyError`/`WindowNotFoundError`), not duplicated here. `test_inference_service.py`'s existing synthetic tests updated to isolate the disk cache to a throwaway directory (`tmp_path`) so they never touch the real `Cache/` tree. `test_integration_real_val.py` intentionally uses the real cache location — its assertions on real Val data now also exercise the full disk-cache path end-to-end.

## 13. Integrity verification

Before and after this phase's changes: `Embeddings/resnet18/{train,val}` checksums identical; `Results/evaluation/semi_hmm_weekend_phaseF/`, `Results/evaluation/e1_hmm_k7_sweep/` checksums identical; `Embeddings/resnet18/test/` mtime unchanged, never referenced by any new code (`cache.py`/`warm_cache.py` are split-agnostic at the storage layer — the `split="test"` refusal happens upstream in `trajectory_service`, unchanged and still tested); `git status` shows only the new `Training/reporting/{cache,warm_cache}.py`, modified `Training/reporting/inference_service.py` (the single, minimal, surgical change routing `_forward_pass` through the cache), new `Tests/reporting/test_cache.py`, and a `.gitignore` addition for the new `Cache/` directory — no scientific file touched. Code changes are confined entirely to the serving/reporting/cache layer, as required.

## 14. Limitations

See `docs/REPORTING_CACHE.md`'s Limitations section: local-disk-only, no TTL (correct given one served model per process), full-Val pre-warm is ~9.7h sequential (parallelizable, not implemented), `warm_cache.py` built but not yet wired into any deploy/ops process.

## 15a. Phase 1.6 recommendation (next_phase_distribution() internal optimization)

**IMPLEMENTED (2026-08-25) — see `docs/SEMI_HMM_PHASE_1_6_REPORT.md` for the full validation.** Summary: the exact table-precomputation fix recommended below was applied; `next_phase_distribution()` went from 326.22s to 8.71s (37.4x) on the same real Val trajectory referenced throughout this section, bit-identical output confirmed on synthetic fixtures, real Val data, and through the Phase 1.5 cache. The recommendation text below is preserved as originally written (historical record of the plan before implementation).

**(Historical, pre-implementation text below — not implemented, not started, a recommendation only, per this phase's explicit instruction not to touch the frozen model, at the time this section was first written.)**

**Exact lines concerned** (`Training/evaluation/models/semi_hmm.py`): `next_phase_distribution()` (function starts line 790) calls `dm.log_survival(a)` (line 812), `dm.log_survival(a + 1)` (line 815), and `dm.log_prob(a)` (line 817) — fresh method calls, inside a loop nest that runs these for every `(t, j, a)` triple, roughly `T×K×Dmax` ≈ 2.1M times for `Patient_319`.

**Why this is fixable, with precedent in the same file**: `_explicit_duration_forward_optimized()` (line 705, used by `filtering()`) already solves the *identical* problem for its own recursion — its own docstring (line 716) states: *"duration_models[j].log_prob(a)/log_survival(a) do not depend on t at all... this was the dominant real cost in practice"* — and precomputes a `(K, Dmax)` table (`log_pmf_table`, `log_surv_table`, lines 737-743) once per call, before the main loop, replacing every repeated method call with an O(1) array lookup. `next_phase_distribution()` never received the equivalent treatment — it still calls the duration model's methods directly inside the hot loop, exactly the pattern the sibling method's own comments identify as the dominant cost.

**Proposed strategy** (for a future session, not this one): precompute the same `(K, Dmax)` `log_prob`/`log_survival` tables once at the top of `next_phase_distribution()` (or reuse them if already computed elsewhere in the same call chain — `_explicit_duration_forward_optimized()` already builds them for its own use, likely one call earlier), then replace `dm.log_survival(a)`/`dm.log_prob(a)` inside the loop with `log_surv_table[j, a-1]`/`log_pmf_table[j, a-1]` array lookups. Mechanically the same fix already validated for `filtering()`.

**How to guarantee numerical equivalence** (mandatory, not optional, given this is the frozen scientific model): (1) the existing test suite already has a validated pattern for exactly this — `_explicit_duration_forward_reference()` is deliberately kept as a slow "correctness oracle," never used in production, checked against the optimized path by dedicated equivalence tests in `Tests/evaluation/models/test_semi_hmm.py`; the same pattern applies directly — keep the current `next_phase_distribution()` as the reference, add an optimized variant, and add a test asserting `np.allclose` (or exact equality, since this is a deterministic table-lookup substitution, not a different algorithm) between the two on the existing synthetic test fixtures; (2) additionally re-run this session's own real-data equivalence check (§11 above) against the optimized version before ever considering it a replacement; (3) any such change is scientific-model territory — would need the same explicit authorization discipline used throughout this project's HMM/Semi-HMM branch before touching `Training/evaluation/models/semi_hmm.py` at all.

**Estimated gain**: `filtering()`'s own optimization (same fix, same file) is documented elsewhere in this project's history as a "133x speedup" for the analogous forward-recursion problem. Applying the identical fix to `next_phase_distribution()` would plausibly bring its cost from ~333-357s down to roughly the same order of magnitude as `filtering()` itself (~1.5-1.7s) — a **rough, non-measured estimate**, not a promise, since `next_phase_distribution()`'s loop structure (nested nested loop building `log_continue`/`log_leave` per state, then a further per-state accumulation over `log_A`) is not byte-for-byte identical to `_explicit_duration_forward_optimized()`'s, so the exact achievable speedup would need to be measured after actually implementing and validating the change, not assumed from the sibling method's number.

**Recommendation: YES, worth doing as a future Phase 1.6** — the fix is well-understood, has a working precedent in the same file, a clear equivalence-testing path, and would remove the last remaining cost in the system worth removing (this session's cache already makes the *repeated*-request cost negligible; a Phase 1.6 would additionally shrink the *first*-request/pre-warm cost, which still matters for `warm_cache.py`'s ~9.7h full-Val pre-warm and for the very first user ever hitting a specific video). **Not urgent** — Phase 1.5's cache already retires the product-blocking version of this problem (repeated requests are already fast); Phase 1.6 would only improve the one-time cold cost, a real but lower-priority follow-up, not a blocker for Phase 2 onward.

## 15. Impact on future RAG / LLM / Web App

Every consumer identified in `docs/PRODUCT_ARCHITECTURE.md`/`docs/LLM_ORCHESTRATION.md` (Web App Analyse panel, `get_inference`/`get_next_phase_distribution`/`get_duration_distribution` tools) can now call `GET /videos/{id}/inference/{window}` and receive a response in single-digit milliseconds for any previously-computed video, and in ~350s exactly once per video otherwise — a bounded, predictable, one-time cost instead of an unbounded per-request one. This directly retires the single largest technical risk flagged in `docs/PRODUCT_ARCHITECTURE.md` §13 for Phase 6 (tool calling). The remaining product-relevant follow-up is operational, not architectural: decide when/how `warm_cache.py` runs (deploy-time hook vs. scheduled job vs. on-demand), a Decision Required already flagged in the roadmap, not resolved by this phase.

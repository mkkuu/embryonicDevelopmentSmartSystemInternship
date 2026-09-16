# Phase 1.6 — next_phase_distribution Optimization

Written 2026-08-25, from real code, real tests, and real measurements against the frozen Semi-HMM on the GPU server — nothing here is estimated or recalled from the Phase 1.5 checkpoint's forward-looking recommendation.

## 1. Motivation

`SemiHMMModel.next_phase_distribution()` (`Training/evaluation/models/semi_hmm.py`) costs ~330-357s per video, cold — the single remaining bottleneck in the Reporting pipeline after Phase 1.5's disk cache eliminated the *repeated*-request cost. Phase 1.5 explicitly deferred fixing the underlying computation (out of scope — infrastructure only) and recorded a precise recommendation (`docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md` §15a) to precompute duration-model lookup tables, mirroring a fix already validated elsewhere in the same file. Phase 1.6 implements exactly that recommendation, and only that.

## 2. Original bottleneck

Confirmed by re-reading the real, current code this session (not assumed from the prior checkpoint): `next_phase_distribution()` (pre-Phase-1.6, now preserved verbatim as `_next_phase_distribution_reference()`) is a `T×K×Dmax` nested Python loop. For `Patient_319` (527 windows, `K=15`, `Dmax=268`), that is ≈2.1M inner iterations. Measured directly this session on the real Val trajectory: **326.22s**.

## 3. Root cause

Inside the innermost loop, for every `(t, j, a)` triple, the code called:

- `dm.log_survival(a)` (original line 812)
- `dm.log_survival(a + 1)` (original line 815)
- `dm.log_prob(a)` (original line 817)

fresh, every time — even though none of these three values depend on `t` at all; each is a pure function of `(j, a)`, and `a` only ranges over `1..min(Dmax, t+1)`. For `NegativeBinomialDurationModel`, each such call invokes real `scipy.stats.nbinom` machinery (`logpmf`/`cdf`) — genuine floating-point work, not a cheap accessor — repeated up to `T` times more often than necessary for a given `(j, a)` pair.

## 4. Existing optimization precedent

The identical problem class was already solved once in this file, for a sibling method: `_explicit_duration_forward_optimized()` (used by `filtering()`) precomputes `(K, Dmax)` tables `log_pmf_table`/`log_surv_table` once per call, before its main loop, and reads them via O(1) array lookup inside the hot loop. Its own docstring states this was "the dominant real cost in practice." `filtering()` costs ~1.5s for the same trajectory — the two methods' raw loop-structure difference (vectorized-over-K vs. not) means this precedent is not a guarantee of an identical final speedup for `next_phase_distribution()`, but it is direct evidence that the specific fix (table precomputation) is valid and safe in this codebase.

## 5. Optimization implemented

**Computational only — no algorithm change.** In `Training/evaluation/models/semi_hmm.py`:

1. The pre-Phase-1.6 implementation of `next_phase_distribution()` is preserved **verbatim**, renamed to `_next_phase_distribution_reference()` — the permanent correctness oracle, never deleted, never used in the default path (same discipline as `_explicit_duration_forward_reference()`).
2. A new `_next_phase_distribution_optimized()` implements the exact same recursion. At the top of the function, `log_pmf_table`/`log_surv_table` (shape `(K, Dmax)`) are built once via `dm.log_prob(a)`/`dm.log_survival(a)` for `a=1..Dmax`. Inside the `(t, j, a)` loop, the three repeated method calls are replaced with array lookups: `log_surv_table[j, a-1]`, `log_surv_table[j, a]` (guarded `a < Dmax`, else `_NEG_INF` — algebraically identical to the original's `a+1<=Dmax` guard), and `log_pmf_table[j, a-1]`. Every other line — loop bounds, branches, `logsumexp` calls, term accumulation — is untouched, copied exactly from the reference.
3. The public `next_phase_distribution()` becomes a thin dispatcher: `self._require_fitted()` then `return self._next_phase_distribution_optimized(traj)` — mirroring `forward()`'s own dispatch to `_explicit_duration_forward()`/`_explicit_duration_forward_optimized()`.

**Not changed, confirmed by inspection and by the equivalence results below**: the algorithm, the transition matrix, `dmax`, the duration distribution family/parameters, any other model parameter, calibration, or training. `_explicit_duration_forward_optimized()`, `filtering()`, `predict()`, `predict_k_step_transition_probability()`, and every other method in the file are byte-identical to before this change (only `next_phase_distribution()`'s region of the file was edited).

## 6. Mathematical equivalence

The three replaced quantities (`dm.log_survival(a)`, `dm.log_survival(a+1)`, `dm.log_prob(a)`) are pure, deterministic functions of the already-fitted duration model's fixed state — calling them twice or reading a value precomputed via the identical call produces the same floating-point bits, not merely a numerically close value. No reordering of summation, no change of associativity, no new arithmetic operation was introduced. This predicts **exact (bit-identical) equivalence**, not just tolerance-based equivalence — confirmed below.

## 7. Synthetic tests

Added to `Tests/evaluation/models/test_semi_hmm.py`, following the file's own established `_compare_reference_vs_optimized`-style pattern (§ "Phase C" tests for `_explicit_duration_forward`):

- `test_next_phase_reference_vs_optimized_various_T` — T ∈ {1,2,3,5,8,15}
- `test_next_phase_reference_vs_optimized_various_dmax` — dmax ∈ {1,3,5,10,20}
- `test_next_phase_reference_vs_optimized_various_n_states` — n_states ∈ {2,3,5,6}
- `test_next_phase_reference_vs_optimized_negative_binomial_and_empirical` — both duration families
- `test_next_phase_reference_vs_optimized_various_transition_smoothing` — 3 (alpha, rho) combinations
- `test_next_phase_reference_vs_optimized_dmax_equals_one` — boundary of the `a+1<=Dmax` guard (dmax=1)
- `test_next_phase_reference_vs_optimized_zero_observed_phase_fallback` — the forced-uniform `EmpiricalDurationModel` fallback path (asserts at least one phase actually hits this fallback in the synthetic fixture, so the test is not vacuous)
- `test_next_phase_reference_vs_optimized_empty_trajectory` — T=0
- `test_public_next_phase_distribution_uses_optimized_path_by_default` — confirms the public API dispatches to the optimized path (`np.array_equal`, by construction)

All comparisons use `np.testing.assert_allclose(..., atol=1e-6)` (matching the file's existing convention for the sibling forward-pass tests, which use the same tolerance despite also being deterministic table substitutions) except the last, which uses `np.array_equal` directly since it compares the same call through two different names.

**Result: 9/9 new tests pass, 83/83 total in `test_semi_hmm.py`** (confirmed on the GPU server, `embryo_env`, 46.30s).

## 8. Real Val equivalence

Loaded the frozen model (`Results/evaluation/semi_hmm_weekend_phaseF/model/`, via `Training/reporting/model_loader.get_model()`, never refit) and the real Val trajectory `Patient_319` (527 windows, via `trajectory_service.get_trajectory`), matching the exact trajectory used for Phase 1.5's own benchmark for direct comparability. Compared `_next_phase_distribution_reference(traj)` against `_next_phase_distribution_optimized(traj)`:

| Check | Result |
|---|---|
| `np.array_equal` (bit-identical) | **True** |
| Shapes match | `(527, 15)` == `(527, 15)` |
| Both finite everywhere | True / True |
| Both non-negative everywhere | True / True |
| Row sums (reference) | min=1.00000000, max=1.00000000 |
| Row sums (optimized) | min=1.00000000, max=1.00000000 |
| Public `next_phase_distribution()` vs `_next_phase_distribution_optimized()` direct call | `np.array_equal` **True** |

No divergence — exact equivalence confirmed on real data, not just synthetic fixtures. Per the phase's own instruction, this was checked **before** proceeding to the benchmark/cache steps below.

## 9. Benchmark before/after

Measured directly on the GPU server, `Patient_319` (527 windows), one run (model/trajectory load are one-time-per-process costs, included for completeness, unaffected by this change):

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| filtering | 1.5323s | 1.5323s (unchanged — not touched by this phase) | 1.0x |
| next_phase_distribution | 326.2167s | 8.7140s | **37.4x** |
| total forward pass (filtering + next_phase_distribution) | 327.7490s | 10.2463s | **32.0x** |

**Does not reach filtering()'s own ~1.5s** — as flagged as a real possibility in Phase 1.5's own recommendation (§15a: "the exact achievable speedup would need to be measured... not assumed from the sibling method's number"). The remaining ~8.7s is the Python-level `T×K×Dmax` loop overhead itself (list construction, per-`(t,j)` `logsumexp` calls) — the *duration-model-call* cost (the dominant term, per the original diagnosis) is what this phase removed; `_explicit_duration_forward_optimized()`'s larger 200x+ speedup also vectorizes its inner loop over `K` via numpy, a further restructuring explicitly **not** attempted here, per this phase's own scope constraint ("pré-calculer... puis utiliser des lookup arrays... NE PAS... changer l'algorithme").

## 10. Cache impact

Verified directly on the GPU server (Phase 1.5's `Cache/reporting/` cache, unmodified by this phase):

| Check | Result |
|---|---|
| Cache miss (new optimized code path) vs. reference oracle, `next_phase_distribution` | `np.array_equal` **True** — the cache does not mask a divergence |
| Cache miss compute time | 10.9991s (was ~354s before Phase 1.6, per Phase 1.5's own benchmark) |
| Cache hit vs. cache miss, `next_phase_distribution` | `np.array_equal` **True** |
| Cache hit vs. cache miss, `phase_probabilities` (posterior) | `np.array_equal` **True** |
| `filtering()` direct call vs. cache-miss posterior | `np.array_equal` **True** |
| Cache hit load time | 0.0096s (unchanged from Phase 1.5) |

`Training/reporting/cache.py` required **zero code changes** — `get_or_compute_forward_pass()` already calls the model's public `filtering()`/`next_phase_distribution()` methods, which now transparently dispatch to the optimized path. The test used the real `Patient_319` cache entry: it was backed up before the test, a fresh miss/hit cycle was exercised, equivalence confirmed, and the original entry was restored afterward — no other video's or model's cache entry was touched. `Cache/reporting/semi_hmm/.../val/{Patient_319,Patient_367}` are the only two entries present, both pre-existing from Phase 1.5's own benchmark, unaffected in content.

## 11. Full test suite

**308/308 passed** (299 pre-Phase-1.6 + 9 new), full clean run on the GPU server, **101.33s** — down from Phase 1.5's own 6m02s / earlier 24m18s runs, because `Tests/reporting/test_integration_real_val.py`'s real-data integration test now hits an already-warm disk cache (from this session's own real-Val equivalence work above), not because any test was removed or weakened.

## 12. Integrity verification

Checksums and mtimes re-verified on the GPU server, before and after this phase's changes:

| Artifact | Before | After | Changed? |
|---|---|---|---|
| `Results/evaluation/semi_hmm_weekend_phaseF/model/state.json` | sha256 `1c853e32...` | sha256 `1c853e32...` | **NO** |
| `Results/evaluation/e1_hmm_k7_sweep/best_model/state.json` | sha256 `ec283401...` | sha256 `ec283401...` | **NO** |
| `Results/resnet18_balanced/best_model.pth` | sha256 `be5460d0...` | sha256 `be5460d0...` | **NO** |
| `Embeddings/resnet18/{train,val,test}/manifest.json` | 3 hashes recorded | identical, all 3 | **NO** |
| `Embeddings/resnet18/test/` mtime | 2026-08-19 15:20:38 | 2026-08-19 15:20:38 | **NO** |
| `Training/evaluation/models/semi_hmm.py` | sha256 `1dadba85...` | sha256 `a0376be3...` | **YES — intentional, this phase's own change** |

`git status --short` (local repo) shows the exact same set of tracked/untracked files as at session start — only the content of the two already-untracked files (`Training/evaluation/models/semi_hmm.py`, `Tests/evaluation/models/test_semi_hmm.py`) changed; no new file was introduced, no scratch/temp file left in the tracked tree (temp scripts used for the real-Val/cache checks ran from the remote's repo root outside git tracking and were deleted immediately after use, per established project convention). No screen session or job was left running on the GPU server before or after this phase.

**Scientific model modified: NO** (frozen checkpoint, state.json, embeddings, historical results all byte-identical). **Semi-HMM source code modified: YES** (this phase's explicit, in-scope, validated change). **Test split touched: NO** — `test_no_test_split_referenced_anywhere_in_semi_hmm_module` still passes.

## 13. Limitations

- The new `_next_phase_distribution_optimized()` still has an unvectorized `T×K×Dmax`-shaped Python loop structure (only the duration-model calls were removed, per this phase's explicit scope) — further speedup (e.g. vectorizing the inner `a`-loop over `K` the way `_explicit_duration_forward_optimized()` does) is possible but was deliberately not attempted here, and would be its own future phase with its own equivalence validation.
- Benchmarked on one trajectory (`Patient_319`, 527 windows), per the phase's own "ne pas faire de benchmark massif" instruction — consistent with how Phase 1.5 benchmarked the same single trajectory. The mechanism (a per-`(j,a)` table lookup replacing a per-call computation) has no reason to behave differently in kind for a longer/shorter trajectory, only in degree (roughly linear in `T`).
- `_next_phase_distribution_reference()` remains O(T·K·Dmax) with real per-call scipy overhead — if ever needed again (e.g. as a future oracle for a further optimization), it will still be slow by design; that is its intended role, not a bug.

## 14. Recommendation

**Phase 1.6 = PASS.** The optimization is correct (bit-identical on both synthetic fixtures and real Val data), safe (zero scientific artifact touched, Test never referenced), and delivers a real, substantial, measured gain (37.4x on `next_phase_distribution()`, 32.0x on the total forward pass) — meaningfully improving the one-time cold cost that Phase 1.5's cache could not address (first-ever request per video, and the ~9.7h full-Val `warm_cache.py` pre-warm estimate from Phase 1.5, which would now be expected to drop to roughly its 1/32nd, though this was not re-measured end-to-end this phase and should be treated as a directional expectation, not a re-benchmarked number).

**Next recommended step**: per this phase's own instruction not to auto-launch further roadmap work, no new phase is started. If a next step is wanted, the natural candidates are (a) re-run/re-time `Training/reporting/warm_cache.py`'s full-Val pre-warm now that the cold-compute cost has dropped ~32x, to get a real updated estimate replacing Phase 1.5's ~9.7h figure, or (b) proceed to Phase 2 (data schemas / tool-calling contracts) per `docs/PRODUCT_ROADMAP.md`. Both are Decisions Required for the user, not started here.

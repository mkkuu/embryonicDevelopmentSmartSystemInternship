# HANDOFF_SEMI_HMM.md — Semi-HMM design & architecture reference

Stable reference (mathematical formulation, file map, conventions) — for day-to-day status, active runs, and next actions see `docs/PROJECT_CHECKPOINT_SEMI_HMM.md` instead; this file should not need updating every session.

## 1. Motivation

The classic HMM (`Training/evaluation/models/hmm.py`) encodes phase dwell-time implicitly and geometrically via its self-loop `A[i,i]` — a single decay rate per state. Two independent diagnostics (`Results/evaluation/e1_hmm_duration_heterogeneity_diagnostic/`, `e1_hmm_segment_count_control/`) established that Brier calibration error correlates with phase mean duration even after controlling for the number of independent segments observed (partial Pearson r=-0.7729, p=0.0032, n=13, robust to dropping the two most extreme phases individually and together) — evidence that an explicit duration model is worth testing, not proof that it will fix calibration.

## 2. Mathematical formulation

```
P(S_1:K, D_1:K, O_1:T) = pi(S_1) * P(D_1|S_1) * prod_{k=2}^K [P(S_k|S_{k-1}) * P(D_k|S_k)] * prod_{t=1}^T b_{S(t)}(O_t)
```

- `K` here = number of SEGMENTS in the trajectory (not to be confused with `n_states`, also called K elsewhere — context disambiguates).
- `S(t)` maps window `t` to whichever segment it belongs to.
- 15 states (`DEFAULT_PHASE_NAMES`, imported from `hmm.py`, never duplicated): `tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB`.
- Transitions `P(S_k|S_{k-1}) = A[i,j]`: strictly `j>i` (no self-loop cell exists in `A` at all — duration owns that). Skips `j>i+1` allowed, estimated by segment-level counts smoothed with a Dirichlet prior decaying geometrically in `(j-i)` (rate `transition_decay_rho`, concentration `transition_smoothing_alpha`) — this is exactly the "destination conditional on leaving `i`" half of `HMMModel.fit()`'s own `A` estimation, reused/reproduced faithfully, minus the self-loop hazard scaling (not applicable here).
- Duration `P(D=d|S=s)`: discrete distribution over `d in {1,...,dmax}`. Unit = **windows**, not frames or real elapsed time (matches the whole pipeline's own per-window time-step convention; real elapsed time via `metadata_with_time.csv` remains a reporting-only derived quantity, never the modeling unit, since its raw unit is still unconfirmed — Phase 0/1 of the HMM branch).
- Emission `b_{S_t}(O_t)`: **identical** to the classic HMM's hybrid discriminative Bayes-converted emission. `SemiHMMModel` never reimplements this — it composes an internal `HMMModel` instance (fit once, used only for `._log_emission`) purely to reuse the validated `StandardScaler`+`LogisticRegression`+Bayes-conversion pipeline without duplicating ~30 lines of it.

## 3. Segment / censoring convention

A segment = a maximal run of consecutive windows sharing the same `last_frame_phase` within one trajectory — computed via `HMMModel._segments()` (reused verbatim, never reimplemented; run-length encoding, no monotonicity assumption baked into the code itself — the fact that every phase appears at most once per video on this dataset is an empirical property of the data, confirmed independently this session, not a code limitation).

A segment is `LEFT_CENSORED` iff it is a video's first segment, `RIGHT_CENSORED` iff its last, `BOTH_CENSORED` if both (single-segment video), else `OBSERVED` (`classify_segment_censoring()`). Mirrors Phase 2's `is_first_segment`/`is_last_segment` criterion exactly. **Both censoring directions contribute the same survival likelihood term `P(D>=d_obs)`** in duration model fitting — a deliberate modeling decision made this project (Phase 2 never fit a censored likelihood, only labeled censoring), not a convention copied from anywhere, since none existed to copy: in both directions the true duration is only known to be at least the observed one.

**Structural fact, confirmed on full real Train (492 trajectories)**: `tPB2` (0/400 observed) and `tEB` (0/265 observed) are **permanently** 100% censored — they are the chronologically first/last phase, so every occurrence is necessarily a video's first/last segment. This is not fixable by more data under this recording protocol. `SemiHMMModel.fit()` therefore **forces** an explicit uniform `EmpiricalDurationModel` fallback for any state with zero observed durations, regardless of the globally configured `duration_family` — a censored-only negative-binomial MLE was shown (STEP 2 smoke test) to degenerate to the `dmax` boundary (a false, overconfident claim), whereas uniform honestly represents "no information."

## 4. Duration model hierarchy

`DurationModel` (ABC) — discrete distribution over `{1,...,dmax}`, `probability(dmax)` absorbs the entire tail `P(D>=dmax)` of the underlying family (so probabilities always sum to exactly 1 over the truncated support). `expected_duration()`/`quantiles()` implemented generically once in the base class from `probability(d)`.

- `NegativeBinomialDurationModel`: `D = 1 + D'`, `D' ~ NegBinom(r,p)` in scipy's own convention (`r` = number of successes, NOT failures; `p` = success probability — stated explicitly to avoid the classic parameterization ambiguity). Fit via Nelder-Mead on the censored log-likelihood, deterministic starting point from method-of-moments on observed-only durations. **Known V1 simplification**: an observation genuinely ending at exactly `dmax` is fit identically to one merely clipped down to `dmax` from something larger (both use the survival term) — keeps fitting consistent with the truncated inference-time distribution, at the cost of this one edge-case approximation.
- `EmpiricalDurationModel`: histogram over observed (uncensored) durations only, Laplace-smoothed, durations `>dmax` folded into the `dmax` bucket. Real, documented asymmetry vs. the negative binomial: it never uses censored data at all.

## 5. Forward algorithm (explicit-duration / semi-Markov)

Two internal quantities, both needed, easy to conflate:
- `log_alpha_end[t,j]` = `log P(O_1:t, a segment of state j ends EXACTLY at t)` — uses the exact duration **pmf**. Needed to chain the recursion across segments.
- `log_age_posterior[t,j,a-1]` = `log P(O_1:t, currently in state j, age exactly a, segment not necessarily over)` — uses the **survival** `P(D>=a|j)`, since `t` need not be the segment's last window. `filtering()`/`predict_proba()` marginalize this over age; `predict()`'s `consistency_flag_prob` and `next_phase_distribution()` need the age breakdown directly (age==1 at `t` means a transition just happened between `t-1` and `t`).

Two implementations, both in `semi_hmm.py`, both tested against each other:
- `_explicit_duration_forward_reference()`: correct, simple, `O(T*K^2*Dmax)` — recomputes redundant work, kept permanently as the correctness oracle, never deleted, never used by default.
- `_explicit_duration_forward_optimized()`: same math, `O(T*K^2 + T*K*Dmax)` — precomputes the "entering mass" once per `t` (reused by every later `t'` needing it, instead of recomputed up to `Dmax` times) and precomputes `(K,Dmax)` duration pmf/survival tables once (instead of `T*K*Dmax` scipy calls). **Measured speedup: 133x** (0.1412s/window → 0.0011s/window; full real Val filtering: 48.6s instead of an extrapolated ~107 min). This is the default (`_explicit_duration_forward()` dispatches here).

`_log_emission()` clips the composed `HMMModel`'s emission to the finite `_NEG_INF=-1e10` sentinel (matching `hmm.py`'s own convention) before the prefix-sum cumulative machinery — without this, a literal `-inf` from sklearn's `predict_log_proba` underflow could produce `-inf-(-inf)=nan` in the windowed emission-sum subtraction (discovered via the STEP 2 smoke test's causality-test corruption, never manifested on real data, fixed defensively regardless).

## 6. Public interface (`Model` ABC + Semi-HMM-specific)

`fit(train, val=None)`, `predict(trajectories)` (returns `consistency_flag_prob` = `P(age==1|O_1:t)`, i.e. lag-1 — see below for why this is NOT what `consistency_flag` itself measures), `save(path)`/`load(path)` (JSON `state.json` + a nested `emission_hmm/` subdirectory reusing `HMMModel.save()`'s own format verbatim — documented exception, not a new convention). Plus: `forward()`, `filtering()`/`predict_proba()`, `next_phase_distribution()`, `duration_distribution(state)`, `expected_duration(state)`, `transition_chain(traj)` (diagnostic chain over GROUND-TRUTH segments — NOT a semi-Markov Viterbi decode, out of scope for the infrastructure step).

**`predict_k_step_transition_probability(traj, k)` / `predict_k_step(trajectories, k)`** (added 2026-08-23): `P(S_t != S_{t-k} | O_1:t)`, generalizing `predict()`'s lag-1 event, needed for the same reason `HMMModel` grew its own k-step method on 2026-08-21 — `consistency_flag` is an intra-window event that only lag `k=window_size-1` (7 at real scale) actually matches. Unlike the classic HMM, this model gets an EXACT closed form (not a fresh k-step propagation) for free: since `A` has no self-loop entries (transitions are strictly `j>i`), consecutive segments always differ in state, so `P(S_t != S_{t-k}|O_1:t) == P(age_t <= k|O_1:t)`, readable directly off the existing `log_age_posterior` table. `k=1` reproduces `predict()` bit-for-bit. Validated against independent full-path brute-force enumeration (`Tests/evaluation/models/test_semi_hmm.py::test_predict_k_step_matches_brute_force_general_k`), 8 new tests total. **Phase F used `predict_k_step(val, k=7)` as its headline metric, NOT `predict()`** — using raw `predict()` against `consistency_flag` would have reproduced the same alignment bug already diagnosed for the classic HMM.

## 7. File map

- `Training/evaluation/models/semi_hmm.py` — everything above. Additive; never imports `hmm.py` for modification, only reads `DEFAULT_PHASE_NAMES`/`HMMModel._segments()`/composes an `HMMModel` instance.
- `Tests/evaluation/models/test_semi_hmm.py` — synthetic-data-only, ~60 tests: segmentation, censoring, duration models (sum-to-one, fit, serialization), transition matrix invariants, forward vs. independent brute-force enumeration, reference-vs-optimized equivalence, causality, determinism, save/load round-trip, edge cases (T=1, Dmax=1, Dmax<observed duration, never-observed phase/transition).
- `Results/evaluation/semi_hmm_smoke/` — STEP 2 smoke test artifacts (8 Train/4 Val trajectories).
- `Results/evaluation/semi_hmm_phase_a2_censoring/`, `semi_hmm_weekend_phaseA/`, `semi_hmm_weekend_phaseBC/`, `semi_hmm_weekend_phaseD/` — this session's phase checkpoints.
- Never touches: `hmm.py`, any `Results/evaluation/e1_*` directory, `Embeddings/`, `Data/`.

## 8. What is NOT yet decided (do not assume any of this is settled)

- Final `duration_family`: Phase F (DONE, 2026-08-23) used `negative_binomial` only (per explicit user decision on dmax) — `empirical` has never been run at full scale, still open.
- `dmax` used for Phase F: **268** (observed_max over full real Train, Phase E measured the candidates, user picked this one 2026-08-23) — frozen for Phase F's result, but not necessarily "the" final Semi-HMM protocol value if the branch continues past Phase F.
- **Phase F outcome (2026-08-23)**: Semi-HMM k=7 vs. frozen HMM k=7 on full real Val — AUROC 0.64271 vs 0.63735, Brier 0.14029 vs 0.14072, ECE 0.12307 vs 0.12312. Marginal improvement on all three, NOT a qualitative win — both remain worse than the naive constant-rate baseline on Brier (0.140 vs 0.115). Full detail: `docs/PROJECT_CHECKPOINT_SEMI_HMM.md` §5, `Results/evaluation/semi_hmm_weekend_phaseF/phaseF_report.json`.
- Whether/how to continue past this result (empirical-duration comparison, per-phase diagnostic on the worst-calibrated phases, or conclude the branch as a negative/neutral result like E1/GRU) — explicit user decision required, not decided this session (`docs/PROJECT_CHECKPOINT_SEMI_HMM.md` §10).

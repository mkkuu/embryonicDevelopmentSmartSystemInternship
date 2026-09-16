# Model Comparison

Every model actually built and evaluated in the SciML extension, in the order it was tried. Companion to `docs/SCIENTIFIC_REPORT.md` (full narrative/evidence) — this file is the per-model comparison table plus a shorter narrative. All results Val unless marked **Test**.

## Canonical comparison table

| Approach | Temporal model | Duration | Emission | AUROC | Brier | ECE | Accuracy | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| base_rate | none | — | — | 0.5000 | — | — | 0.8594 | Val, closed |
| identity_dynamics | none (content only) | — | — | 0.7384 | — | — | 0.8593 | Val, closed |
| persistence | fixed (A=I) | — | — | 0.5922 | — | — | 0.8593 | Val, closed |
| linear_ssm | learned linear | — | — | 0.5858 | — | — | 0.8592 | Val, closed |
| gru_forecast | non-linear (residual) | — | — | 0.5292 | — | — | — | **Test**, closed |
| gru_classification | non-linear (direct) | — | — | 0.7494 (0.7353 shuffled) | — | — | — | **Test**, closed (confound) |
| HMM k=1 | discrete, chronological | implicit geometric | unweighted logreg | 0.6096 | 0.1270 | — | — | Val, superseded by k=7 |
| HMM k=7 | discrete, chronological | implicit geometric | unweighted logreg | 0.6374 | 0.1407 | 0.1231 | 0.8315 | Val, **frozen reference** |
| Semi-HMM k=7 | discrete, chronological | explicit (negative binomial) | unweighted logreg | 0.6427 | 0.1403 | 0.1231 | 0.8323 | Val, **frozen reference, current best** |
| HMM k=7 (balanced, alpha=1) | discrete, chronological | implicit geometric | balanced logreg | 0.5516 | 0.1561 | 0.1436 | 0.8138 | Val, closed (Case D) |
| Semi-HMM k=7 (balanced, alpha=1) | discrete, chronological | explicit (negative binomial) | balanced logreg | 0.5669 | 0.1544 | 0.1417 | 0.8186 | Val, closed (Case D) |
| HMM k=7 (alpha=0.5) | discrete, chronological | implicit geometric | partially weighted | 0.5990 | 0.1498 | 0.1343 | 0.8205 | Val, closed |
| Semi-HMM k=7 (alpha=0.5) | discrete, chronological | explicit (negative binomial) | partially weighted | 0.6064 | 0.1499 | 0.1351 | 0.8214 | Val, closed |
| HMM k=7 (alpha=0.75) | discrete, chronological | implicit geometric | partially weighted | 0.5724 | 0.1538 | 0.1404 | 0.8166 | Val, closed |
| Semi-HMM k=7 (alpha=0.75) | discrete, chronological | explicit (negative binomial) | partially weighted | 0.5862 | 0.1525 | 0.1397 | 0.8202 | Val, closed |
| HMM k=7 (alpha=0.25) | discrete, chronological | implicit geometric | partially weighted (best emission ECE) | 0.6206 | 0.1453 | 0.1278 | 0.8256 | Val, closed |
| Semi-HMM k=7 (alpha=0.25) | discrete, chronological | explicit (negative binomial) | partially weighted (best emission ECE) | 0.6252 | 0.1450 | 0.1294 | 0.8271 | Val, closed |

**Do not compare AUROC across the E1/GRU rows and the HMM/Semi-HMM rows as one ranking** — different task formulations (frame-independent-or-recurrent classification vs. causal filtering posterior over a constrained discrete state space) scoring the same `consistency_flag` label. GRU rows are the only **Test**-evaluated rows in the table; everything else is Val-only, never selected on Test.

## Qualitative analysis

- **Best discrimination (Val, HMM/Semi-HMM block)**: Semi-HMM k=7, unweighted (AUROC 0.6427). Best discrimination overall (any block, any split): `identity_dynamics` (Val AUROC 0.7384) — but it is not a temporal model and does not produce a probabilistic chain.
- **Best calibration**: Semi-HMM k=7, unweighted (Brier 0.1403, ECE 0.1231) — marginally ahead of HMM k=7, both well ahead of every emission-reweighted variant. Neither beats the trivial constant-rate Brier baseline (0.1148) in absolute terms.
- **Best for the probabilistic chain**: Semi-HMM k=7, unweighted — the only model in the project with functional `next_phase_distribution`/`duration_distribution`/`expected_duration`/`transition_chain` (see `docs/INFERENCE_SCHEMA.md`).
- **Best overall compromise**: Semi-HMM k=7, unweighted — best or tied-best on every metric among temporal models, plus unique reporting-relevant capability.
- **Model retained for the next phase (CURRENT_PIPELINE_CANDIDATE)**: see dedicated section below.
- **Models kept only as baselines/reference** (never to be extended further without new evidence): `base_rate`, `persistence`, `linear_ssm`, both GRU formulations, HMM k=1, all emission-reweighted variants (alpha∈{0.25,0.5,0.75,1}) — each closed out a specific hypothesis, none is a candidate for further tuning.

## CURRENT_PIPELINE_CANDIDATE

**Semi-HMM k=7** — `Results/evaluation/semi_hmm_weekend_phaseF/model/` (dmax=268, `duration_family='negative_binomial'`, `logreg_class_weight=None`, HMM-side transition alpha=2.0/rho=0.5 reused for the shared emission/transition-destination logic).

**WHY_SELECTED**: best or tied-best on every Val metric among all discrete-state temporal models (AUROC/Brier/ECE); the only model offering the full probabilistic-chain API (`next_phase_distribution`, `duration_distribution`, `expected_duration`, `transition_chain`) needed for reporting/web app/RAG; every alternative reweighting of its shared emission has been tried and closed without improvement (§14 of `SCIENTIFIC_REPORT.md`), so this is not a placeholder pending a better emission — it is the best currently achievable configuration.

**KNOWN_LIMITATIONS**: shared emission is the dominant, unresolved bottleneck — per-phase accuracy on t3/t5/t6/tPNf remains poor (3-17% emission-only); absolute calibration (Brier/ECE) is worse than a trivial constant-rate predictor; tPB2/tEB durations are structurally inestimable; Semi-HMM's edge over plain HMM is real in direction but not trajectory-significant (n=106, p=0.29); Val-only validation, never Test-confirmed; emission independence assumption known-violated at the windowing scale used.

**SUPPORTED_OUTPUTS**: current phase posterior (`filtering`/`smoothing` via the composed HMM), `next_phase_distribution` (duration-aware, one step ahead), `duration_distribution`/`expected_duration`/quantiles for 13/15 phases, `transition_chain` (ground-truth-segment diagnostic), entropy/uncertainty per window (via the composed HMM's `entropy()`).

**UNSUPPORTED_OUTPUTS**: multi-step-ahead chained forecasts beyond one transition (no semi-Markov Viterbi decode implemented — `transition_chain` is a diagnostic over ground-truth segments, not a forward forecast); any reliable duration estimate for tPB2/tEB; any claim validated on Test; any output implying the emission's minority-phase weaknesses have been resolved.

---

## Per-model narrative

## Ladder

```
base_rate → identity_dynamics → persistence → linear_ssm → GRU(forecast/classification)
                                                                    │
                                                          (dynamics: negative, both branches)
                                                                    ▼
                                          HMM(k=1) → HMM(k=7) → Semi-HMM → [emission_balanced] → [emission_weighted_sweep, in progress]
```

---

### base_rate

- **Motivation**: null floor — constant prediction at the training-set transition rate.
- **Architecture**: no learned parameters.
- **Advantage**: uncontestable reference point; AUROC exactly 0.5 by construction, proves the bootstrap/metric pipeline itself is sane.
- **Limit**: carries zero information by design.
- **Result**: AUROC 0.5000.
- **Decision**: kept as the first rung; only `identity_dynamics vs base_rate` is a pre-registered comparison against it.

### identity_dynamics

- **Motivation**: isolate whether embedding content alone (no temporal reasoning) is informative.
- **Architecture**: logistic regression on the window's own standardized embedding; provably order-invariant (dedicated unit test).
- **Advantage**: the strongest single result in the E1/GRU block (AUROC 0.7384) — cheap, simple, robust.
- **Limit**: cannot express any explicitly temporal claim (deliberately, as a control).
- **Result**: AUROC 0.7384 [0.7189,0.7563], +0.2384 vs base_rate, significant.
- **Decision**: becomes the reference rung every dynamics model must beat to justify itself. None of them do (see below).

### persistence

- **Motivation**: minimal temporal-dynamics rung — does the *change* between consecutive windows (fixed identity transition, $A=I$) add anything?
- **Architecture**: logistic regression on $\lVert e_i-e_{i-1}\rVert$.
- **Advantage**: genuinely order-dependent (frame-shuffle Δ−0.070) — a real, checkable temporal signal, just not a useful one for this task.
- **Limit**: worse than content alone.
- **Result**: AUROC 0.5922, −0.1462 vs identity_dynamics, significant, wrong direction.
- **Decision**: negative; motivates trying a *learned* transition before concluding dynamics modelling fails outright.

### linear_ssm

- **Motivation**: does learning the transition matrix (vs. fixing it to identity) rescue linear dynamics?
- **Architecture**: ridge-regressed transition $A,b$ on consecutive within-trajectory embedding pairs, optional PCA latent (dim 16).
- **Advantage**: also implements forecast/imputation, giving it its own native task to be judged on, independent of the classification ladder.
- **Limit**: loses on both fronts — classification ladder (still behind persistence, −0.0064) AND its own forecast/imputation task (loses to a naive "unchanged embedding" baseline on 6/6 measures). In-sample double-dipping bias in transition-classifier calibration, documented not fixed.
- **Result**: AUROC 0.5858; forecast MSE k=1: 0.01453 vs naive 0.00873.
- **Decision**: **clear negative for linear dynamics.** Per the pre-registered blueprint rule, pivots away from RQ2/RQ3 toward testing whether non-linearity specifically was the missing piece — GRU.

### GRU (forecast)

- **Motivation**: the mechanistically cleanest linear-vs-nonlinear test — same residual/forecast structure as `linear_ssm`, non-linear transition function.
- **Architecture**: gated recurrent unit, hidden_dim=32, 1 layer, residual/forecast head.
- **Advantage**: causality/determinism formally proven (unit tests); clean ablation against `linear_ssm`.
- **Limit**: degenerates to predicting "no transition" for every window — worst-performing model in the entire ladder+GRU set.
- **Result (Test)**: AUROC 0.5292 (seed 0), significantly worse than `identity_dynamics` (Cohen's d ≈ −17) and than `linear_ssm`.
- **Decision**: negative, unambiguous.

### GRU (classification)

- **Motivation**: same architecture, direct classification from full-history hidden state — the more powerful, less mechanistically-clean formulation.
- **Architecture**: as above, direct classification head.
- **Advantage**: real Test win over `identity_dynamics` (+0.010 to +0.016, all 3 seeds, significant) — the only model in E1+GRU to beat `identity_dynamics` on Test.
- **Limit**: that win **does not survive frame-shuffle** — shuffled Test AUROC (0.7353) falls below `identity_dynamics`'s order-invariant baseline (0.7384). Confound (more context/regularization from longer input), not real non-linear-dynamics exploitation.
- **Result (Test)**: AUROC 0.7494 real / 0.7353 shuffled (seed 0).
- **Decision**: extends E1's negative to non-linear dynamics. Project pivots from "linear vs non-linear dynamics" to "discrete chronological state with explicit duration" — HMM.

### HMM (k=1)

- **Motivation**: `RESEARCH_BLUEPRINT.md`'s own "plain HMM" baseline, never built in E1 — not gated by E1/GRU's negative result (tests decoding/segmentation, not dynamics content).
- **Architecture**: 15-state chronological HMM, hybrid discriminative emission (logistic regression → Bayes pseudo-likelihood), `j≥i` transition constraint, `predict()`'s lag-1 inter-window event.
- **Advantage**: structured, causal, interpretable decoding; phase-decoding accuracy 46.4% (15-way, chance≈6.7%).
- **Limit**: scored against `consistency_flag`, badly miscalibrated (Brier 0.1270, worse than trivial baseline 0.1148) — diagnosed as a **target-alignment bug** (lag-1 inter-window event ≠ `consistency_flag`'s intra-window event), not a math bug (math proven exact by brute force).
- **Result**: AUROC(k1) 0.6096, Brier 0.1270.
- **Decision**: implement the mathematically correct k-step generalization rather than patch k=1.

### HMM (k=7)

- **Motivation**: `k=window_size-1=7` makes the k-step two-slice joint compute exactly the event `consistency_flag` measures, by a windowing-arithmetic identity.
- **Architecture**: same HMM, new `predict_k_step_transition_probability` (additive, `predict()` untouched), matched by `window_starts` value.
- **Advantage**: fixes ranking (AUROC 0.6096→0.6373); frozen, re-swept, alpha/rho essentially flat (~0.0003 spread) so alpha=2.0/rho=0.5 chosen by min Brier.
- **Limit**: still never beats the trivial constant-rate Brier baseline (0.1148 vs 0.1407) — miscalibration relocated (under- to over-confidence), not removed.
- **Result**: AUROC 0.63735, Brier 0.14072, ECE 0.12312. **Frozen reference for everything downstream.**
- **Decision**: investigate *why* calibration fails (duration? emission?) rather than keep tuning alpha/rho.

### Semi-HMM

- **Motivation**: duration-heterogeneity diagnostic + segment-count control gave a real (partial-correlation-robust), independent duration↔Brier effect — worth testing explicit duration modelling.
- **Architecture**: composes the identical HMM emission/transition-destination logic + per-phase negative-binomial duration model, dmax=268 (zero truncation). `tPB2`/`tEB` forced to a uniform fallback (structurally 100% censored).
- **Advantage**: small, directionally consistent gain over HMM on all three metrics; a genuinely new, functional capability (`next_phase_distribution`, `duration_distribution`, `expected_duration`, `transition_chain`) no other model in the project offers.
- **Limit**: the gain is **not significant at trajectory level** (p=0.29, 58/106 videos favor it — near coin-flip) and ~45x smaller than E1's headline gain. Transition structure is cosine-identical (1.0000) to HMM — 100% of the difference is in duration, and duration turns out not to be the dominant lever either (Murphy decomposition: reliability dominates resolution ~9x for both models, nearly identically).
- **Result**: AUROC 0.64271, Brier 0.14029, ECE 0.12307.
- **Decision**: stop optimizing the duration/transition layer; the emission diagnostic pointed at the shared emission as the real bottleneck — test it directly (emission_balanced) rather than run the also-prepared-but-lower-value Empirical duration family comparison (deliberately not launched, `DECISION.md` §11).

### Balanced emission (`emission_balanced`)

- **Motivation**: emission diagnostic found catastrophic minority-phase emission accuracy (t3=4.3%, t5=3.4%, t6=3.0%) plausibly from unweighted 67x class imbalance — first (and only, until the weighted sweep) time this branch touched the shared emission.
- **Architecture**: identical HMM/Semi-HMM, `logreg_class_weight='balanced'` (sklearn's inverse-frequency reweighting) on the emission LogisticRegression only.
- **Advantage**: large, gate-passing improvement on all 5 target phases (+8 to +20 accuracy points) — confirms class imbalance is a real, fixable driver of minority-phase emission failure.
- **Limit**: reference-phase accuracy collapses (t9+ −23pts) and, critically, HMM/Semi-HMM k=7 AUROC/Brier/ECE all degrade significantly (AUROC −0.08/−0.09, Brier +0.014/+0.015, ECE +0.019/+0.021) — plain `'balanced'` trades calibration for raw accuracy.
- **Result**: HMM balanced AUROC 0.55157 / Brier 0.15613 / ECE 0.14359; Semi-HMM balanced AUROC 0.56688 / Brier 0.15435 / ECE 0.14174.
- **Decision**: **do not adopt** `class_weight='balanced'` as-is. Test whether a moderated, partial reweighting recovers some minority-phase gain without the calibration collapse — `emission_weighted_sweep`.

### Weighted emission sweep (`emission_weighted_sweep`)

- **Motivation**: `emission_balanced` was Case D (too aggressive) — test whether a moderate, partial rebalancing threads the needle.
- **Architecture**: same HMM/Semi-HMM, `w_c(alpha)=(N/(K·N_c))^alpha`, alpha∈{0,0.25,0.5,0.75,1} (alpha=0/1 reuse the two frozen models above, never refit; only 0.25/0.5/0.75 fit fresh). Pre-registered gate (target-phase accuracy gain ≥+0.05, emission Brier/ECE ratio ≤1.15, overall-accuracy drop ≤0.15) decided before any result was seen.
- **Advantage**: revealed a genuine, non-monotonic finding at the emission level — top-label ECE actually *improves* over baseline at alpha≈0.25-0.5 (0.114-0.116 vs 0.121) — a real local optimum invisible in the accuracy/Brier columns.
- **Limit**: at the level that actually matters (HMM/Semi-HMM k=7), degradation is monotonic and significant from the mildest gate-passing alpha (0.5) onward — no rescue point found among the alphas actually evaluated there. And critically, alpha=0.25 (the best emission-calibrated candidate) never reached k=7 — screened out by the gate's own target-accuracy threshold (+0.0295 < +0.05) — so whether it would help or hurt downstream is genuinely unanswered.
- **Result**: HMM/Semi-HMM AUROC/Brier/ECE at alpha=0.5: 0.5990/0.1498/0.1343 and 0.6064/0.1499/0.1351; at alpha=0.75: 0.5724/0.1538/0.1404 and 0.5862/0.1525/0.1397 — both strictly worse than alpha=0 on every metric, significantly.
- **Decision**: test the one remaining gap (alpha=0.25) directly against HMM/Semi-HMM k=7 — see next entry.

### Alpha=0.25 closing experiment (`emission_alpha025`)

- **Motivation**: close the sweep's one open question — does alpha=0.25's genuinely better emission-level ECE (0.1139 vs 0.1214 baseline) survive propagation to HMM/Semi-HMM, given the gate never let it through?
- **Architecture**: identical to the sweep, single config, propagated unconditionally (no gate — explicitly authorized regardless of outcome).
- **Result**: **it does not survive.** HMM k=7: AUROC 0.63735→0.62057 (−0.0169, p<0.0001), ECE 0.12312→**0.12784** (worse, +0.0048, p<0.0001) — the emission-level ECE gain (−0.0075) is more than reversed once propagated. Semi-HMM: same pattern (AUROC −0.0175, ECE +0.0062, both p<0.0001).
- **Mechanism**: `class_weight` improves the emission's raw 15-way top-label calibration; the HMM/Semi-HMM's k=7 quantity is a different, derived binary event computed via a non-linear Bayes-ratio transformation of that same emission (`b_i(O)=P(S=i|O)/P(S=i)`) — an upstream calibration gain does not imply the downstream transition-probability estimate improves, and here it demonstrably worsens.
- **Decision**: **class-reweighting is CLOSED as a branch.** No alpha in [0.25, 1.0] improves HMM or Semi-HMM over the original, unweighted emission — the original models remain the project's best-validated configuration. Future work: post-hoc calibration (a different mechanism) or proceed to the reporting chain on the original models. Full detail: `Results/evaluation/emission_alpha025/REPORT.md`, `Results/evaluation/emission_weighted_sweep/REPORT.md`, `docs/HANDOFF_EMISSION_WEIGHTED.md`.

---

## Reading this table

Do not treat the E1/GRU AUROC column and the HMM/Semi-HMM AUROC column as one ranking — they are different tasks (frame-independent-or-recurrent classification vs. causal filtering over a constrained discrete state space) scoring the same label. Within each block, only the pre-registered/gated comparisons documented above are statistically load-bearing.

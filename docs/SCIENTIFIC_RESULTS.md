# Scientific Results

Primary scientific reference for the SciML research extension (`Training/embeddings/`, `Training/evaluation/`). Consolidated 2026-08-24 from `docs/HANDOFF.md`, `docs/RESEARCH_BLUEPRINT.md`/`BLUEPRINT_SUMMARY.md`, `docs/PROJECT_STATE.md`, `docs/HMM_RESEARCH_PLAN.md`, `docs/PROJECT_CHECKPOINT_SEMI_HMM.md`, `docs/HANDOFF_EMISSION_BALANCED.md`, and every `Results/evaluation/*` artifact referenced below (server-side, read directly, not recalled from memory). Supersedes nothing — `PROJECT_STATE.md` remains the day-to-day working log; this file is the stable synthesis for the write-up.

## 1. Dataset

Embryo time-lapse image sequences, ResNet18 classifier embeddings (512-dim), extracted via `Training/embeddings/build_cache.py` from checkpoint `Results/resnet18_balanced/best_model.pth` (SHA-256 `be5460d0712d47e916589aa0a230a4132ce8f1d76c682b6e66a21dde1ffc5704` — **this is the sole surviving checkpoint**; the original `Results/resnet18/` was lost in a 2026-07-24 incident, see `docs/REPRODUCIBILITY.md`). `focal_type=F0`, `image_size=224`, `window_size=8`, `stride=1`, `mode=image_seq`.

Splits (window-consistency-filtered, `Embryo_Transition_Dataset`): Train 492 videos / 196,934 windows, Val 106 videos / 43,339 windows, Test 106 videos / 42,519 windows. State space: 15 chronological phases (`tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB`), identical `phase_to_index` mapping confirmed across all three splits.

**Test = LOCKED** for this entire branch except one pre-registered, explicitly authorized exception: the GRU branch's Step 6B/6C Test evaluation (§4). Every other experiment below is Val-only.

## 2. Experimental protocol

Shared infrastructure (`Training/evaluation/`): `Model` interface (`fit`/`predict`/`save`/`load`, JSON-serialized), `Trajectory`/`TrajectoryPrediction` (order-preserving, window-aligned), `metrics.py` (accuracy/precision/recall/F1/AUROC/log-loss at 0.5 threshold, patient-level percentile bootstrap n=1000, Holm-Bonferroni correction across pre-registered comparisons), `calibration.py` (Brier, ECE via 10-bin reliability diagrams). Target task: `consistency_flag` (binary, 1 iff a window's first and last frame differ in phase) except where noted (HMM/Semi-HMM's raw phase decoding is a separate, secondary quantity). Validity control used by every dynamics-claiming experiment: frame-shuffle sanity check (window order scrambled; a result that reverses under shuffling is a confound, not real temporal reasoning).

## 3. Baselines (E1 ladder)

Four-model ladder, real full Train/Val/Test, bootstrap-compared in 3 pre-registered adjacent-rung comparisons (Holm-Bonferroni corrected):

| Model | AUROC (Val) | 95% CI | vs. previous rung | Verdict |
|---|---:|---|---|---|
| `base_rate` | 0.5000 | [0.5000,0.5000] | — | floor, by construction |
| `identity_dynamics` | 0.7384 | [0.7189,0.7563] | +0.2384 vs base_rate, **significant, right direction** | embedding content is informative |
| `persistence` | 0.5922 | [0.5773,0.6077] | −0.1462 vs identity_dynamics, significant, **wrong direction** | fixed-identity dynamics hurts |
| `linear_ssm` | 0.5858 | [0.5722,0.5994] | −0.0064 vs persistence, significant, **wrong direction** | learned linear transition doesn't rescue it |

`linear_ssm` also loses to a naive "unchanged embedding" baseline on its own native forecast/imputation task, 6/6 measures (e.g. 1-step forecast MSE 0.01453 vs naive 0.00873). Frame-shuffle control: `base_rate`/`identity_dynamics` exactly invariant (0.0 difference, proves the shuffle mechanism is correct); `persistence`/`linear_ssm` both genuinely order-dependent (Δ−0.070/−0.057) yet remain behind `identity_dynamics` after shuffling — the negative result is not a task-design artifact.

**Conclusion: clear negative for LINEAR dynamics specifically** (not for temporal information in general — content alone is already informative). Per `RESEARCH_BLUEPRINT.md`'s pre-registered rule ("clear negative on RQ1 → do not build RQ2/RQ3 as scoped; execute pivot directions"), this triggered the GRU branch (non-linear dynamics) rather than an immediate regime-model (RQ2) build.

## 4. GRU

Tests whether non-linearity (not linearity) was the missing ingredient, holding embeddings/task/evaluation fixed. Two formulations, full real Train (492)/Val (106), 3 seeds, `hidden_dim=32, num_layers=1, epochs=50, patience=8, grad_clip_norm=1.0`.

| Formulation | Val AUROC (seed 0/1/2) | Test AUROC (seed 0/1/2) | vs identity_dynamics (Test) | Frame-shuffle (seed 0, Test) |
|---|---|---|---|---|
| `forecast` (residual) | 0.5226/0.5219/0.5224 | 0.5292/0.5290/0.5294 | −0.209, significant, **worse** | 0.5292→0.5246 (Δ−0.0047, negligible — already near chance) |
| `classification` (direct) | 0.7638/0.7783/0.7616 | 0.7494/0.7542/0.7484 | +0.010 to +0.016, significant, **better** | 0.7494→**0.7353** (Δ−0.0141) |

`classification`'s real-Test win reverses under frame-shuffle: shuffled AUROC (0.7353) falls **below** `identity_dynamics`'s order-invariant baseline (0.7384) — the exact signature of a confound (more context/regularization from a longer input), not genuine non-linear-dynamics exploitation. `forecast` is mechanistically the cleaner linear-vs-nonlinear test and simply loses outright, worse even than `linear_ssm`.

**This is the one documented, pre-registered exception to Test being locked** in the entire project (outside the original classifier's own training) — run once, deliberately, as the final confirmatory step of a fully specified protocol.

**Conclusion: extends E1's negative result to non-linear dynamics, for both formulations.** No GRU configuration provides validated evidence of dynamics information beyond `identity_dynamics` (content alone). No further GRU hyperparameter search was undertaken; the project's next pivot changed axis entirely, from "is the dynamics model linear/non-linear" to "is the state discrete/chronological with explicit duration structure" — the HMM/Semi-HMM branch.

## 5. HMM

`HMMModel`: state = phase (`last_frame_phase`, 15 states), hybrid discriminative emission (multinomial LogisticRegression → Bayes pseudo-likelihood, proven valid for ranking/decoding by an exact cancellation argument, not for absolute likelihood), transition `A[i,j]=0` for `j<i` (chronological, non-decreasing — relaxed from a stricter `i→i+1`-only design after Phase 2 found ~11.5% genuine "skip" transitions across 81% of Train videos), Dirichlet-smoothed with geometric decay in `(j-i)`. Not gated by E1/GRU's negative result — it tests decoding/segmentation (RESEARCH_BLUEPRINT.md Part VIII), not dynamics content.

**k=1** (original `predict()`, lag-1 inter-window event): first Train/Val alpha/rho sweep (9 configs) selected `alpha=0.5,rho=0.2` by k=1 AUROC (0.6096), phase-decoding accuracy 46.4%. Scored against `consistency_flag`, Brier=0.1270 — *worse* than the trivial constant-rate baseline (0.1148). **Root cause diagnosed**: `consistency_flag` is an intra-window event (window's own first vs. last frame, spans `window_size=8` raw frames); `predict()`'s lag-1 event is inter-window (~1 raw frame apart) — a 5.2x rate mismatch (13.24% vs 2.54% on Val), proven by brute-force path enumeration to be a target-alignment bug, not a math bug.

## 6. HMM k=7

`predict_k_step_transition_probability(k=7)`: exploits the windowing identity `first_frame_phase(window t) == last_frame_phase(window t-7)` to compute exactly the event `consistency_flag` measures. Re-swept (same 9-config grid, k=7-scored):

| alpha | rho | AUROC k7 | Brier k7 | ECE k7 |
|---:|---:|---:|---:|---:|
| **2.0** | **0.5** | **0.63735** | **0.14072** | **0.12312** |
| (8 other configs) | | 0.6372–0.6374 | 0.1408–0.1410 | 0.1232–0.1234 |

Frozen reference: `Results/evaluation/e1_hmm_k7_sweep/best_model/` (alpha=2.0, rho=0.5, selected by min Brier among AUROC-eligible configs). alpha/rho sweep is essentially flat (~0.0003 spread) — **not the lever that matters**. Baseline (constant-rate) Brier = 0.11483 — the HMM never beats this trivial baseline on Brier. Fixing k alignment repaired ranking (AUROC 0.6096→0.6373) but not calibration (miscalibration relocated from under-confidence at k=1 to over-confidence at k=7 in the mid/high bins).

## 7. Semi-HMM

Motivated by two diagnostics on the frozen HMM: duration-heterogeneity (Brier correlates with mean phase duration, r=−0.715, p=0.006, but NOT with duration CV, r=0.247, p=0.416) and segment-count control (the duration effect survives controlling for segment count via partial correlation, r=−0.773, p=0.0032, n=13 — a real, independent effect, not a data-volume artifact).

`SemiHMMModel`: composes the identical `HMMModel` emission/transition-destination logic (never duplicated) + explicit per-phase duration model, `duration_family='negative_binomial'`, `dmax=268` (`observed_max`, zero truncation — chosen since cost was never the binding constraint at any candidate dmax). `tPB2`/`tEB` (first/last chronological phase, 0 observed interior segments each — structurally 100% censored) get a forced uniform-empirical fallback regardless of configured family.

Full Train(492)/Val(106) fit+eval (Phase F): AUROC=0.64271, Brier=0.14029, ECE=0.12307 (baseline Brier 0.11483, same Val).

| metric | HMM k=7 | Semi-HMM k=7 | Δ |
|---|---:|---:|---:|
| AUROC | 0.63735 | 0.64271 | +0.00536 |
| Brier | 0.14072 | 0.14029 | −0.00043 |
| ECE | 0.12312 | 0.12307 | −0.00005 |

Bootstrap patient-level (n=1000): gain is directionally real (Wilcoxon p<0.001) but **not significant at trajectory level** (n=106, p=0.29, 58/106 videos favor Semi-HMM — near coin-flip) and ~45x smaller than E1's `identity_dynamics` gain. Transition-structure cosine similarity between the two models: **1.0000** across all 14 non-terminal states — 100% of the difference between HMM and Semi-HMM is in duration representation, none in transitions. Murphy decomposition (Brier = reliability − resolution + uncertainty): reliability dominates resolution ~9x for **both** models, near-identical values — both are calibration-bottlenecked in the same place, pointing at the **shared emission**.

**Empirical duration family: never run.** Explicitly scoped and deliberately deferred (`DECISION.md` §11, 2026-08-23) — negative binomial already reproduces the HMM-implicit first moment closely for 13/15 phases, `tPB2`/`tEB`'s fallback is independent of the family choice, and neither prior diagnostic gave reason to expect a duration-family change would touch the dominant bottleneck. The emission diagnostic (§9) was judged higher information value per unit cost and was run instead.

## 8. Duration modelling

See §7. Net finding: explicit duration modelling (negative binomial vs. HMM's implicit geometric dwell time) is real but marginal — it does not solve the calibration problem, because the calibration problem is not primarily a duration problem.

## 9. Emission diagnostic

Read-only, both frozen models (confirmed to share the literal same fitted emission — `same_scaler_mean`/`same_logreg_coef`), no training. Emission-only accuracy (zero temporal information) on Val, per phase: t3=4.3%, t5=3.4%, t6=3.0%, tPNf=17.1%, tB=22.0% vs. t9+=66.2%, t8=36.9% on long/common reference phases. HMM/Semi-HMM correct only 10-33% of these emission errors and **agree on 89-99%** of residual mistakes. Duration CV does not predict the gap (tPNf CV≈0.46 ≈ t9+ CV≈0.51 despite 17% vs 66% accuracy); Train sample volume is the best observed predictor (67x range: t9+ 35,747 windows vs. t3 3,450). Embedding separability confirms heavy class overlap for the worst phases (e.g. tPNf: nearest-class inter/intra dispersion ratio 0.112).

**Strongest evidentiary finding of the whole HMM/Semi-HMM branch**: the shared, unweighted multinomial-logistic emission is the dominant, load-bearing bottleneck for both temporal models — not transition structure, not duration.

## 10. Balanced emission

`class_weight='balanced'` on the shared emission LogisticRegression, gated experiment (proceed to HMM/Semi-HMM k=7 only if mean target-phase emission accuracy improves by >+0.05 vs. baseline — **passed**, delta +0.136).

| | Baseline (None) | Balanced |
|---|---:|---:|
| Emission overall accuracy | 0.4639 | 0.3811 |
| Emission target-phase mean accuracy | 0.0996 | 0.2356 |
| HMM k=7 AUROC / Brier / ECE | 0.63735 / 0.14072 / 0.12312 | 0.55157 / 0.15613 / 0.14359 |
| Semi-HMM k=7 AUROC / Brier / ECE | 0.64271 / 0.14029 / 0.12307 | 0.56688 / 0.15435 / 0.14174 |

Target phases (tPNf/t3/t5/t6/tB) all gain +8 to +20 accuracy points; reference phases regress hard (t9+ −23pts, t8 −5pts); overall accuracy drops. All three HMM/Semi-HMM k=7 metrics degrade significantly (bootstrap p<0.0001). **Case D of the pre-registered taxonomy: emission accuracy improves, temporal calibration degrades.** Class imbalance is confirmed as a real driver of poor minority-phase emission, but plain `class_weight='balanced'` is too aggressive to adopt — it trades accuracy for calibration rather than fixing both.

**Follow-up: `emission_weighted_sweep`** (moderated `w_c(alpha)=(N/(K·N_c))^alpha`, alpha∈{0,0.25,0.5,0.75,1}, alpha=0/1 reused from the frozen models above, alpha=0.25/0.5/0.75 freshly fit, gated per pre-registered thresholds). **No intermediate alpha improves HMM/Semi-HMM k=7** — degradation is monotonic and significant from the mildest gate-passing alpha (0.5) onward:

| alpha | HMM AUROC/Brier/ECE | Semi-HMM AUROC/Brier/ECE |
|---:|---|---|
| 0.0 | 0.6374/0.1407/0.1231 | 0.6427/0.1403/0.1231 |
| 0.5 | 0.5990/0.1498/0.1343 | 0.6064/0.1499/0.1351 |
| 0.75 | 0.5724/0.1538/0.1404 | 0.5862/0.1525/0.1397 |
| 1.0 (=balanced) | 0.5516/0.1561/0.1436 | 0.5669/0.1544/0.1417 |

The original, unweighted emission (alpha=0) remains the best-performing configuration on every temporal metric measured in this entire project.

**Closing follow-up (`emission_alpha025`, 2026-08-24)**: the one open thread above was tested directly — alpha=0.25's genuinely better emission-level top-label ECE (0.1139 vs 0.1214 baseline, reproduced exactly) does **not** survive propagation. HMM k=7: AUROC 0.63735→0.62057 (−0.0169, p<0.0001), Brier 0.14072→0.14528 (+0.0046, p<0.0001), ECE 0.12312→**0.12784** (+0.0048, p<0.0001 — worse, not better, despite the improved emission underneath it). Semi-HMM k=7: AUROC 0.64271→0.62518, Brier 0.14029→0.14498, ECE 0.12307→0.12939 — same pattern, all significant. Mechanism: `class_weight` improves the emission's raw 15-way top-label calibration, but the HMM/Semi-HMM's k=7 quantity is a different, derived binary event computed via a non-linear (Bayes-ratio) transformation of that same emission — an emission-level calibration gain does not imply the downstream transition-probability estimate is better calibrated, and here it demonstrably is not. **The class-reweighting branch (`emission_balanced` → `emission_weighted_sweep` → `emission_alpha025`) is now CLOSED** — no tested alpha in [0.25, 1.0] improves HMM or Semi-HMM over the original, unweighted emission. Full detail: `Results/evaluation/emission_alpha025/REPORT.md`, `Results/evaluation/emission_weighted_sweep/REPORT.md`, `docs/HANDOFF_EMISSION_WEIGHTED.md`.

## 11. Global comparison

See `docs/MODEL_COMPARISON.md` for the full per-model table with motivation/architecture/advantage/limit/decision. Metric summary (Val unless marked):

| Model | AUROC | Brier | ECE | Split |
|---|---:|---:|---:|---|
| base_rate | 0.5000 | — | — | Val |
| identity_dynamics | 0.7384 | — | — | Val |
| persistence | 0.5922 | — | — | Val |
| linear_ssm | 0.5858 | — | — | Val |
| gru_forecast (seed0) | 0.5292 | — | — | **Test** |
| gru_classification (seed0) | 0.7494 (0.7353 shuffled) | — | — | **Test** |
| HMM k=1 | 0.6096 | 0.1270 | — | Val |
| HMM k=7 | 0.6374 | 0.1407 | 0.1231 | Val |
| Semi-HMM k=7 (negative binomial) | 0.6427 | 0.1403 | 0.1231 | Val |
| HMM k=7 (emission balanced) | 0.5516 | 0.1561 | 0.1436 | Val |
| Semi-HMM k=7 (emission balanced) | 0.5669 | 0.1544 | 0.1417 | Val |

**Do not compare AUROC across the E1/GRU block and the HMM/Semi-HMM block as if measuring the same thing** — E1/GRU score `consistency_flag` via frame-independent-or-recurrent classification; HMM/Semi-HMM score the exact same target but via a structured, causal filtering posterior over a constrained discrete state space. Both are legitimate but architecturally different predictors of the same label; the ladder comparisons that are statistically valid are the ones actually pre-registered within each branch (E1's 3 adjacent-rung comparisons, GRU vs. identity_dynamics, HMM-vs-Semi-HMM vs. baseline within the HMM branch).

## 12. Scientific conclusions

1. Embedding **content** is informative (identity_dynamics >> base_rate) — the classifier's representation carries real signal.
2. Explicit **dynamics modelling**, linear or non-linear, does not improve on content alone for this task (E1 + GRU, both negative, GRU's apparent Test win is a confound that fails frame-shuffle).
3. A **structured discrete-state** reformulation (HMM) does add real, if modest, value once its target-alignment bug is fixed (k=7) — AUROC 0.61→0.64 vs. content alone's 0.74 is still lower, but this is a genuinely different task formulation (causal filtering posterior vs. classification), not a strictly comparable number.
4. **Explicit duration modelling** (Semi-HMM) adds a further small, directionally-consistent but not trajectory-significant gain over HMM.
5. **The dominant, load-bearing bottleneck across both temporal models is the shared emission**, not transition or duration structure — confirmed by three independent lines of evidence (Murphy decomposition, transition-structure identity, direct emission-only accuracy).
6. **Naively fixing the emission's class imbalance (`class_weight='balanced'`) backfires**: it improves what it targets (minority-phase accuracy) but degrades what matters for downstream reporting (calibration) — a real trade-off, not a free win.

## 13. Limitations

- Every HMM/Semi-HMM/emission-balanced/emission-weighted-sweep result is Val-only (106 trajectories) — none of these numbers are a Test-confirmed generalization estimate.
- HMM/Semi-HMM Brier never beats the trivial constant-rate baseline (0.1148) — both are, in absolute calibration terms, worse than predicting a constant.
- Emission independence (O_t ⊥ everything else | S_t) is known-violated at `window_size=8, stride=1` (consecutive windows share 7/8 frames) — documented, not corrected; inflates confidence in any quantity multiplying emissions across many timesteps (not `predict_k_step`, which is checked exact by brute force, but flagged for `sequence_log_likelihood`).
- Semi-HMM's transition hyperparameters (alpha=1.0/rho=0.3) were never separately swept the way HMM's were — a protocol asymmetry, not an oversight to hide.
- `tPB2`/`tEB` durations are structurally inestimable (100% censored) — any downstream duration reporting must flag these two phases explicitly, never present a number as if informative.
- The original `Results/resnet18/` classifier checkpoint is lost; `Results/resnet18_balanced/best_model.pth` is the sole surviving checkpoint behind every embedding currently cached (see `docs/REPRODUCIBILITY.md`).

## 14. Probabilistic transition chain

See `docs/PROJECT_CHECKPOINT_SEMI_HMM.md` §"Valeur de la chaîne probabiliste" (DECISION.md) for the full prior assessment; summary here.

Available today (Semi-HMM, functional): `next_phase_distribution()`, `duration_distribution()`, `expected_duration()`, `transition_chain()`. Concrete example (`Patient_319`): tPNa duration=54, P(D=observed)=0.019; t3 duration=4, P(D=observed)=0.062 — plausible, consistent with Train distributions.

- **Exploitable today, with guardrails**: 13/15 phases (all except tPB2/tEB) have informative duration distributions. Point pmf values are individually small by construction (spread over up to 268 values) — must be presented as CDF/quantiles ("70% chance of resolving within N windows"), never as raw pmf.
- **Fragile / must be flagged explicitly**: `expected_duration()`/`duration_distribution()` for tPB2 and tEB specifically reflect a forced uniform fallback (0 observed segments each), not an informative estimate — must be marked "duration not estimable (structural censoring)" rather than shown as a number alongside the other 13.
- **Not yet solved**: the underlying emission's per-phase accuracy (§9) directly limits how much to trust `phase_posterior`/`next_phase_probability` for the worst phases (t3/t5/t6/tPNf) — the chain's downstream quantities inherit the emission's own unreliability there.
- **Verdict**: already good enough to produce a real, usable chain for 13/15 phases with the guardrails above; not yet good enough to present without those guardrails, and not improved by the balanced-emission experiment (which degraded calibration further).

## 15. Future work

**Class-reweighting is CLOSED as a branch** (`emission_balanced` → `emission_weighted_sweep` → `emission_alpha025`, 2026-08-24) — every tested alpha in [0.25, 1.0] degrades HMM/Semi-HMM k=7, including the one candidate (alpha=0.25) with genuinely better emission-level calibration. Not decided/launched — two remaining directions identified, evaluated conceptually, neither started: (a) post-hoc probability calibration (Platt/isotonic) applied on top of the original, unweighted emission, or a different emission strategy entirely (richer features, different classifier family) — same overall goal (fix the emission bottleneck, §9) via a different mechanism than reweighting; (b) proceed to build the probabilistic reporting chain on the existing frozen, best-validated (unweighted) models with the guardrails in §14, treating the emission bottleneck as a known, documented limitation rather than a solved problem. Improving the embeddings themselves remains a longer-horizon option not evaluated here.

## 16. Test status

**TEST = LOCKED** for the entire HMM/Semi-HMM/emission-balanced/emission-weighted-sweep branch — zero code paths capable of loading `Embeddings/resnet18/test/` exist anywhere in `Training/evaluation/models/{hmm,semi_hmm}.py`, `run_hmm_evaluation.py`, or any of the emission-rebalancing experiment scripts (grep-verified repeatedly across sessions). The **sole exception in the entire project** is the GRU branch's Step 6B/6C evaluation (§4), run once, deliberately, as a pre-registered confirmatory step — not a precedent for any other branch to follow without equivalent explicit authorization.

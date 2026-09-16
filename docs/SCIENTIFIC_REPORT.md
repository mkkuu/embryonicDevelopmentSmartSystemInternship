# Scientific Report

**Primary scientific reference for the project**, consolidated 2026-08-24. Supersedes `docs/SCIENTIFIC_RESULTS.md` as the main entry point (that file remains, accurate, as a shorter-form companion — not deleted, not contradicted). Sources: every artifact in `Results/evaluation/*` (read directly, not recalled), `docs/HANDOFF.md`, `docs/RESEARCH_BLUEPRINT.md`/`BLUEPRINT_SUMMARY.md`, `docs/PROJECT_STATE.md`, `docs/HMM_RESEARCH_PLAN.md`, `docs/PROJECT_CHECKPOINT_SEMI_HMM.md`, `docs/HANDOFF_EMISSION_WEIGHTED.md`.

## 1. Project objective

Detect developmental-stage transitions in embryo time-lapse image sequences (the original project's classifier task), and — as an additive research extension, never editing the original `Training/`/`WebApplication/` halves — determine whether explicit temporal-dynamics modeling on top of the classifier's embeddings beats frame-independent classification, and whether a structured probabilistic model can support a future transition-forecasting / reporting product.

## 2. Dataset

Embryo time-lapse image sequences, 15 chronological developmental phases (`tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB`). Splits (window-consistency-filtered, `Embryo_Transition_Dataset`, `window_size=8`, `stride=1`): Train 492 videos/196,934 windows, Val 106 videos/43,339 windows, Test 106 videos/42,519 windows. `phase_to_index` confirmed identical across all three splits (Phase 0 check, HMM branch).

## 3. Data preprocessing

`Training/preProcess.py` (original pipeline, one-time): renames raw folders, generates the annotation/split CSVs, produces `Data/Splits/F0.csv`. Never modified by the SciML extension. `Data/embryo_dataset_time_elapsed/` (real inter-frame elapsed time) exists in the raw dataset but is never joined by `preProcess.py` — the HMM branch built its own additive join (`metadata_with_time.csv`, Phase 1) without touching the original CSV.

## 4. Embedding generation

`Training/embeddings/build_cache.py` extracts and caches ResNet18 embeddings (512-dim) per split, deliberately bypassing `Load_data.py`/`ConfigArgs()` (every parameter passed explicitly on the CLI, avoiding a known config-path bug — see `docs/REPRODUCIBILITY.md`). **Checkpoint provenance**: `Results/resnet18_balanced/best_model.pth` (SHA-256 `be5460d0712d47e916589aa0a230a4132ce8f1d76c682b6e66a21dde1ffc5704`, confirmed in `Embeddings/resnet18/*/manifest.json`) — **the sole surviving classifier checkpoint**; the original `Results/resnet18/` was destroyed in a 2026-07-24 incident and never recovered. If this checkpoint is ever lost, the currently-cached embeddings cannot be regenerated to match.

## 5. Baselines (E1 ladder)

Four-model ladder, real full Train/Val/Test, bootstrap-compared in 3 pre-registered adjacent-rung comparisons (patient-level percentile bootstrap n=1000, Holm-Bonferroni corrected).

**`base_rate`** — architecture: constant prediction at the training-set transition rate. Data: full Train/Val/Test. Hyperparameters: none. Metrics: AUROC 0.5000 (by construction). Advantage: uncontestable floor. Limit: zero information. Conclusion: intended null reference.

## 6. Identity / Persistence

**`identity_dynamics`** — architecture: logistic regression on the window's own standardized embedding, provably order-invariant. Metrics: AUROC 0.7384 [95% CI 0.7189,0.7563], +0.2384 vs base_rate (significant). Advantage: strongest single E1 result — cheap, robust, proves embedding content is informative. Limit: expresses no temporal claim by design. Conclusion: the reference rung every dynamics model must beat.

**`persistence`** — architecture: logistic regression on $\lVert e_i-e_{i-1}\rVert$, fixed transition $A=I$. Metrics: AUROC 0.5922 [0.5773,0.6077], −0.1462 vs identity_dynamics (significant, wrong direction). Frame-shuffle: 0.5922→0.5224 (Δ−0.070, genuinely order-dependent, still behind after shuffling). Advantage: a real, checkable temporal signal. Limit: worse than content alone. Conclusion: fixed-identity dynamics hurts; motivates a *learned* transition.

## 7. Linear SSM

**`linear_ssm`** — architecture: ridge-regressed transition $A,b$ on consecutive within-trajectory embedding pairs, optional PCA latent (`latent_dim=16`). Also implements native `forecast()`/`impute_missing_window()`. Data: full Train/Val/Test. Known caveat: in-sample double-dipping bias in transition-classifier calibration (documented, not fixed). Metrics: AUROC 0.5858 [0.5722,0.5994], −0.0064 vs persistence (significant, wrong direction); forecast MSE k=1: 0.01453 vs naive-baseline 0.00873 (loses on 6/6 forecast/imputation measures). Frame-shuffle: 0.5858→0.5290 (Δ−0.057, genuinely order-dependent). Advantage: has its own native, independent evaluation task. Limit: loses on both fronts. **Conclusion: clear negative for linear dynamics specifically** (not dynamics/temporal information in general — content alone is already informative). Per the pre-registered blueprint rule, triggers the GRU branch (non-linear dynamics) rather than an immediate regime-model build.

**E1 validity control** (`run_frame_shuffle_sanity_check.py`): `base_rate`/`identity_dynamics` exactly invariant under shuffling (0.0 difference — proves the mechanism is correct); `persistence`/`linear_ssm` genuinely order-dependent yet remain behind `identity_dynamics` — the negative result is not a task-design artifact.

## 8. GRU

Tests whether non-linearity (not linearity) was the missing ingredient. Two formulations, full Train(492)/Val(106), 3 seeds, `hidden_dim=32, num_layers=1, epochs=50, patience=8, grad_clip_norm=1.0`.

**`forecast` (residual)**: Test AUROC 0.5292/0.5290/0.5294 (seeds 0/1/2) — significantly worse than `identity_dynamics` (Cohen's d ≈ −17) and than `linear_ssm`; worst-performing model in the whole ladder+GRU set; degenerates to predicting "no transition" for every window.

**`classification` (direct)**: Test AUROC 0.7494/0.7542/0.7484 — significantly better than `identity_dynamics` (+0.010 to +0.016, all 3 seeds). **But this win reverses under frame-shuffle**: shuffled Test AUROC 0.7353, falling *below* `identity_dynamics`'s order-invariant baseline (0.7384) — the exact signature of a confound (more context/regularization from a longer input), not genuine non-linear-dynamics exploitation.

This is **the one pre-registered, explicitly authorized exception to Test being locked** in the entire project (outside the original classifier's own training) — run once, deliberately, as the final confirmatory step of a fully specified protocol. **Conclusion: extends E1's negative result to non-linear dynamics, for both formulations.** No further GRU hyperparameter search was undertaken; the next pivot changed axis entirely, to a discrete chronological state with explicit duration structure — HMM.

## 9. HMM

**Motivation**: `RESEARCH_BLUEPRINT.md` Part VIII's "plain HMM" baseline, never built by E1 — not gated by E1/GRU's negative result (tests decoding/segmentation, not dynamics content). **Architecture**: 15-state chronological HMM, hybrid discriminative emission (multinomial LogisticRegression → Bayes pseudo-likelihood — proven valid for ranking/decoding by an exact cancellation argument, not for absolute likelihood), transition `A[i,j]=0` for `j<i` (chronological, non-decreasing — relaxed from a stricter `i→i+1` design after Phase 2 found ~11.5% genuine "skip" transitions in 81% of Train videos), Dirichlet-smoothed with geometric decay in `(j-i)`. **Data**: Train 492/Val 106, Test never loaded (no code path capable of it exists).

**k=1** (`predict()`, lag-1 inter-window event): first sweep (9 configs) selected `alpha=0.5,rho=0.2` by k=1 AUROC (0.6096), phase-decoding accuracy 46.4% (15-way, chance≈6.7%). Scored against `consistency_flag`, Brier=0.1270 — worse than the trivial constant-rate baseline (0.1148). **Root cause**: `consistency_flag` is an intra-window event (spans `window_size=8` raw frames); `predict()`'s event is inter-window (~1 raw frame) — a 5.2x rate mismatch, proven by brute-force path enumeration to be a target-alignment bug, not a math bug.

## 10. HMM k=7

`predict_k_step_transition_probability(k=7)` exploits `first_frame_phase(window t) == last_frame_phase(window t-7)` to compute exactly the event `consistency_flag` measures. Re-swept (same 9-config grid, k=7-scored):

| alpha | rho | AUROC k7 | Brier k7 | ECE k7 |
|---:|---:|---:|---:|---:|
| **2.0** | **0.5** | **0.63735** | **0.14072** | **0.12312** |
| (8 other configs) | | 0.6372–0.6374 | 0.1408–0.1410 | 0.1232–0.1234 |

**Frozen reference** (`Results/evaluation/e1_hmm_k7_sweep/best_model/`): alpha=2.0/rho=0.5, selected by min Brier among AUROC-eligible configs. alpha/rho sweep essentially flat (~0.0003 spread) — not the lever that matters. Baseline (constant-rate) Brier = 0.11483, never beaten. Fixing k alignment repaired ranking (0.6096→0.6373) but not calibration (miscalibration relocated from under- to over-confidence, not removed).

## 11. Semi-HMM

Motivated by two diagnostics on the frozen HMM: Brier correlates with mean phase duration (r=−0.715, p=0.006) but NOT with duration CV (r=0.247, p=0.416); this survives controlling for segment count (partial correlation r=−0.773, p=0.0032, n=13) — a real, independent effect. **Architecture**: composes the identical HMM emission/transition-destination logic + explicit per-phase duration model, `duration_family='negative_binomial'`, `dmax=268` (`observed_max`, zero truncation). `tPB2`/`tEB` (0 observed interior segments each — first/last chronological phase, structurally 100% censored) get a forced uniform-empirical fallback regardless of family.

Full Train/Val fit+eval (Phase F): AUROC=0.64271, Brier=0.14029, ECE=0.12307 (vs. HMM: +0.00536/−0.00043/−0.00005). Bootstrap patient-level: directionally real (p<0.001) but **not significant at trajectory level** (n=106, p=0.29, 58/106 videos favor Semi-HMM — near coin-flip), ~45x smaller than E1's headline gain. Transition-structure cosine similarity to HMM: **1.0000** across all 14 non-terminal states — 100% of the difference is in duration, none in transitions. Murphy decomposition: reliability dominates resolution ~9x for **both** models, nearly identically — both are calibration-bottlenecked in the same place, pointing at the shared emission.

## 12. Duration modelling

See §11. Net finding: explicit duration modelling is real but marginal — it does not solve calibration, because calibration is not primarily a duration problem. **Empirical duration family: never run** — explicitly scoped and deliberately deferred (`DECISION.md` §11, 2026-08-23): negative binomial already reproduces the HMM-implicit first moment for 13/15 phases, `tPB2`/`tEB`'s fallback is family-independent, and no prior diagnostic suggested a duration-family change would touch the dominant bottleneck.

## 13. Emission diagnosis

Read-only, both frozen models confirmed to share the literal same fitted emission. Emission-only accuracy (zero temporal information), Val, per phase: t3=4.3%, t5=3.4%, t6=3.0%, tPNf=17.1% vs. t9+=66.2%, t8=36.9%. HMM/Semi-HMM correct only 10-33% of these errors and **agree on 89-99%** of residual mistakes. Duration CV does not predict the gap (tPNf CV≈0.46 ≈ t9+ CV≈0.51 despite 17% vs 66% accuracy); Train sample volume is the best observed predictor (67x range: t9+ 35,747 windows vs. t3 3,450). **Strongest evidentiary finding of the branch**: the shared, unweighted emission is the dominant, load-bearing bottleneck — not transition, not duration.

## 14. Class weighting experiment

Three-stage, now **closed**, investigation:

1. **`emission_balanced`** (`class_weight='balanced'`, gated on target-phase accuracy gain >+0.05, passed at +0.136): target phases gain +8 to +20 accuracy points, reference phases regress hard (t9+ −23pts), overall accuracy drops (0.4639→0.3811). HMM/Semi-HMM k=7 all three metrics degrade significantly (AUROC −0.08/−0.09, Brier +0.014/+0.015, ECE +0.019/+0.021, p<0.0001). **Case D**: emission improves, temporal calibration degrades.
2. **`emission_weighted_sweep`** (`w_c(alpha)=(N/(K·N_c))^alpha`, alpha∈{0.25,0.5,0.75,1}, pre-registered gate): alphas 0.5/0.75 passed the gate and were evaluated; both show monotonic, significant HMM/Semi-HMM degradation, smaller than alpha=1 but real (e.g. alpha=0.5: AUROC −0.038, Brier +0.009, ECE +0.011, all p<0.0001). **alpha=0.25 — the single best emission-level top-label ECE (0.1139 vs 0.1214 baseline) — was screened out by the gate** (target-phase gain +0.0295 < +0.05 threshold), never reaching k=7 in that run.
3. **`emission_alpha025`** (closing experiment, explicit unconditional authorization): alpha=0.25 propagated to k=7 anyway. Result: **the emission-level ECE gain does not survive** — HMM ECE 0.12312→0.12784 (worse, p<0.0001), AUROC 0.63735→0.62057 (p<0.0001); Semi-HMM same pattern (ECE 0.12307→0.12939, AUROC 0.64271→0.62518). Mechanism: `class_weight` improves the emission's raw 15-way top-label calibration, but the HMM/Semi-HMM's k=7 quantity is a different, non-linearly derived (Bayes-ratio) binary event — an upstream gain does not imply a downstream one.

**Class-reweighting is CLOSED as a branch.** No alpha in [0.25, 1.0] improves HMM or Semi-HMM over the original, unweighted emission.

## 15. Global comparison

See `docs/MODEL_COMPARISON.md` for the full canonical table + qualitative analysis. **Do not compare AUROC across the E1/GRU block and the HMM/Semi-HMM block as the same ranking** — different task formulations (frame-independent-or-recurrent classification vs. causal filtering over a constrained discrete state space) scoring the same label. Within each block, only the pre-registered/gated comparisons documented above are statistically load-bearing. Emission-reweighted variants (Case D and all sweep alphas) are Val-only, same splits as their alpha=0 references, directly comparable to those references and to each other, but not to E1/GRU.

## 16. Current best validated approach

**Semi-HMM k=7, unweighted emission, dmax=268, negative_binomial, frozen at `Results/evaluation/semi_hmm_weekend_phaseF/model/`** — AUROC 0.64271, Brier 0.14029, ECE 0.12307. Marginally ahead of HMM k=7 (same config family, `Results/evaluation/e1_hmm_k7_sweep/best_model/`) on every metric, though the edge is not trajectory-significant. Both remain the best-validated temporal models in the project after exhausting the class-reweighting branch — see §17 for why "best validated" still has real limitations.

## 17. Current limitations

- Every HMM/Semi-HMM/emission-experiment result is Val-only (106 trajectories) — none is a Test-confirmed generalization estimate.
- HMM/Semi-HMM Brier never beats the trivial constant-rate baseline (0.1148) — both are, in absolute calibration terms, worse than predicting a constant.
- The shared emission is confirmed (§13) as the dominant bottleneck, and the most direct fix tried (class reweighting, §14) is now closed without success — the bottleneck remains open.
- Emission independence is known-violated at `window_size=8, stride=1` (7/8 frame overlap between consecutive windows) — documented, not corrected; flagged for `sequence_log_likelihood` specifically (not for `predict_k_step`, checked exact by brute force).
- `tPB2`/`tEB` durations are structurally inestimable (100% censored) — must be flagged explicitly in any downstream use, never presented as an informative number.
- Semi-HMM's transition hyperparameters were never separately swept the way HMM's were (protocol asymmetry, documented not hidden).
- The original classifier checkpoint is lost; `Results/resnet18_balanced/best_model.pth` is the sole surviving link to the currently-cached embeddings.

## 18. Probabilistic transition chain

Available today (Semi-HMM, functional): `next_phase_distribution(traj)`, `duration_distribution(state)`, `expected_duration(state)`, `transition_chain(traj)`. See `docs/INFERENCE_SCHEMA.md` for exact signatures/shapes. Concrete example (`Patient_319`): tPNa duration=54, P(D=observed)=0.019; t3 duration=4, P(D=observed)=0.062 — plausible, consistent with Train distributions.

- **Exploitable today, with guardrails**: 13/15 phases (all except tPB2/tEB) have informative duration distributions. Point pmf values are individually small by construction (spread over up to 268 values) — present as CDF/quantiles, never raw pmf.
- **Fragile, must be flagged explicitly**: tPB2/tEB reflect a forced uniform fallback, not an informative estimate.
- **Not yet solved**: the emission's per-phase accuracy directly limits trust in `next_phase_distribution`/`phase_posterior` for t3/t5/t6/tPNf specifically — the chain inherits the emission's unreliability there, and class-reweighting (the direct attempted fix) is now closed.
- **New finding (2026-08-24, `docs/GLOBAL_MODEL_COMPARISON.md` §4)**: scored directly for the first time, the raw 15-way `phase_probabilities` posterior (`filtering()`) is severely miscalibrated — multiclass Brier/log-loss/ECE are all far worse than the raw emission alone, and worse than a uniform 15-way guess on Brier and log loss, even though ranking quality (AUROC/PR-AUC) improves. Anywhere this posterior is surfaced (including the Reporting API's `current_state.phase_probabilities`), it should be presented as a ranking, not a calibrated probability.
- **Verdict**: already good enough to produce a real, usable chain for 13/15 phases with guardrails; not yet good enough to present without them.

## 19. Future RAG

Not built, not started this pass. See `docs/RAG_DATA_MODEL.md` for the full data-model separation (structured / semantic / experimental knowledge) this report and the underlying artifacts already provide. The chronology in §5-14 above, the negative-result reasoning throughout, and the metric tables are exactly the "experimental knowledge" layer a future RAG would need — already documented, not yet indexed/retrievable.

## 20. Future web application

Not built, not started this pass. See `docs/WEBAPP_DATA_REQUIREMENTS.md` for the full per-element data-source mapping against the two-pane design (timelapse viewer + probabilistic analysis) already specified. `docs/INFERENCE_SCHEMA.md` defines the canonical inference output shape this app would consume.

## 21. Test status

**TEST = LOCKED** for the entire HMM/Semi-HMM/emission-reweighting branch — zero code paths capable of loading `Embeddings/resnet18/test/` exist anywhere in `Training/evaluation/models/{hmm,semi_hmm}.py`, `run_hmm_evaluation.py`, or any emission-rebalancing script (grep-verified repeatedly across sessions, most recently for `emission_alpha025`). The **sole exception in the entire project** is the GRU branch's Step 6B/6C evaluation (§8), run once, deliberately, as a pre-registered confirmatory step — never a precedent for any other branch without equivalent explicit authorization. Confirmed this pass: `Embeddings/resnet18/test/` mtime unchanged throughout every experiment run in this consolidation.

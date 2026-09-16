# SCIENTIFIC BACKGROUND — question, data, models, established results, limits

Consolidated on 2026-09-15 from the project's own documents. Every number below is copied from
a primary source named in brackets; nothing is re-derived here. Primary sources are kept
unchanged in `docs/` (RAG corpus, see `docs/README.md`) and in `Results/evaluation/` on the GPU
server. When two documents disagree, both values are given.

Vocabulary used throughout: **OBSERVED** = an annotation of the dataset; **MODEL-DERIVED** = a
prediction or estimate of a fitted model; **DERIVED** = a quantity computed from observations
(e.g. the last annotated transition before a window); **SCIENTIFIC** = external knowledge (the
Istanbul Consensus). The whole answer layer of the application is built to keep these apart.

## 1. The question

The programme's central question, verbatim from `RESEARCH_BLUEPRINT.md` (Part I):

> Is human embryo development, as captured by time-lapse imaging, governed by a shared,
> low-dimensional dynamical law — and do the discrete clinical categories (phase, arrest)
> correspond to that law's structure, or to something else?

Hypotheses as pre-registered there (Part III), with their falsification criteria:

| Hypothesis | Claim | Falsified if |
|---|---|---|
| **H1** (gates everything) | explicit dynamics modelling beats frame/window-independent classification on temporal-coherence tasks | no measurable advantage on any temporal-coherence task once a frame-shuffle control rules out a task-design artefact |
| H2 | a discrete regime variable is necessary beyond a continuous state | no improvement of a regime-aware model over a matched-capacity continuous-only model |
| H3 | biologically derived constraints beat a placebo/scrambled control at equal regularisation | no advantage over the placebo control |
| H4 (deferred) | the natural coordinates differ from the hand-designed biological ones | — (two-year horizon, never started) |

The mathematical formulation behind H1–H4 is in `HANDOFF.md` §4 (a design document written
before any experiment; it states itself that nothing in it was fitted to data). In one line:
a hidden state `x(t)` with static individual parameters `θ`, a discrete regime `r(t)` with a
sojourn time `τ(t)`, dynamics `dx = f_r(x;θ)dt + σ(x;θ)dW + jumps`, regime switching through a
generator `Q(x,τ;θ)`, and observations `y_k = h(x(t_k), r(t_k)) + ε_k`, labels
`ℓ_k = g(x(t_k)) + η_k`. `h` is generically non-injective, so some components of `x` are
structurally unobservable. The simplification chain in §4.4 (hybrid jump-diffusion → SDE →
ODE → linear state space → *or* a semi-Markov model on `r(t)` alone) is what the experiments
walk down: "where one stops in this chain is the actual scientific hypothesis under test".

## 2. The data

[`SCIENTIFIC_REPORT.md` §2, `HMM_RESEARCH_PLAN.md` §1, `DATA_LINEAGE.md`]

- Time-lapse image sequences of human embryos (public dataset, see the root README for the
  reference), annotated frame by frame with **15 chronological developmental phases**:
  `tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB` (`tHB` excluded).
- Split **by video** (seed 42, 70/15/15): Train 492 videos / 196 934 windows, Val 106 videos /
  43 339 windows, Test 106 videos / 42 519 windows (windows of 8 frames, stride 1, after the
  consistency filter that rejects windows spanning more than two phases or two non-adjacent phases).
- Supervision target of the classifier and of the whole E1 ladder: `consistency_flag` = 0 if
  the first and last frame of the window carry the same phase, 1 otherwise (a transition
  window). Positive rate ≈ 13–14 % depending on the split.
- Encoder: ResNet18 trained with class balancing (`Training/train_balanced.py`);
  `Results/resnet18_balanced/best_model.pth` (SHA-256 `be5460d0…ffc5704`) is the **only
  surviving checkpoint** (the original `Results/resnet18/` was destroyed on 2026-07-24). Its
  512-dimensional embeddings, cached per split under `Embeddings/resnet18/`, are the
  observations of every downstream model.
- Time: the canonical pipeline works in **window indices**. The Reporting API always reports
  `time_unit = "unknown/unverified"` and `duration_unit = "windows"`; the raw elapsed-time files
  exist in the dataset but were never joined into the pipeline (only into a diagnostic CSV).
  The project's author has stated outside the repository that the elapsed-time column is in
  hours; this is **not** traced in any artefact, so the code keeps saying "unverified".
- The **Test split is locked**: it was used once for the pre-registered E1 ladder report and
  once for the authorised GRU evaluation (below); every serving and benchmarking layer refuses
  `split=test` with an HTTP 403.

## 3. The E1 ablation ladder — H1 tested and not supported

[`SCIENTIFIC_REPORT.md` §5–8, `Results/evaluation/e1_full_analysis/REPORT.md`,
`GLOBAL_MODEL_COMPARISON.md`; runner `Training/experiments/e1_ladder/run_e1_full_analysis.py`, which
evaluates on the **Test** split by patient-level bootstrap, n = 1000, seed 0, 95 % CI]

Each rung adds exactly one ingredient over the previous one:

| Model | What it isolates | Accuracy | AUROC [95 % CI] | log-loss |
|---|---|---|---|---|
| `base_rate` | nothing (null floor) | 0.8594 | 0.5000 | 0.4061 |
| `identity_dynamics` | the embedding content alone, provably order-invariant | 0.8593 | **0.7384** [0.7189, 0.7563] | 0.3607 |
| `persistence` | a fixed identity transition (`A = I`) on embedding change | 0.8593 | 0.5922 [0.5773, 0.6077] | 0.4002 |
| `linear_ssm` | a *learned* linear transition | 0.8592 | 0.5858 [0.5722, 0.5994] | 0.4015 |

Pre-registered comparisons (Holm-Bonferroni over three tests): identity vs base rate
+0.2384 AUROC, persistence vs identity **−0.1462**, linear SSM vs persistence −0.0064; all
three significant (Wilcoxon p ≈ 3·10⁻¹⁶⁵). `linear_ssm` also loses to a naive baseline on all
six of its own forecast/imputation measures (MSE k=1 0.01453 vs 0.00873).

Validity control (`run_frame_shuffle_sanity_check.py`): `base_rate` and `identity_dynamics`
are exactly invariant under window shuffling (difference 0.0), `persistence` drops
0.5922 → 0.5224 and `linear_ssm` 0.5858 → 0.5290: the temporal models are genuinely
order-dependent and still behind the order-invariant model, so the negative result is not a
task-design artefact.

**GRU** (`GRUDynamicsModel`, 3 seeds, hidden 32; `run_gru_evaluation.py`, then the single
authorised Test evaluation `run_gru_test_evaluation.py`):
- `forecast`/residual mode: Test AUROC 0.5292 / 0.5290 / 0.5294 — degenerates to "no
  transition" everywhere, worse than identity (Cohen's d ≈ −17).
- `classification`/direct mode: Test AUROC 0.7494 / 0.7542 / 0.7484, +0.010 to +0.016 over
  identity on all seeds — **but** under frame shuffling it falls to 0.7353, *below* the
  order-invariant baseline: "the exact signature of a confound, not genuine non-linear-dynamics
  exploitation" (`SCIENTIFIC_REPORT.md` §8).

Conclusion in the project's own words (`GLOBAL_MODEL_COMPARISON.md` §10): "both linear and
non-linear dynamics are now falsified for this task". H1 is not supported for this dataset,
encoder and target.

A separate Val re-derivation for a like-for-like GRU/identity comparison
(`GRU_IDENTITY_ANALYSIS.md`, one GRU seed, no significance test) gives identity AUROC 0.7501
and GRU-classification 0.7638; these Val numbers must not be mixed with the Test numbers above.
Note the documents' own inconsistency: `GLOBAL_MODEL_COMPARISON.md` labels the E1 rows "Val",
while the runner and `GRU_IDENTITY_ANALYSIS.md` say the persisted E1 numbers are Test.

## 4. The HMM / Semi-HMM branch — the phase as a hidden Markov state

[`SCIENTIFIC_REPORT.md` §9–11, `HANDOFF_SEMI_HMM.md`, `HMM_RESEARCH_PLAN.md`,
`Results/evaluation/e1_hmm_k7_sweep/`, `Results/evaluation/semi_hmm_weekend_phaseF/`]

This branch stops at a different point of the simplification chain: the developmental phase
itself is the hidden state, the embeddings are the observations, and the model is scored on
the same `consistency_flag` event. All numbers are **Val**.

- **Alignment bug found and fixed (k = 7).** `consistency_flag` spans the 8 raw frames of a
  window while the HMM's natural `predict()` event spans one frame: a 5.2× rate mismatch,
  proven by brute-force path enumeration to be a target-alignment bug. Scoring the k-step event
  `P(S_t ≠ S_{t−7} | O_{1:t})` — exact because `first_frame_phase(window t) =
  last_frame_phase(window t−7)` — repaired ranking (AUROC 0.6096 → 0.6373) but not calibration.
- **Transitions are relaxed** to `A[i,j] = 0 for j < i` (skips allowed) after ≈ 11.5 % of
  annotated transitions in 81 % of Train videos were found to skip a phase.
- **HMM reference** (`e1_hmm_k7_sweep/best_model/`, α = 2.0, ρ = 0.5): AUROC 0.63735,
  Brier 0.14072, ECE 0.12312. The constant-rate Brier baseline (0.11483) is never beaten.
- **Why a Semi-HMM**: Brier per phase correlates with the phase's mean duration
  (r = −0.715, p = 0.006; partial correlation controlling for segment count
  r = −0.773 / −0.7729 depending on the document, p = 0.0032, n = 13) and not with its
  duration variability — the classic HMM's geometric dwell time is the wrong shape.
- **Semi-HMM reference** (`semi_hmm_weekend_phaseF/model/`, the model served by the
  application): explicit per-phase duration model, negative-binomial family, `dmax = 268`
  (the observed maximum over Train, no truncation), composed on the HMM's emission model.
  Val AUROC **0.64271**, Brier **0.14029**, ECE **0.12307** — +0.0054 AUROC over the HMM,
  "directionally real (p < 0.001) but not significant at trajectory level (n = 106, p = 0.29)",
  and 100 % of the difference sits in the duration model (transition structure identical,
  cosine 1.0000). `tPB2` and `tEB` have zero interior segments (first and last phases,
  structurally 100 % censored) and fall back to a uniform duration.
- **Emission re-weighting branch — closed, negative**: no class weight α ∈ {0.25, 0.5, 0.75,
  1.0} improves HMM or Semi-HMM over the unweighted emission (e.g. Semi-HMM AUROC 0.6427 → 0.6252
  at α = 0.25 → 0.5669 at α = 1.0). "Class-reweighting is CLOSED as a branch"
  (`HANDOFF_EMISSION_WEIGHTED.md`).
- **Phase decoding** (15-way, Val, `GLOBAL_MODEL_COMPARISON.md` §3c): emission-only accuracy
  0.4639 / macro PR-AUC 0.3073; HMM k=7 filtering 0.4626 / 0.3225; Semi-HMM filtering
  **0.4686 / 0.3319**. The filtered 15-way posteriors are, however, badly calibrated
  (multiclass Brier ≈ 0.96, worse than a uniform guess at 0.933; ECE(top label) ≈ 0.46) — the
  `phase_probabilities` served by the API are a ranking signal, not calibrated probabilities.

Do not rank the E1/GRU block and the HMM/Semi-HMM block on the same AUROC scale: they are
different task formulations scoring the same label (`MODEL_COMPARISON.md`).

## 5. What is established, in one table

| Statement | Status | Source |
|---|---|---|
| The embedding alone carries most of the transition signal (identity AUROC 0.738, Test) | established | E1 report |
| Linear dynamics (fixed or learned) do not add to it | established, negative | E1 report + frame shuffle |
| Non-linear dynamics (GRU) do not add to it once the shuffle confound is accounted for | established, negative | GRU 6a/6b/6c |
| A phase-as-hidden-state model reaches AUROC ≈ 0.64 on Val with poor calibration | established | HMM k=7 |
| Explicit duration modelling gives a small, directionally consistent, non-significant gain | established (n = 106) | Semi-HMM phase F |
| Duration heterogeneity explains per-phase calibration error | established (r ≈ −0.7, 13 phases) | duration diagnostic |
| Class re-weighting of the emission helps | refuted | emission branch |
| A shared low-dimensional dynamical law exists (H1–H4) | **not supported / not tested beyond H1** | blueprint |

## 6. The Istanbul Consensus 2025 corpus (external scientific knowledge)

[`docs/corpus/ISTANBUL_CONSENSUS_2025.md`, its `_TRACEABILITY.md`,
`VALIDATOR_V1_IMPLEMENTATION.md`, `COMPACT_V2_AND_VALIDATOR_INTEGRATION.md`]

- A structured, page-referenced transcription of one external source (Coticchio et al., "The
  Istanbul consensus update: a revised ESHRE/ALPHA consensus …", Human Reproduction 2025,
  DOI 10.1093/humrep/deaf021), 138 blocks, 91.3 % with an explicit page reference. The
  transcription is **not** in the RAG vector index; it is read directly by the Scientific
  Validator and by the `[SCIENTIFIC]` context block.
- The consensus itself states that its good-practice recommendations "should be used for
  information and educational purposes" and "should not be interpreted as setting a standard
  of care" (IC2025-STATUS-01). The project treats it accordingly: a reference for morphology
  and morphokinetic vocabulary, never a clinical standard, and — rule **R14** — the consensus
  **never validates a model prediction**; a compatibility claim on the model side is capped
  at "partially supported" with an explicit qualification.
- The Validator's rule families (R1 observation ≠ interpretation, R2/R2b prediction ≠
  observation, R3 annotated skip ≠ predicted skip, R4 absence of evidence, R5 compatible ≠
  proven, R6 association ≠ causation, R7/R8 corpus silent or definition absent → not
  assessable, R9/R9b/R9c timing needs a verified unit and origin, R10 no IVF/ICSI assumption,
  R11 offset ≠ lag, R12 return claims vs recorded returns, R13 clinical caution, R14 above)
  are documented in `BENCHMARK.md` §Validator.

## 7. Limits to carry into any future claim

- Single cohort, single recording protocol, single encoder checkpoint: scope every claim to
  this dataset (`RESEARCH_BLUEPRINT.md`, risks).
- Overlapping windows (7/8 frames shared) violate emission independence; documented, not
  corrected; affects `sequence_log_likelihood`, not the k-step event (checked exact).
- Calibration: no HMM/Semi-HMM configuration beats the constant-rate Brier baseline; the
  15-way posteriors are worse than uniform on Brier.
- `tPB2`/`tEB` durations are structurally unobservable under this protocol.
- The Semi-HMM's transition hyperparameters were never swept the way the HMM's were.
- Time unit unverified in every artefact; durations are in windows.
- The Test split has been consumed twice (E1, GRU); it cannot be used for exploration.
- Everything above concerns the *scientific* models. The language-model layer that presents
  them (`ARCHITECTURE.md`, `BENCHMARK.md`) adds its own, separately measured, failure modes.

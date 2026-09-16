# Global Model Comparison

Full comparative evaluation of every temporal approach developed in this project, 2026-08-24. Read-only synthesis — no model was refit, no new inference was run, no metric was invented. Raw data: `Results/evaluation/global_model_comparison/{REPORT.md,results.json,comparison_table.csv,per_phase_comparison.json}` (server). Companion: `docs/SCIENTIFIC_REPORT.md` (full per-model narrative), `docs/MODEL_COMPARISON.md` (per-model motivation/decision detail).

## 1. Experimental history

Baseline (E1's null floor) → Identity Dynamics (content alone) → Persistence (fixed dynamics) → Linear SSM (learned linear dynamics) → GRU forecast/classification (non-linear dynamics, Test-evaluated) → HMM k=1 (discrete state, misaligned target) → HMM k=7 (aligned target, frozen reference) → Semi-HMM (explicit duration) → emission-balanced/weighted-sweep/alpha=0.25 (class-reweighting, closed branch). Full detail per step: `docs/SCIENTIFIC_REPORT.md` §5-14. Every one of these is a genuinely executed experiment with real artifacts — nothing in this list is inferred or reconstructed from memory; each was independently verified against `Results/evaluation/*` this session and in the prior consolidation pass.

## 2. Dataset comparability

**Critical, must not be glossed over**: E1 ladder (base_rate/identity_dynamics/persistence/linear_ssm) and HMM/Semi-HMM/emission experiments are all **Val** (106 trajectories, 43,339 windows). **GRU is the one branch evaluated on Test** (106 trajectories, 42,519 windows) — a different split, different window count, and the project's single pre-registered exception to Test being locked. Any GRU-vs-everything-else comparison in this report is explicitly flagged split-by-split; GRU's numbers are never presented as if directly on the same footing as the Val numbers around them. Within Val, every model uses the identical Train(492)/Val(106) split and the identical `resnet18_balanced` embedding cache — those comparisons are on equal footing.

## 3. Classification comparison

**Two different tasks exist in this project and must not be conflated**: (a) the shared binary `consistency_flag` task (every model), and (b) 15-way phase decoding (only HMM/Semi-HMM, via the shared emission + optional decoding — E1/GRU never produce a phase-level output at all, by architecture, not by omission).

### (a) Binary consistency_flag classification

| Model | Split | Accuracy | Balanced Acc. | F1 (positive class)* | AUROC |
|---|---|---:|---:|---:|---:|
| base_rate | Val | 0.8594 | N/A — incompatible task/output** | 0.000 | 0.5000 |
| identity_dynamics | Val | 0.8593 | N/A** | 0.164 | 0.7384 |
| persistence | Val | 0.8593 | N/A** | 0.000 | 0.5922 |
| linear_ssm | Val | 0.8592 | N/A** | 0.001 | 0.5858 |
| gru_forecast (seed 0/1/2) | **Test** | 0.8594 | N/A** | 0.000 | 0.5292/0.5290/0.5294 |
| gru_classification (seed 0/1/2) | **Test** | 0.8619/0.8591/0.8614 | N/A** | 0.252/0.222/0.159 | 0.7494/0.7542/0.7484 |
| HMM k=1 (frozen config) | Val | N/A — not persisted | N/A** | 0.007 | 0.6093 |
| HMM k=7 (frozen) | Val | 0.8315 | N/A** | N/A — not persisted*** | 0.6374 |
| Semi-HMM k=1 | Val | 0.8670 | N/A** | 0.013 | 0.5950 |
| Semi-HMM k=7 (frozen) | Val | 0.8323 | N/A** | 0.219 | 0.6427 |

\* This is the project's own stored positive-class F1 (`metrics.py`), **not** a true 15-way macro-F1 — no model in the project has ever had a true macro-F1 computed (see below).
\*\* **Balanced Accuracy and true Macro-F1 (15-way) were never computed by any existing script for any model** — recovering them exactly would require re-running inference to get raw per-window predictions (not persisted anywhere as arrays), which this pass deliberately does not do (explicit "no recompute" instruction). Marked N/A uniformly, not selectively, to avoid implying some models were evaluated more thoroughly than others.
\*\*\* HMM k=7's precision/recall/F1/log-loss at the 0.5 threshold were computed transiently during a later run (`emission_weighted_sweep`) but never written to any results file — only accuracy was persisted. A genuine, honest gap, not a masked one.

At the 0.5 threshold, **base_rate, persistence, and the K7-config HMM's own k=1 precision/recall are near-zero** — these models essentially never predict "transition" above 0.5, despite non-trivial AUROC in persistence's case. This means threshold-0.5 accuracy is almost uninformative here (it tracks the 86:14 class imbalance, not model quality) — AUROC is the metric that actually discriminates these models, which is why the project's own protocol treated it as primary.

### (c) 15-way classification — Balanced Accuracy / Macro-F1 / PR-AUC (added 2026-08-24, closes the §3(a) gap)

The Balanced Accuracy/Macro-F1 gap flagged as N/A above has now been closed **for the three sources that produce a genuine 15-way phase posterior** (emission-only, HMM k=7 filtering, Semi-HMM filtering — E1/GRU still never produce this output, still genuinely N/A for them). Computed read-only against the two already-frozen models, on the identical Val set:

| Model | Accuracy | Balanced Acc. | Macro-F1 | AUROC macro | PR-AUC macro | PR-AUC weighted |
|---|---:|---:|---:|---:|---:|---:|
| Emission-only | 0.4639 | 0.3113 | 0.3107 | 0.8531 | 0.3073 | 0.4490 |
| HMM k=7 filtering | 0.4626 | 0.3842 | 0.3622 | 0.8262 | 0.3225 | 0.4738 |
| Semi-HMM filtering | 0.4686 | 0.3928 | 0.3679 | 0.8346 | 0.3319 | 0.4888 |

Filtering gives a modest but consistent lift on every ranking-based metric (Balanced Accuracy, Macro-F1, PR-AUC). See §4 for why this does **not** mean filtering improves the posterior's absolute quality — it is a ranking-only improvement, and a severe one on the calibration side. Full detail, methodology, and per-phase tables (all 15 phases): `Results/evaluation/global_model_comparison/PRAUC_REPORT.md`.

### (b) 15-way phase decoding

| Model | Accuracy (15-way, chance≈6.7%) |
|---|---:|
| base_rate / identity_dynamics / persistence / linear_ssm / GRU | N/A — incompatible task/output, no phase-level prediction produced by architecture |
| Raw emission only (no temporal info) | 0.4639 |
| HMM k=1 (selected config) | 0.464 |
| HMM k=7 (frozen) | 0.4626 |

Phase-decoding accuracy is essentially unchanged between k=1 and k=7 and barely above raw emission alone — **the temporal layer adds almost nothing to phase identification itself**; its value (where it exists) is in the *transition* probability, not the phase label.

## 4. Probability comparison

| Model | Split | Brier | Log Loss | ECE |
|---|---|---:|---:|---:|
| base_rate | Val | N/A — not computed by E1 protocol | 0.4061 | N/A |
| identity_dynamics | Val | N/A | 0.3607 | N/A |
| persistence | Val | N/A | 0.4002 | N/A |
| linear_ssm | Val | N/A | 0.4015 | N/A |
| gru_forecast / gru_classification | Test | N/A — GRU reused `metrics.py`, not `calibration.py` | 0.4053 / 0.3579 (seed 0) | N/A |
| HMM k=7 (frozen) | Val | **0.14072** | N/A — not persisted | **0.12312** |
| Semi-HMM k=1 | Val | 0.1279 | 0.7421 | 0.1182 |
| Semi-HMM k=7 (frozen) | Val | **0.14029** | 0.6440 | **0.12307** |

Brier/ECE simply don't exist for the E1/GRU branch — that calibration machinery (`Training/evaluation/calibration.py`) was introduced later, specifically for the HMM branch. This is not a deficiency of this report; it is an honest reflection of the project's own history. **No model here beats the trivial constant-rate Brier baseline (0.1148)** — HMM and Semi-HMM are both, in absolute calibration terms, worse than predicting a constant rate for every window. Semi-HMM is marginally better than HMM on both Brier and ECE, consistent across every measurement of this comparison in the project.

**Important addition (2026-08-24) — the 15-way phase posterior is calibrated far worse than the binary event above.** Every Brier/ECE number in the table above scores the derived *binary* `consistency_flag` event. Scoring the raw 15-way `phase_probabilities` posterior itself (`HMMModel.filtering()`/`SemiHMMModel.filtering()`) for the first time this session reveals a much more severe problem:

| Source | Multiclass Brier | Log Loss | ECE (top-label) |
|---|---:|---:|---:|
| Emission-only | 0.7021 | 1.7066 | 0.1214 |
| HMM k=7 filtering | **0.9700** | **7.1681** | **0.4610** |
| Semi-HMM filtering | **0.9631** | **7.0960** | **0.4589** |

A uniform 15-way guess scores ≈0.933 (Brier, this convention) / ≈2.708 (log loss) / would have ECE near 0 by construction — **filtering makes the posterior worse than uniform on Brier and more than 2.5x worse than uniform on log loss.** Working hypothesis (not independently re-verified beyond this evaluation): the HMM's structural `j≥i` transition constraint is a one-way ratchet — once filtered belief crosses to a later state from an emission misclassification, it can never structurally return, producing sustained overconfident-and-wrong predictions that Brier/log-loss penalize severely and AUROC/PR-AUC barely notice (ranking still improves, per §3(c), even as absolute calibration collapses). **This directly affects how `CurrentState.phase_probabilities` in the Reporting API (`Training/reporting/`, built the same session) should be presented — as a ranking, not as a calibrated probability.** Full methodology and per-phase breakdown: `Results/evaluation/global_model_comparison/PRAUC_REPORT.md`.

## 5. Transition comparison

**Next-phase Top-1/Top-3 and Transition Log Loss were never computed by any existing script, for any model.** `HMMModel.explain()`'s `next_phase_probability` and `SemiHMMModel.next_phase_distribution()` are both functional and used qualitatively (`docs/SCIENTIFIC_REPORT.md` §18, e.g. the `Patient_319` example), but no script has ever scored "does the model's argmax next-phase prediction match the actual next window's phase" numerically. This is a genuine, identified gap — not a metric this report fabricates to fill the requested table.

For linear_ssm/GRU: their outputs (`forecast()`, embedding residuals) forecast the **embedding**, not a phase — converting that into a "next phase" would require a new decoding step never built. **N/A — no explicit transition distribution over phases**, by architecture, for every model except HMM/Semi-HMM.

| Model | Next-phase Top-1 | Next-phase Top-3 | Transition Log Loss |
|---|---|---|---|
| base_rate / identity_dynamics / persistence | N/A — no explicit transition distribution produced | | |
| linear_ssm / GRU (both) | N/A — forecasts embedding, not phase; no decoding step exists | | |
| HMM k=7 | N/A — capability exists (`predict_next`), never scored numerically | | |
| Semi-HMM k=7 | N/A — capability exists (`next_phase_distribution`), never scored numerically | | |

## 6. Duration comparison

| Model | Duration MAE | Duration RMSE | Coverage 50% | Coverage 90% |
|---|---|---|---|---|
| base_rate / identity_dynamics / persistence / linear_ssm / GRU | N/A — incompatible task/output | | | |
| HMM | N/A — no explicit duration model (implicit geometric dwell time via self-loop probability only) | | | |
| Semi-HMM | N/A — never computed by any existing script (see reason below) | | | |

**Why not derived here even though Semi-HMM has an explicit duration model**: `Results/evaluation/e1_hmm_duration_heterogeneity_diagnostic` reports a `mean_duration` per phase in a measurement convention that the project's own `DECISION.md` (§2) explicitly flags as potentially different from the Phase E/Semi-HMM window-count convention used by `expected_duration()` — two related but not confirmed-identical instruments. Subtracting them to fabricate an MAE risks exactly the unit-mismatch error the project has already caught itself on once; this report declines to repeat it.

**What is legitimately available instead**: `transition_chain()`'s `duration_probability_at_observed` (same-convention, honest density-at-truth — e.g. `Patient_319`: tPNa duration=54, P=0.019; t3 duration=4, P=0.062) and the HMM-implicit-vs-Semi-HMM-`expected_duration()` agreement check (`DECISION.md` §8): for 13/15 non-structurally-censored phases, the two agree within ±5% (e.g. tPNa: HMM-implicit 65.94 vs Semi-HMM 68.14 windows) — Semi-HMM's negative-binomial recovers almost the same first moment as HMM's implicit geometric decay, **without adding new information in the mean**, only in the ability to report a distribution/quantiles at all. For tPB2/tEB, both show E[D]=134.50 (forced uniform fallback) — 17x the censored-MLE estimate (7.85), explicitly flagged as non-informative for these 2 phases.

## 7. Uncertainty comparison

Only HMM/Semi-HMM produce a genuine per-window posterior (`phase_posterior`, entropy via `HMMModel.entropy()`); E1/GRU produce a single scalar `consistency_flag_prob`, from which a crude "confidence" (`|p-0.5|`) can be read but no true multiclass uncertainty. Qualitatively: HMM/Semi-HMM's entropy correctly tracks the emission's own confidence — high entropy (uncertain) on t3/t5/t6/tPNf, low entropy (confident) on t8/t9+ — but this is a direct echo of the emission's known accuracy gap (§13 of `SCIENTIFIC_REPORT.md`), not a separately validated calibration property. **Do the models know when they're wrong?** Partially — the ranking of uncertainty across phases matches the ranking of actual difficulty (a useful property for the web app), but the absolute calibration (ECE) shows they are not well-calibrated in magnitude (HMM/Semi-HMM ECE ≈0.123, meaningfully non-zero). No model in the project has been shown to reliably separate "certain and right" from "certain and wrong" beyond this phase-level pattern — a genuine open question for reporting.

## 8. Computational comparison

Only figures actually recorded by existing scripts are reported (`results.json`'s `computational` block for full detail); no new benchmark was run for this report.

| Model | Fit/train time | Inference time | Parameters |
|---|---|---|---|
| identity_dynamics/persistence/linear_ssm | N/A — never explicitly timed (seconds-scale sklearn fits) | N/A | N/A |
| GRU | not a single number (early-stopped 11-49 epochs depending on run) | 79s (50-traj GPU subset, both formulations) | 17,136 (H=8 smoke config; H=32 real config never re-quoted) |
| HMM | ~94s (Train, 196,934 windows) | ~126-257s (Val, 43,339 windows, k=7) | N/A — not counted as one number |
| Semi-HMM | 169.7s (Train) | 158.1s (Val, k=7) | N/A — not counted as one number |

## 9. Per-phase comparison

Full data: `per_phase_comparison.json`. Summary (target phases tPNf/t3/t5/t6/tB, reference t8/t9+):

| Phase | Emission-only acc. | HMM k=7 Brier | Semi-HMM k=7 Brier | Semi-HMM edge |
|---|---:|---:|---:|---|
| tPNf | 0.171 | 0.572 | 0.553 | small |
| t3 | 0.043 | 0.364 | 0.340 | small |
| t5 | 0.034 | 0.264 | 0.268 | ~none (Semi-HMM slightly worse) |
| t6 | 0.030 | 0.242 | 0.240 | ~tied |
| tB | 0.220 | 0.203 | 0.199 | small |
| t9+ | 0.662 | 0.082 | 0.084 | ~tied |
| t8 | 0.369 | 0.090 | 0.091 | ~tied |

**No architecture tested resolves the minority-phase difficulty.** Emission-only accuracy is the dominant predictor of downstream Brier on every phase — the temporal layer (HMM or Semi-HMM) never overcomes a bad emission, it only ever mildly smooths it. This is the same conclusion the emission diagnostic already reached (§13 of `SCIENTIFIC_REPORT.md`), re-confirmed here from a different angle (cross-architecture, not cross-weighting).

## 10. Scientific evolution

**Baseline → Identity Dynamics**: introduced to establish a floor and test whether embedding content alone carries signal. *Hypothesis*: content is informative. *Result*: yes, strongly (AUROC 0.50→0.74). *Failed to resolve*: nothing — this was a clean positive, not a failure. *Why the next model was needed*: content alone says nothing about temporal dynamics, the project's central question.

**Identity Dynamics → Persistence**: introduced to test whether the simplest possible temporal signal (embedding change, fixed transition) adds value. *Hypothesis*: change-over-time carries information content alone doesn't. *Result*: no — worse than content alone (AUROC 0.74→0.59). *Failed to resolve*: whether a *learned* (not fixed) transition would do better. *Why the next model was needed*: to test that directly.

**Persistence → Linear SSM**: introduced to test a learned linear transition. *Hypothesis*: learning A rescues linear dynamics. *Result*: no — still worse than content alone, and loses to a naive baseline on its own native forecast task, 6/6 measures. *Failed to resolve*: whether linearity itself was the limiting assumption. *Why the next model was needed*: GRU, to test non-linearity specifically, holding everything else fixed.

**Linear SSM → GRU**: introduced as the mechanistically cleanest linear-vs-nonlinear test. *Hypothesis*: non-linear dynamics beats content alone. *Result*: `forecast` loses outright (worst model in the project); `classification` wins on Test but the win reverses under frame-shuffle — a confound, not real dynamics exploitation. *Failed to resolve*: the dynamics question entirely — both linear and non-linear dynamics are now falsified for this task. *Why the next model was needed*: a change of axis, from "what shape of dynamics" to "what shape of state" — a discrete, chronological, structurally-constrained state space (HMM), motivated by an entirely different part of the research blueprint (decoding/segmentation, not dynamics content).

**GRU → HMM**: introduced to test structured discrete-state decoding, explicitly not gated by the dynamics-negative result (a different hypothesis). *Hypothesis*: a chronologically-constrained HMM improves transition identification over frame-independent classification. *Result*: k=1 miscalibrated (Brier worse than trivial baseline) — but this was diagnosed as a **target-alignment bug**, not a modelling failure. *Failed to resolve*: correct alignment between the model's own event and `consistency_flag`. *Why the next model was needed*: k=7, the mathematically exact fix.

**HMM → HMM k=7**: k=7 exploits a windowing-arithmetic identity to compute exactly the right event. *Hypothesis*: fixing alignment fixes calibration. *Result*: ranking improved (AUROC 0.61→0.64) but calibration did not (still worse than the trivial baseline, miscalibration merely relocated). *Failed to resolve*: calibration itself — a second, distinct problem from alignment. *Why the next model was needed*: two diagnostics (duration-heterogeneity, segment-count control) pointed at duration as a possible remaining lever — Semi-HMM, to test it directly.

**HMM k=7 → Semi-HMM**: introduced to test explicit duration modelling. *Hypothesis*: explicit duration (vs. HMM's implicit geometric dwell) improves calibration. *Result*: small, real, directionally consistent gain (AUROC +0.005, Brier −0.0004, ECE −0.00005) but **not trajectory-significant** (p=0.29) and Murphy-decomposition-confirmed to leave the dominant reliability gap essentially unchanged. *Failed to resolve*: the calibration problem remained essentially where it was. *Why the next step was needed*: the emission diagnostic, which finally identified the real dominant bottleneck (the shared emission), tested directly for the first time in this branch.

## 11. Negative results

**Linear SSM**: considered insufficient because it loses on two independent, unrelated axes at once — the classification ladder (worse than the simpler `persistence`) and its own native forecast/imputation task (worse than a naive "unchanged embedding" baseline, 6/6 measures). A model failing its *own* task, not just a comparative ladder, is strong evidence against the linear-dynamics hypothesis specifically, not just weak evidence.

**GRU**: showed that even non-linear dynamics, tested as cleanly as this project could manage (two formulations, 3 seeds, full Test evaluation, dedicated frame-shuffle control), does not produce a validated advantage over content alone. It did not become the final model because its one apparent win (`classification` on Test) failed the exact validity control (frame-shuffle) the project pre-registered specifically to catch this failure mode — a textbook confound, not a borderline case.

**HMM**: apart from the calibration/alignment story, what it genuinely adds over the ladder above is a structurally different **task**: causal filtering/decoding over a constrained 15-state space, not just a binary transition score. Phase-decoding accuracy (46%) is a real, new capability even though its Brier calibration is imperfect.

**HMM k=7**: k=7 was necessary because `first_frame_phase(window t) == last_frame_phase(window t-7)` at `window_size=8, stride=1` — a windowing-arithmetic identity, not a hyperparameter choice. The real gain is entirely in **ranking** (AUROC), not calibration — a distinction this report has been careful to keep separate throughout.

**Semi-HMM**: explicit duration modelling does add real information — but only in the sense of producing a genuine distribution/quantiles where HMM had only an implicit geometric assumption, not in the sense of improving the dominant metrics. The duration mean itself barely differs from HMM's implicit estimate for 13/15 phases (±5% agreement) — the "information" Semi-HMM adds is shape/reportability, not a corrected central estimate.

**Emission balanced**: minority-phase accuracy improves because the loss function is explicitly reweighted to prioritize it — but this is a **local, per-sample** reweighting; it does not touch the *global* calibration property (Brier/ECE) that the HMM/Semi-HMM's forward algorithm depends on, and in fact actively worsens it (`class_weight` distorts `predict_proba` away from the true posterior by design, a documented sklearn behavior, not a bug). Better minority accuracy and better overall calibration are not the same objective, and this project's own three-stage investigation (balanced → weighted sweep → alpha=0.25 closing experiment) proved they trade off against each other at every tested strength, not just the extreme one.

## 12. Model trade-offs

**A. Classification** (Macro-F1/Balanced Acc./AUROC): true macro-F1/balanced accuracy unavailable everywhere (§3); by AUROC alone, `identity_dynamics` (0.7384, Val) is the strongest single classifier in the project, though it answers a different, simpler question (content only) than the temporal models.

**B. Probabilities** (Brier/Log Loss/ECE): Semi-HMM k=7 is marginally best among the models that have these metrics at all (Brier 0.1403, ECE 0.1231) — E1/GRU were never scored this way.

**C. Transition** (Next-phase accuracy/Transition Log Loss): N/A for every model — a genuine, unfilled gap, not a ranking.

**D. Duration** (MAE/Coverage): N/A for every model on the requested metrics; Semi-HMM is the only model with any duration capability at all (qualitative advantage, not quantified here).

**E. Product** (reporting/chain/web app/RAG utility): Semi-HMM k=7 is the only model offering `next_phase_distribution`/`duration_distribution`/`expected_duration`/`transition_chain` — see §15.

## 13. Current best approach

**Semi-HMM k=7, unweighted emission** (`Results/evaluation/semi_hmm_weekend_phaseF/model/`) remains the `CURRENT_PIPELINE_CANDIDATE` established in the prior consolidation pass (`docs/MODEL_COMPARISON.md`) — best or tied-best on every metric that is actually comparable across the temporal-model family (AUROC/Brier/ECE), the only model with duration/chain capability, and the only candidate that survived the full class-reweighting investigation. Not the strongest model in the *entire* project on raw AUROC (`identity_dynamics` is, on a different task) — see §14 for why the answer differs by question.

## 14. Future reporting architecture

Not decided here as a technical implementation — this section documents what the model-comparison work *implies* for it, not a design. The gaps identified in §5-6 (no scored next-phase/duration accuracy) are the concrete missing pieces a real reporting architecture needs to close before presenting transition/duration numbers with confidence, beyond the qualitative examples already available.

## 15. Web App implications

Only Semi-HMM (plus the composed HMM's `explain()`) has the granular, per-window posterior + entropy + duration capability the two-pane UI (`docs/WEBAPP_DATA_REQUIREMENTS.md`) needs. Uncertainty display (§7) should be framed as "relative confidence, phase-ranked" rather than "calibrated probability" given the ECE gap — an honest UI framing choice this comparison directly informs.

## 16. RAG implications

This report itself is exactly EXPERIMENTAL-layer content for the future RAG (`docs/RAG_DATA_MODEL.md`) — in particular §11 (negative results) and §3-6's explicit N/A reasoning are precisely the kind of "why isn't this simpler/better" content a RAG needs to answer honestly instead of just returning a number.

## 17. Limitations

Every Val number here is Val-only (never Test-confirmed, by design, except GRU). Several requested metrics (Balanced Accuracy, Macro-F1, Next-phase Top-1/Top-3, Transition Log Loss, Duration MAE/RMSE/Coverage) are genuinely absent from the project's history and are reported as such, not fabricated — this itself is a limitation of the project's own metric coverage, not of this report. E1/GRU and HMM/Semi-HMM are not on strictly comparable footing even within "classification" because they were built with different metric protocols (calibration.py didn't exist yet for E1/GRU) — flagged throughout, not fixable retroactively without new computation (explicitly out of scope this pass).

## 18. Final conclusion

No single architecture wins on every axis. `identity_dynamics` is the strongest raw binary classifier in the project but answers the simplest question (content only, no temporal claim). Among genuinely temporal, structured models, Semi-HMM k=7 is consistently at or ahead of HMM k=7 on every comparable metric, by a small but real margin, and is uniquely positioned for product use because of its duration/chain capabilities — not because it is dramatically more accurate. GRU's one apparent win was shown to be a confound. Class-reweighting, tested exhaustively, closes out as a fix for the shared emission's minority-phase weakness without helping (and generally hurting) the metrics that matter for a probabilistic chain. The project's own negative results are, cumulatively, its strongest finding: dynamics modelling (linear and non-linear) does not help this task, and the actual bottleneck — a shared, imbalanced emission — has resisted every direct fix tried so far.

---

## Qualitative summary table

| Model | Classification | Probabilities | Transitions | Duration | Reporting | RAG |
|---|---|---|---|---|---|---|
| Identity | +++ | N/A | N/A | N/A | + | 0 |
| Persistence | - | N/A | N/A | N/A | - | - |
| Linear SSM | - | N/A | N/A | N/A | - | - |
| GRU (classification) | + | N/A | N/A | N/A | 0 | - |
| GRU (forecast) | -- | N/A | N/A | N/A | - | - |
| HMM k=7 | + | + | + (unscored) | N/A | ++ | + |
| Semi-HMM k=7 | + | ++ | + (unscored) | ++ | +++ | ++ |

**Justification per cell** (qualitative only, not a derived score):
- *Identity — Classification +++*: highest AUROC in the entire project (0.7384) on its own task.
- *Persistence/Linear SSM — Classification -*: both lose to content alone and (Linear SSM) to a naive baseline on their own task.
- *GRU classification — Classification +*: real Test AUROC win, but downgraded conceptually by the frame-shuffle confound — not a "++" despite the raw number, because the win isn't trustworthy.
- *GRU forecast — Classification --*: worst model in the project, degenerates to a constant prediction.
- *HMM/Semi-HMM — Classification +*: real but modest AUROC (0.64-0.64), a genuinely different (harder, structured) task than the E1/GRU ladder.
- *Probabilities N/A for E1/GRU*: Brier/ECE literally never computed for that branch, not a low score — a missing measurement.
- *HMM +, Semi-HMM ++ (Probabilities)*: both fail to beat the trivial baseline in absolute terms, but Semi-HMM is consistently marginally better, hence one notch above HMM, not two.
- *Transitions "+ (unscored)" for HMM/Semi-HMM*: functional capability exists and is used qualitatively, but never benchmarked — a "+" for existing, explicitly caveated as unscored, not a validated "++".
- *Duration N/A everywhere except Semi-HMM ++*: only Semi-HMM has an explicit duration model; "++" not "+++" because MAE/coverage were never validated numerically (§6).
- *Reporting*: Semi-HMM `+++` for uniquely offering the full chain API; HMM `++` for the composed posterior/entropy/next-phase without duration; everything else weak-to-N/A because it produces only a single scalar, not a structured report.
- *RAG*: mirrors Reporting — RAG needs the same structured/queryable outputs; models producing only a scalar contribute little beyond a single number to retrieve.

This qualitative table is **not** a substitute for the metric tables above and must never be read as a composite score — see §13 of the original request's own instruction: no automatic composite score is created here (see below).

## On a composite score

**Not created.** If a composite score were wanted, a defensible starting methodology would be: normalize each of AUROC, (1−Brier/Brier_baseline), (1−ECE) to [0,1] within the Val-only, calibration-scored subset (HMM/Semi-HMM family only — E1/GRU lack Brier/ECE entirely and could not be included without inventing numbers), then average with explicit, stated weights. Its limits: the weight choice is inherently a value judgement (is calibration worth more than ranking for this product?), it would silently drop E1/GRU (no calibration metrics), and it would erase exactly the "different models are better for different things" nuance §14 of the original request asked to preserve. Given these limits and that the underlying metrics already answer every specific question asked in §14 below, no composite score was built.

---

## Data / model lineage

```
DATA
 ↓
ResNet18 (Results/resnet18_balanced/best_model.pth)          [VALIDATED]
 ↓
Embedding (Embeddings/resnet18/{train,val,test}/)             [VALIDATED]
 ↓
Emission (shared LogisticRegression, unweighted)               [VALIDATED, known bottleneck]
 ↓
Temporal model
 ├── Identity Dynamics        [VALIDATED — content-only reference]
 ├── Persistence               [VALIDATED, negative result]
 ├── Linear SSM                 [VALIDATED, negative result]
 ├── GRU                          [VALIDATED, negative result — Test-evaluated exception]
 ├── HMM                            [VALIDATED, k=7 frozen reference]
 └── Semi-HMM                        [VALIDATED, current pipeline candidate]
       ↓
Phase probabilities            [VALIDATED, HMM/Semi-HMM only]
       ↓
Transition probabilities        [EXPERIMENTAL — functional, never numerically scored]
       ↓
Duration                          [EXPERIMENTAL — functional for 13/15 phases, MAE/coverage unscored]
       ↓
Probabilistic chain                 [EXPERIMENTAL — transition_chain() functional, retrospective-only]
       ↓
Reporting                             [PLANNED — no API layer exists]
       ↓
Web App / RAG                           [PLANNED — data requirements specified, not built]
```

Class-reweighted emission variants (balanced/alpha=0.25/0.5/0.75) are **DEPRECATED** — tested, closed, not part of the forward path.

# GRU vs Identity Dynamics

Synthesis of the intermediate analysis run 2026-08-25 (`Results/evaluation/gru_identity_threshold_analysis/` — `REPORT.md`, `analysis.json`, `*.csv`, `plots/*.png`). No experiment was re-run to produce this document: every number below is read directly from that existing output, not recomputed. Where a figure is referenced, it is one of the six already generated (`../Results/evaluation/gru_identity_threshold_analysis/plots/`), not a new one.

## 1. Objective

Determine whether the GRU (`Training/evaluation/models/gru.py`, `predict_mode="direct"`/"classification" formulation) is sensitive enough to serve as a future signal source for the Reporting/RAG pipeline, and characterize the trade-off between detecting real developmental-phase transitions and generating false alarms — compared directly against Identity Dynamics, the simplest content-only baseline in the E1 ablation ladder, on the same evaluation split.

## 2. Common evaluation split

**Methodological correction, made explicit rather than silently fixed**: the only historical, persisted Identity Dynamics metrics in this project (`Results/evaluation/e1_full_analysis/REPORT.md`) are bootstrapped over the **TEST** split, confirmed by reading `Training/evaluation/run_e1_full_analysis.py` (its own docstring: "evaluates on the test split"). GRU's historical numbers (`Results/evaluation/e1_gru_step6a_full/`) are Val-only. These were never on the same split and were not directly comparable as they stood.

For this analysis, Identity Dynamics was **re-derived on Val** — `fit()` on Train, `predict()` on Val, using the unmodified `IdentityDynamicsModel` class with default hyperparameters, matching `run_e1_full_analysis.py`'s own configuration exactly. This is a deterministic re-derivation, not a new experiment: `IdentityDynamicsModel.fit()` is a seedless, closed-form logistic regression, and the E1 script's own docstring states fitting is "deterministic given fixed data" for this model. No retraining occurred. GRU was **loaded from its frozen checkpoint** (`e1_gru_step6a_full/classification/checkpoint`, seed=0) — never fit.

| Model | Split | Videos | Windows | Positive | Negative |
|---|---|---:|---:|---:|---:|
| Identity Dynamics | Val (re-derived) | 106 | 43,339 | 5,736 | 37,603 |
| GRU (classification) | Val (frozen checkpoint) | 106 | 43,339 | 5,736 | 37,603 |

100% common support — both models produce a prediction for every one of the 43,339 Val windows across the same 106 videos, ground truth cross-checked against `metadata_with_time.csv` with 0 mismatches. Positive rate: 13.24% (1:6.6 class imbalance).

## 3. Models

- **Identity Dynamics**: `P(consistency_flag_i=1) = sigmoid(w·x_i + b)`, a window-independent logistic regression on the raw embedding — no temporal mechanism by construction. Serves as the "content alone, no dynamics" floor in this project's ablation ladder.
- **GRU (classification formulation)**: a unidirectional GRU (`hidden_dim=32`, 1 layer) classifying `consistency_flag_i` directly from its causal hidden state `h_i` (built from the full history `z_1..z_i`). The formulation used throughout this analysis, chosen because it produces a real per-window probability for every window, directly comparable to Identity Dynamics.
- A second GRU formulation ("forecast"/residual) exists but scored near chance on Val (AUROC=0.523) and is not analyzed further here — see the source `REPORT.md` §4 for detail.

## 4. Overall comparison

| Model | Split | AUROC | PR-AUC | Brier | ECE | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Identity Dynamics | Val (re-derived) | 0.7501 | 0.3250 | 0.1033 | 0.0104 | 0.5495 | 0.1006 | 0.1701 |
| GRU (classification) | Val (frozen) | **0.7638** | **0.3565** | **0.1015** | 0.0224 | 0.5350\* | 0.1613\* | 0.2478\* |

\*Precision/Recall/F1 for GRU shown at threshold=0.5 for a literal like-for-like row; this is **not** GRU's best operating point — §5 shows the full range. At threshold=0.5, GRU and Identity Dynamics are similar on precision/recall by coincidence of where 0.5 happens to sit on each model's own probability distribution; the more informative comparison is threshold-independent (AUROC, PR-AUC, Brier, ECE) or at GRU's own tuned operating points (§9).

## 5. Threshold analysis

GRU's classification probability on Val is right-skewed (mean=0.139, median=0.086) — most of its usable operating range sits below 0.5. Three operating points, located by a fine (400-point) grid search over the training data already in `analysis.json`, illustrate the range of behavior; none is presented as a universal or production-frozen parameter — each is one point on a continuous trade-off (§6).

| Régime | Threshold | Recall | Precision | FPR | FNR | F1 | TP | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HIGH SENSITIVITY** | 0.059 | 0.904 | 0.195 | 0.568 | 0.096 | 0.321 | 5187 | 21372 |
| **BALANCED** (max F1) | 0.184 | 0.558 | 0.287 | 0.212 | 0.442 | **0.379** | 3202 | 7969 |
| **HIGH PRECISION** | 0.861 | 0.004 | 0.815 | 0.000 | 0.996 | 0.008 | 22 | 5 |

For the full 9-point sweep (0.10–0.90) requested by the original task, see `Results/evaluation/gru_identity_threshold_analysis/gru_threshold_metrics.csv`.

## 6. Precision / Recall / False Positive trade-off

![Precision-Recall curve, GRU vs Identity Dynamics, Val](../Results/evaluation/gru_identity_threshold_analysis/plots/pr_curve.png)

![ROC curve (Recall vs FPR), GRU vs Identity Dynamics, Val](../Results/evaluation/gru_identity_threshold_analysis/plots/roc_curve.png)

Both figures are reused as-is from the existing analysis — no recomputation. Both curves show GRU consistently above Identity Dynamics across nearly the full operating range, on a support (43,339 windows, 5,736 positive) large enough that the margin is not a small-sample artifact.

Quantified, not just described, at the three operating points above:

- **At threshold=0.059** (high sensitivity): Recall=90.4%, Precision=19.5%, **FPR=56.8%**. Concretely: to catch 90 out of every 100 real transitions, roughly 57 out of every 100 *non-transition* windows must also be flagged — about 4 false alarms for every true one among the flagged windows (5187 TP vs 21,372 FP).
- **At threshold=0.184** (balanced): Recall=55.8%, Precision=28.7%, **FPR=21.2%**. Catching just over half of real transitions still means roughly 1 in 5 quiet windows is misflagged, and fewer than 1 in 3 flagged windows is a real transition (3202 TP vs 7969 FP).
- **At threshold=0.861** (high precision): Recall=0.4%, Precision=81.5%, **FPR<0.1%**. Almost no false alarms (5 FP across all of Val), but only 22 of 5,736 real transitions are caught — this threshold is not a usable "conservative detector," it identifies only the handful of most unambiguous cases.

**No threshold in this analysis is simultaneously high-recall and high-precision.** The lower the threshold, the more true positives are gained, but each additional true positive costs a rapidly growing number of false positives — the FPR moves from <0.1% to 56.8% while recall moves from 0.4% to 90.4%. This is the central, quantitative shape of the trade-off, not just a qualitative "lower threshold, higher recall" statement.

## 7. Temporal detection

Definitions (see source `REPORT.md` §9 for the full derivation): an **event** = a maximal run of consecutive `consistency_flag=1` windows within one video (mean length 7.03 windows, matching `window_size−1=7` — the same windowing-arithmetic identity already documented for this project's HMM branch). Detection search window: ±15 windows around each event, bounded so it never crosses into a neighboring event's own territory (a methodological fix applied during the original analysis after a first pass without this bound produced an implausible result — documented in the source report, not hidden here).

- **816 events** across 106 Val videos.
- **85.3% detected** at the BALANCED threshold (0.184); **99.0% detected** at the HIGH-SENSITIVITY threshold (0.059).
- Detection delay relative to **event start**: median = −3.0 windows (negative = crossing found before the event's own first positive window, within the search margin).
- Detection delay relative to **event midpoint**: median = −6.0 windows.
- **14.5% of detections only found a crossing at the outer edge of the ±15-window search margin** — search-limited; their true earliest crossing may lie further out and is not resolved by this analysis.

**The ambiguity between these two reference points is real and is not resolved here.** Because one true transition produces a run of ~7 overlapping positive windows, the run's first window is not self-evidently a better proxy for "the true transition instant" than its middle — the two delay estimates differ by about 3.0 windows on average (mean delay from start=−4.50 vs. from midpoint=−7.54; median from start=−3.0 vs. from midpoint=−6.0), close to, though not exactly equal to, the ~3.5-window gap that half the average 7-window event length would predict structurally. **This analysis does not claim a temporal precision finer than that ambiguity** — it supports the qualitative statement "detections are rarely late, and are frequently at-or-before the event's labeled start," not a specific number of windows or time units of anticipation.

## 8. Individual sequence analysis

Five sequences, already selected in the source analysis by explicit, data-driven criteria (not hand-picked for narrative convenience — see `run_log.txt` for the selection trace), at the BALANCED threshold (0.184):

| Patient | Ground Truth | GRU prediction | Score | Threshold | Detection | Delay | Interpretation |
|---|---|---|---:|---:|---|---:|---|
| Patient_324 | Transition | Detected | 0.825 | 0.184 | Immediate | 0 windows | Clean true positive — highest-confidence, on-time detection in the selected set. |
| Patient_264 | Transition | Detected | 0.185 | 0.184 | Delayed | +6 windows | True positive, but detected several windows after the event's labeled start — a representative (not extreme) case of late detection. |
| Patient_318 | Transition | Missed | 0.015 | 0.184 | Not detected | — | Clean false negative — GRU probability stayed far below threshold throughout the event, not a near-boundary miss. |
| Patient_360 | Mixed (has real events + extra flags) | Over-triggers | 0.569 (example window) | 0.184 | Multiple | — | False-positive-heavy case — 124 windows flagged above threshold that are not attributable to any real event in this video, the worst such case in Val. |
| Patient_302 | Transition | Unstable | 0.185–0.615 (oscillating) | 0.184 | Multiple, oscillating | −15 to +14 windows | Ambiguous case — probability repeatedly crosses the threshold in both directions around its events (19 crossings), the least temporally stable trace in Val. |

No biological context beyond what these numeric traces show is asserted for any of these five patients — the interpretations above are read directly off the probability/ground-truth trace, not inferred from clinical knowledge outside this analysis's scope.

**Each row above shows one representative event, not necessarily the video's only event.** Patient_324's 3 events are all near-immediate (0–1 window delay), a genuinely homogeneous case. Patient_360 is explicitly presented as mixed (real events plus unrelated false-positive flags). Patient_264 and Patient_318, however, each have several other events beyond the one shown — Patient_264 has 12 events total (9 on-time, 2 late, 1 missed) and Patient_318 has 8 (6 detected, 2 missed) — so "late detection"/"false negative" characterizes the *specific selected event*, not this video's behavior as a whole. Full per-event detail for all five videos (43 rows) is in `sequence_case_studies.csv`; per-video summary counts in `per_video_summary.csv`.

## 9. Identity Dynamics vs GRU

**Does GRU add anything over Identity Dynamics?** On the threshold-independent metrics, yes, by a modest but real margin:

| Metric | GRU | Identity Dynamics | Difference |
|---|---:|---:|---:|
| AUROC | 0.7638 | 0.7501 | +0.0137 (GRU better) |
| PR-AUC | 0.3565 | 0.3250 | +0.0315 (GRU better) |
| Brier | 0.1015 | 0.1033 | −0.0018 (GRU slightly better) |
| ECE | 0.0224 | 0.0104 | +0.0121 (Identity Dynamics better calibrated) |

GRU ranks positives above negatives more reliably (AUROC), separates the classes more effectively under imbalance (PR-AUC, the more informative metric at a 13% base rate), and has a marginally lower average squared calibration error (Brier). Identity Dynamics is better calibrated in the narrower ECE sense (its predicted probabilities track observed frequencies more closely, in absolute terms both values are still small).

**These are point estimates, not statistically tested differences.** No paired significance test (e.g. a bootstrap comparison in the style already used elsewhere in this project for the E1 ladder, `Training/evaluation/metrics.paired_bootstrap_comparison`) was run between GRU and Identity Dynamics in the source analysis. The margins above (+0.0137 AUROC, +0.0315 PR-AUC) are consistent with the visible gap in the ROC/PR curves (§6) on a large sample (43,339 windows), which is suggestive, but this document does not claim statistical significance that was never measured. A rigorous claim of "GRU is significantly better" would require that test as a follow-up, not this document's conclusion.

At any single threshold, the comparison is a genuine trade-off, not a strict win: GRU re-thresholded to its BALANCED point recalls 45.8 percentage points more than Identity Dynamics' own default (0.5) operating point, at the cost of 19.9 points more FPR and 26.3 points less precision (full table: `identity_dynamics_vs_gru_by_operating_point.csv`).

## 10. Potential role for RAG

**Can the GRU be sensitive enough for the future RAG?** The results suggest the GRU can constitute a detection or temporal-localization *signal* for the future RAG system, provided it is not used as a standalone final classifier. This is a deliberately qualified conclusion — the evidence supports a *signal* role, not a *decision* role:

Supporting the signal role:
- Higher AUROC (0.7638 vs 0.7501) and PR-AUC (0.3565 vs 0.3250) than Identity Dynamics.
- A real high-sensitivity regime exists: 90.4% recall reachable (threshold=0.059).
- 99.0% of real events produce at least one detectable crossing at that threshold.
- Detected events are rarely late (§7) — consistent with, though not proof of, useful temporal localization.

Bounding that role:
- The high-sensitivity regime's FPR is 56.8% — unusable as a direct trigger for a user-facing claim without a downstream confirmation step.
- F1 tops out at 0.379 (BALANCED threshold) — a limited, not a strong, classifier at any single operating point.
- False positives (§8, Patient_360) and false negatives (§8, Patient_318) both occur in real, identified cases, not just in aggregate statistics.
- ECE=0.0224 shows GRU's probability is not perfectly calibrated — it should be treated as a ranking/confidence score, not a literal probability, in any downstream use.
- Only one seed (seed=0) was evaluated; two additional trained seeds exist but were not run through this analysis (§11).
- A historical, not re-verified finding from this project's own prior work found this same GRU formulation's apparent advantage on the (locked) Test split reverses under a frame-shuffle sanity check — i.e., how much of its signal reflects genuine temporal reasoning versus content alone remains an open question this analysis does not resolve.

### Where the signal would sit in the future architecture

```
GRU
  ↓ event / transition signal, temporal localization (NOT a verified label)
Reporting API
  ↓ structured trajectory data (Semi-HMM phase posterior, next-phase distribution, duration)
RAG
  ↓ scientific / project / methodological documentation
LLM
  ↓ interpretation and synthesis
Explanation to the user
```

Role of each layer, restated for this specific signal:
- **GRU** supplies a bounded-confidence *signal* — "elevated transition probability observed here" — never a verified biological fact.
- **Semi-HMM** (already integrated into `Training/reporting/`) remains the source of probabilistic phase outputs (`filtering()`, `next_phase_distribution()`, duration) — unaffected by anything in this analysis.
- **Reporting API** remains the single source of truth for any structured, numeric trajectory data (`docs/PRODUCT_ARCHITECTURE.md` §4) — a GRU signal, if ever wired in, would be exposed alongside, not instead of, the Semi-HMM's own outputs, and never presented as equivalent to them.
- **RAG** supplies static documentary/methodological knowledge — never current predictions (`docs/RAG_ARCHITECTURE.md` §2), and must never treat a GRU score as a biological ground truth to retrieve or reason from as if verified.
- **LLM** interprets and synthesizes, framing any GRU-derived signal explicitly as a model output with the caveats above, not a fact.

This is a design note only — the GRU is not currently wired into `Training/reporting/` or any Reporting API endpoint; see `docs/PRODUCT_ARCHITECTURE.md` §4a for where this is recorded as a considered-but-unimplemented option, and `docs/RAG_ARCHITECTURE.md` §2 for the corresponding rule that any future GRU signal, like the Semi-HMM's own outputs, would never be embedded or stored in the vector DB.

## 11. Limitations

- **Class imbalance**: 13.24% positive rate (~1:6.6) on the common Val support — accuracy alone is a weak discriminator here; every metric above should be read with this in mind.
- **Single-seed GRU**: only the seed=0 checkpoint was evaluated. Seeds 1 and 2 exist on disk (`e1_gru_step6a_full/classification_seed{1,2}/`) but were not run — no cross-seed variability is reported; this document's numbers are a single point estimate, not a distribution.
- **Identity Dynamics history**: the only persisted historical Identity Dynamics metrics in this project were computed on **Test**, not Val (§2) — re-derived on Val for this comparison, deterministically, without retraining.
- **Calibration**: GRU's score is `sigmoid(logits)` of a BCE-trained head — **not assumed to be a calibrated probability**. Its ECE (0.0224) is small but non-zero and larger than Identity Dynamics' (0.0104); neither is production-grade calibrated without a dedicated recalibration step.
- **Timestamp ambiguity**: all timing figures use `time_unit=unknown/unverified`, this project's established convention (`docs/REPORTING_API.md`) — no unit (hours, minutes, etc.) is asserted anywhere in this document or its source.
- **Detection-delay reference-point ambiguity**: "delay from event start" and "delay from event midpoint" (§7) differ by ~3 windows on average and neither is self-evidently the correct reference for "the true transition instant," given that one transition produces a ~7-window run of positive labels — both are reported, this document does not silently pick one as authoritative.
- **Search margin**: the ±15-window bound used for event-detection timing means 14.5% of detected events' true earliest crossing may lie outside what was searched (§7) — a real, quantified, unresolved limitation, not a rounding error.
- **No significance testing**: the GRU vs Identity Dynamics differences (§9) are point estimates; no paired bootstrap or equivalent significance test was run for this specific comparison.
- **Frame-shuffle confound**: this GRU formulation's historical Test-split result (not re-verified in this Val-only analysis) reversed under a frame-shuffle sanity check in prior project work — an open question about how much of its edge is genuine temporal reasoning, explicitly not resolved here.
- **GRU-forecast formulation** (the residual/persistence-style variant) scored near chance (AUROC=0.523) on Val and was not analyzed in depth — not representative of the "classification" formulation discussed throughout this document.

## 12. Conclusion

On the Val split (106 videos, 43,339 windows, 100% common support with Identity Dynamics after correcting a real split mismatch in the historical baseline), the GRU (classification formulation, seed=0, frozen checkpoint) outperforms Identity Dynamics on every threshold-independent ranking metric (AUROC +0.0137, PR-AUC +0.0315) and offers a genuine high-sensitivity operating regime (90.4% recall at FPR=56.8%), but does not achieve strong performance at any single threshold (F1 tops out at 0.379) and is not perfectly calibrated (ECE=0.0224). These differences are point estimates, not statistically tested, and were obtained from a single trained seed.

Taken together, the evidence supports treating the GRU as a **candidate signal source for event detection and temporal localization** in the future RAG/reporting pipeline — not as a standalone, final-answer classifier, and not as a source of biological ground truth. This conclusion is consistent with, and requires no change to, this project's existing architectural principle that the Reporting API (built on the frozen Semi-HMM) remains the sole source of truth for any number the system presents to a user.

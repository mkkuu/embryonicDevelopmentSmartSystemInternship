# HANDOFF — EMISSION BALANCED

Short, self-contained. Full detail: `docs/PROJECT_CHECKPOINT_EMISSION_BALANCED.md`. Full analysis: `Results/evaluation/emission_balanced/FINAL_REPORT.md` (server).

## Current state

Experiment `emission_balanced` is **DONE**. Ran 2026-08-23T15:23:42Z → 15:38:35Z (14m53s), `EXITCODE:0`, no NaN/Inf, gate passed (target-phase emission accuracy +0.136 > +0.05 threshold), full pipeline (HMM balanced + Semi-HMM balanced + all comparisons) executed. Live-verified on the server 2026-08-24 — screen session gone (normal, job exited on its own), integrity of all historical artifacts confirmed (checksums identical for `Results/`; embeddings confirmed untouched by mtime — none newer than the run's launch time).

## Result — one-line summary

**`class_weight='balanced'` fixes emission accuracy on the 5 target phases (gate passed) but makes HMM and Semi-HMM k=7 calibration and ranking significantly WORSE (AUROC -0.08, Brier +0.015, ECE +0.02, all p<1e-4) — this is Case D of the pre-registered taxonomy: emission improves, temporal calibration degrades.** Do not adopt `class_weight='balanced'` as-is for the temporal models.

## Key numbers

| | Baseline | Balanced |
|---|---|---|
| Emission overall accuracy | 0.4639 | 0.3811 |
| Emission mean target-phase accuracy | 0.0996 | 0.2356 |
| HMM k=7 AUROC / Brier / ECE | 0.63735 / 0.14072 / 0.12312 | 0.55157 / 0.15613 / 0.14359 |
| Semi-HMM k=7 AUROC / Brier / ECE | 0.64271 / 0.14029 / 0.12307 | 0.56688 / 0.15435 / 0.14174 |

Semi-HMM still beats HMM on AUROC after rebalancing (+0.0153, CI excludes 0), but not significantly on Brier/ECE (CIs cross 0) — same qualitative edge as before, unchanged by this experiment.

## What must NOT be touched

- `Results/evaluation/e1_hmm_k7_sweep/`, `Results/evaluation/semi_hmm_weekend_phaseF/`, any `e1_*`/GRU/Linear SSM result — confirmed untouched (checksums/mtimes).
- `Embeddings/resnet18/{train,val}/` — confirmed untouched.
- `Embeddings/resnet18/test/` — never loaded (grep-confirmed in the experiment script + `results.json` literally records `"Test never loaded"`).

## Next action (not launched)

See `Results/evaluation/emission_balanced/FINAL_REPORT.md` §11-12 for full reasoning. Recommended single next step: refine the emission fix (e.g. milder/custom class weighting, or `class_weight='balanced'` + post-hoc probability calibration such as Platt/isotonic scaling on the emission output) rather than either (a) adopting plain `'balanced'` as-is, or (b) moving straight to building the probabilistic reporting chain — current models (both baseline and balanced) are not calibrated well enough for that yet. **Not authorized, not started.**

## Test status

TEST = LOCKED (confirmed this session, not just carried over).

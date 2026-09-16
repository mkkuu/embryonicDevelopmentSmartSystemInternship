# HANDOFF — EMISSION WEIGHTED SWEEP

Short, self-contained. Full detail: `Results/evaluation/emission_weighted_sweep/REPORT.md` (server).

## Current state

**DONE**. Ran 2026-08-24T07:34:33Z→08:07:07Z (32m34s), `EXITCODE:0`, 0 NaN/Inf. Integrity verified: embeddings + all historical Results/ (HMM, Semi-HMM, E1, GRU Test, Linear SSM, emission_balanced) checksum-identical to pre-run; `Embeddings/resnet18/test/` mtime unchanged, never referenced.

## Result — one-line summary

**No weighting in {0.25, 0.5, 0.75, 1.0} improves HMM/Semi-HMM k=7 over the original unweighted emission (alpha=0).** Degradation at the temporal level is monotonic and significant at every gate-passing alpha, including the mildest (0.5: AUROC −0.038, Brier +0.009, ECE +0.011, all p<0.0001). The original, unweighted models remain the best-validated HMM/Semi-HMM configuration in the whole project.

## The one open thread — now CLOSED (2026-08-24, `emission_alpha025`)

At the **emission level only**, top-label ECE is non-monotonic — alpha≈0.25-0.5 is *better calibrated* than the unweighted baseline (ECE 0.1139 vs 0.1214 at alpha=0.25). This candidate never reached HMM/Semi-HMM k=7 in the sweep (gate screened it out: target-phase gain +0.0295 < +0.05 threshold). **Tested directly, unconditionally, this session**: it does not survive propagation. HMM k=7 ECE 0.12312→**0.12784** (worse, p<0.0001), AUROC 0.63735→0.62057 (p<0.0001); Semi-HMM same pattern (ECE 0.12307→0.12939, AUROC 0.64271→0.62518). The emission-level calibration gain is more than reversed once passed through the HMM/Semi-HMM forward algorithm's Bayes-ratio conversion. **Class-reweighting is now definitively CLOSED as a branch** — see `Results/evaluation/emission_alpha025/REPORT.md`.

## Key numbers

| alpha | HMM AUROC/Brier/ECE | Semi-HMM AUROC/Brier/ECE |
|---:|---|---|
| 0.0 | 0.6374 / 0.1407 / 0.1231 | 0.6427 / 0.1403 / 0.1231 |
| 0.5 | 0.5990 / 0.1498 / 0.1343 | 0.6064 / 0.1499 / 0.1351 |
| 0.75 | 0.5724 / 0.1538 / 0.1404 | 0.5862 / 0.1525 / 0.1397 |
| 1.0 | 0.5516 / 0.1561 / 0.1436 | 0.5669 / 0.1544 / 0.1417 |

## What must NOT be touched

- `Results/evaluation/e1_hmm_k7_sweep/`, `semi_hmm_weekend_phaseF/`, `emission_balanced/`, any `e1_*`/GRU/Linear SSM result — confirmed untouched.
- `Embeddings/resnet18/{train,val}/` — confirmed untouched. `Embeddings/resnet18/test/` — never loaded.
- New artifacts (`hmm_alpha_0p5/`, `hmm_alpha_0p75/`, `semi_hmm_alpha_0p5/`, `semi_hmm_alpha_0p75/`) are gate-passing candidates only, not recommended for adoption — kept for reproducibility/comparison, not as a new reference model.

## Branch status: CLOSED

Class-reweighting has been tested at every meaningful strength (`emission_balanced` alpha=1, `emission_weighted_sweep` alpha=0.5/0.75, `emission_alpha025` alpha=0.25 — the one candidate with genuinely better emission calibration) and **none improve HMM or Semi-HMM k=7** over the original, unweighted emission. This branch is closed; do not relaunch any further alpha value without a new, different hypothesis (not "try a value in between again" — the sweep already established the relationship is monotone-worsening at the temporal level across the whole tested range).

## Two remaining directions (recommendation only, neither launched)

**(A) Post-hoc calibration / different emission strategy.** Attacks the same emission bottleneck (`docs/SCIENTIFIC_RESULTS.md` §9) via a different mechanism than reweighting — e.g. Platt/isotonic scaling on top of the original (unweighted) emission's output, or a richer feature representation / different classifier family. Cost: moderate (new code, new fit). Risk: unknown until tried — no evidence yet either way. Relevance: directly targets the confirmed dominant bottleneck.

**(B) Proceed to the probabilistic reporting chain** on the existing frozen, best-validated (unweighted) models, with the guardrails already documented (`docs/SCIENTIFIC_RESULTS.md` §14: CDF/quantiles not raw pmf, tPB2/tEB marked non-estimable). Cost: low-moderate (mostly integration, not new modeling). Risk: inherits the known emission limitation on t3/t5/t6/tPNf — acceptable if the guardrails are followed, since the limitation is now precisely documented rather than papered over. Relevance: converts already-validated capability into product/reporting value without waiting on an uncertain further modeling win.

**Recommendation (not launched)**: (A) is the more scientifically informative next step if the goal is still to improve the models — the emission bottleneck is confirmed and well-characterized, but no fix has yet been tried besides reweighting. (B) is the more product-relevant next step if the goal is to start delivering value from what's already validated. Both are reasonable; the choice depends on which of "improve the model further" vs. "ship what's proven" the project prioritizes next. **Neither has been started.**

## Test status

TEST = LOCKED (confirmed this session).

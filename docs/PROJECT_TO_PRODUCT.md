# Project to Product

Transition assessment from research state to product-readiness, 2026-08-24 (updated same day — Reporting API built, tested, and documented; full product architecture planned). Synthesizes `docs/SCIENTIFIC_REPORT.md`, `docs/MODEL_COMPARISON.md`, `docs/INFERENCE_SCHEMA.md`, `docs/WEBAPP_DATA_REQUIREMENTS.md`, `docs/RAG_DATA_MODEL.md`, `docs/REPORTING_API.md`, `docs/PRODUCT_ARCHITECTURE.md`, `docs/RAG_ARCHITECTURE.md`, `docs/LLM_ORCHESTRATION.md`, `docs/WEBAPP_ARCHITECTURE.md`, `docs/PRODUCT_ROADMAP.md`.

**Update since this document was first written**: the Reporting API (§"Ce qui nécessite encore du développement" originally listed this as a gap) is now **built, tested (287/287), and documented** — see `docs/REPORTING_API.md`. Two new findings from that work materially affect the sections below: (1) `SemiHMMModel.next_phase_distribution()` costs ~333s/video cold — a real, unmitigated performance risk for any RAG/LLM tool calling it synchronously; (2) the raw 15-way `phase_probabilities` posterior is severely miscalibrated (Brier≈0.96, worse than uniform) even though ranking remains useful — this must shape UI/LLM presentation choices, not just be a footnote. Full product architecture (RAG, LLM orchestration, Web App, roadmap) is now planned in the five docs listed above — none of it implemented yet.

## Ce qui est scientifiquement validé

- Embedding content (ResNet18) carries real signal for phase/transition detection (`identity_dynamics` AUROC 0.7384 vs. base_rate 0.5000).
- Explicit dynamics modelling — linear (E1) and non-linear (GRU) — does **not** beat content alone on this task; both negatives survive frame-shuffle validity controls.
- A structured discrete-state (HMM) reformulation adds real, if modest, ranking value once its target-alignment bug is fixed (k=7).
- Explicit duration modelling (Semi-HMM) adds a further small, directionally consistent gain over HMM.
- The shared emission is the dominant calibration bottleneck (three independent lines of evidence: Murphy decomposition, transition-structure identity, direct emission-only accuracy).
- Naive class-reweighting, at every tested strength, fails to fix that bottleneck without breaking calibration further — this pathway is closed, not just untried.

## Ce qui reste expérimental

- Whether any fix to the emission bottleneck exists at all (post-hoc calibration and richer emission features are both untried).
- Semi-HMM's edge over plain HMM (real in direction, not trajectory-significant, n=106, p=0.29).
- Any Test-confirmed generalization estimate for HMM/Semi-HMM (Val-only throughout, by design — Test remains locked).
- Multi-step-ahead forecasting beyond one transition (no semi-Markov Viterbi decode implemented).
- Real-time (clinical hours) duration reporting — durations exist only in window-index units today.

## Ce qui peut alimenter directement la web app

- Current phase + full 15-way posterior (`HMMModel.explain()`).
- Next-phase distribution, causal (`explain()`) or duration-aware (`SemiHMMModel.next_phase_distribution()`).
- Duration distribution / expected duration / quantiles for 13/15 phases (Semi-HMM), with the tPB2/tEB fallback flag already present in the data (`fallback_uniform`).
- Ground-truth phase/frame data for annotated videos (timelapse viewer's "real phase if available").
- Model version/provenance (from any loaded `state.json`).

**Not yet wired**: no serving/API layer exists between these outputs and any UI — see `docs/DATA_LINEAGE.md`'s gap list.

## Ce qui peut alimenter directement le RAG

- SEMANTIC layer: `docs/SCIENTIFIC_REPORT.md`, `docs/HANDOFF.md`, `docs/HMM_RESEARCH_PLAN.md` — already markdown prose, ready to chunk/index as-is.
- EXPERIMENTAL layer: the full experiment chronology (§5-14 of `SCIENTIFIC_REPORT.md`), negative results, model-selection reasoning — same, ready to chunk/index.
- STRUCTURED layer: same sources as the web app section above, once a live serving layer exists.

**Not yet available**: a phase-level biological glossary (a real content gap, not just technical — see `RAG_DATA_MODEL.md`), and any mechanism ensuring a RAG answer surfaces the relevant documented limitation automatically.

## Ce qui nécessite encore du développement

- ~~A prediction-serving API~~ — **DONE**: `Training/reporting/`, 7 endpoints, 287/287 tests, see `docs/REPORTING_API.md`.
- **New, discovered during that build**: a pre-warming/caching strategy for `next_phase_distribution()`'s ~333s/video cold cost — see `docs/PRODUCT_ARCHITECTURE.md` §9, roadmap Phase 1.5.
- A `SemiHMMModel.explain()`-equivalent unified per-window record (today the Reporting API composes it in `inference_service.py`, at the service layer, not inside the model itself — a reasonable, working compromise, not a hard blocker).
- A frame-index → real frame / timestamp resolution layer (the `Data/embryo_dataset_time_elapsed/` join was only ever done additively inside the HMM branch's own diagnostics, never at the canonical metadata/model level) — the Reporting API surfaces this honestly (`time_unit="unknown/unverified"`) rather than solving it.
- Frame **image** serving (raw `.jpg`/`.png` over HTTP) — a real, currently-unaddressed gap for the Web App's Visionneuse, distinct from the metadata already served.
- RAG chunking/indexing of the SEMANTIC and EXPERIMENTAL layers, and retrieval logic that reliably attaches documented limitations to structured answers — now fully designed (`docs/RAG_ARCHITECTURE.md`), not yet built.
- LLM orchestration, tool calling, Web App, deployment — all designed (`docs/LLM_ORCHESTRATION.md`, `docs/WEBAPP_ARCHITECTURE.md`, `docs/PRODUCT_ARCHITECTURE.md`), none implemented. Full phased plan: `docs/PRODUCT_ROADMAP.md`.
- A phase-level biological glossary (content authoring, not engineering).

## Ce qui nécessite encore de la validation scientifique

- Any post-hoc calibration approach (Platt/isotonic) — proposed, never tried.
- A richer emission (different features or classifier family) as an alternative fix to class-reweighting.
- Multi-step probabilistic forecasting beyond the current one-step-ahead / segment-diagnostic capabilities.
- Any generalization claim — nothing in the HMM/Semi-HMM/emission-experiment lineage has been Test-evaluated (deliberately).

## Ce qui doit rester hors du produit

- Any number derived from tPB2/tEB's duration model presented as if informative (must always carry the "structurally not estimable" flag, per `docs/INFERENCE_SCHEMA.md`).
- Raw duration pmf values (must be CDF/quantile-summarized, never shown point-wise).
- Any reweighted-emission model (balanced, alpha∈{0.25,0.5,0.75,1}) as a production candidate — all closed, all worse than the current pipeline candidate.
- Fabricated textual justification not backed by the RAG's actual retrieval — the web app must not synthesize ad hoc explanatory text itself (`docs/WEBAPP_DATA_REQUIREMENTS.md`'s cross-cutting note).
- `run_e1_baselines.py` as an active entry point (superseded, kept only for historical/methodological reference).
- Any Test-derived number outside the GRU branch's one pre-registered exception.

## Architecture cible

```
DATA
  ↓
MODEL  (Semi-HMM k=7, unweighted emission — CURRENT_PIPELINE_CANDIDATE)
  ↓
INFERENCE  (HMMModel.explain() + SemiHMMModel's duration/chain methods — needs unification)
  ↓
PROBABILISTIC STATE  (docs/INFERENCE_SCHEMA.md's canonical shape — not yet serialized as one record)
  ↓
REPORTING API  (NOT BUILT — the biggest concrete gap between today and the next phase)
  ↓
WEB APP  (NOT BUILT — requirements specified in docs/WEBAPP_DATA_REQUIREMENTS.md)
  ↓
RAG  (NOT BUILT — data model specified in docs/RAG_DATA_MODEL.md, content gaps identified)
```

The critical path is the **Reporting API** — everything upstream (model, inference, probabilistic state) already exists in usable form; everything downstream (web app, RAG) is blocked on it, not on further modelling work.

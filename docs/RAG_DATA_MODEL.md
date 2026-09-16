# RAG Data Model

**Chunk/document schema and vector-DB layer: IMPLEMENTED, Phase 3 of `docs/PRODUCT_ROADMAP.md`** — see §"Chunk and document schema (implemented)" below and `docs/RAG_INGESTION.md`. The three-layer STRUCTURED/SEMANTIC/EXPERIMENTAL framing below predates Phase 3 and remains the model's conceptual backbone; only SEMANTIC + a slice of EXPERIMENTAL knowledge are actually indexed as of Phase 3 (see the updated STRUCTURED DATA note immediately below) — retrieval/orchestration on top of this index is not implemented.

## STRUCTURED DATA

Per-sample, machine-generated, numeric/categorical — changes with every new inference. **Never indexed in the RAG** (see `docs/RAG_ARCHITECTURE.md` §2's "dynamic data never in the vector DB" rule, verified structurally in Phase 3 — `docs/RAG_PHASE_3_REPORT.md` §12) — this layer's real, current implementation is the Reporting API, not a document to retrieve:

| Data | Source (today) |
|---|---|
| Current phase | `GET /videos/{id}/inference/{window}` (`Training/reporting/inference_service.py`) |
| Phase probabilities (15-way posterior) | same endpoint — `SemiHMMModel.filtering()` |
| Next-phase distribution | same endpoint — `SemiHMMModel.next_phase_distribution()` (duration-aware, optimized Phase 1.6) |
| Transitions (segment-level chain) | `GET /videos/{id}/trajectory` — `SemiHMMModel.transition_chain()` |
| Durations (distribution, expected value, quantiles) | same endpoint block — `SemiHMMModel.duration_distribution()`/`expected_duration()`/`quantiles()` |
| Model outputs / provenance | `GET /model` — loaded model's `state.json` (hyperparameters), see `docs/INFERENCE_SCHEMA.md` |

Exact field shapes: `docs/INFERENCE_SCHEMA.md`, full endpoint table: `docs/REPORTING_API.md`. This layer is what answers questions like *"what is the model's current estimate"* — it has a live, working, cached (`docs/REPORTING_CACHE.md`) connection to the frozen Semi-HMM today, reached only via the Reporting API, never via the RAG index.

## SEMANTIC KNOWLEDGE

Static, human-authored or biologically-grounded — does not change per sample, changes rarely (only when the science itself is revised).

- Phase definitions (`tPB2, tPNa, tPNf, t2, ..., tEB` — the 15-phase chronological taxonomy, currently only implicitly documented via `DEFAULT_PHASE_NAMES` in code and scattered domain knowledge; **not yet written up as a standalone glossary**).
- Biological description / interpretation of each phase — **not currently documented anywhere in this repository** (a real gap; the original project's biological domain knowledge, if written down at all, lives outside `Training/`/`docs/`).
- Dataset documentation — `README.md` (original project), `docs/SCIENTIFIC_REPORT.md` §2-4 (SciML-specific: splits, embeddings, preprocessing).
- Methodology — `docs/HANDOFF.md` (mathematical formulation), `docs/HMM_RESEARCH_PLAN.md` (HMM/Semi-HMM design), `docs/SCIENTIFIC_REPORT.md` (full narrative).
- Model limitations — `docs/SCIENTIFIC_REPORT.md` §17, `docs/MODEL_COMPARISON.md`'s `KNOWN_LIMITATIONS`.

This layer is the most RAG-ready today — it already exists as markdown prose, just not chunked/indexed.

## EXPERIMENTAL KNOWLEDGE

The scientific history — why the current model is what it is, not what it isn't.

- Every experiment's motivation/hypothesis/method/result/conclusion/decision — `docs/SCIENTIFIC_REPORT.md` §5-14, in full.
- Metrics per experiment — same sections, plus raw `Results/evaluation/*/results.json`/`REPORT.md` for exact numbers.
- Model-selection decisions — `docs/MODEL_COMPARISON.md`'s `CURRENT_PIPELINE_CANDIDATE` section (`WHY_SELECTED`).
- Negative results — E1 (linear dynamics), GRU (non-linear dynamics, both formulations), Semi-HMM's non-trajectory-significant gain, class-reweighting (Case D, then closed entirely) — all in `docs/SCIENTIFIC_REPORT.md`, written as negatives, not reframed as positives.

This layer is what lets the RAG explain *why*, not just *what*.

## Example questions and what layer(s) they need

| Question | Layer(s) needed |
|---|---|
| "Pourquoi le modèle estime-t-il t3 ?" | STRUCTURED (current posterior) + SEMANTIC (what t3 means) + EXPERIMENTAL (known emission weakness on t3, 4.3-17.6% accuracy depending on weighting — must be surfaced as a caveat, not hidden) |
| "Quelle est la prochaine phase la plus probable ?" | STRUCTURED only |
| "Quelle est la probabilité de transition t3 → t4 ?" | STRUCTURED (`next_phase_distribution`) |
| "Quelle durée est attendue ?" | STRUCTURED (`expected_duration`) + EXPERIMENTAL (flag if the phase is tPB2/tEB — structurally inestimable) |
| "Quelle est l'incertitude ?" | STRUCTURED (entropy/quantiles) |
| "Quelle méthode produit cette prédiction ?" | STRUCTURED (model version) + SEMANTIC (methodology docs) |
| "Quelles sont les limites connues de cette prédiction ?" | EXPERIMENTAL (this is the layer's whole purpose) |

Every one of these except pure "what does the model say right now" questions needs the EXPERIMENTAL layer — a RAG built only on STRUCTURED data would answer confidently and misleadingly on exactly the phases (t3/t5/t6/tPNf) where confidence is least warranted. This is the single most important design constraint for the future RAG: **it must always be able to attach a documented limitation to a structured answer when one exists**, not just retrieve the number.

## Chunk and document schema (implemented, Phase 3)

`Training/rag/schemas.py`. Two levels — `DocumentRecord` (one per source file) and `Chunk` (one per retrievable unit, denormalizing its parent document's metadata so a retrieval result never needs a second lookup):

```json
{
  "chunk_id": "gru_identity_analysis::0009",
  "document_id": "gru_identity_analysis",
  "document_type": "project",
  "source": "docs/GRU_IDENTITY_ANALYSIS.md",
  "title": "GRU vs Identity Dynamics",
  "status": "authoritative",
  "authoritative": true,
  "version": "<sha256 of the source file>",
  "section": "10. Potential role for RAG",
  "section_level": 2,
  "chunk_index": 9,
  "content": "...",
  "content_hash": "<sha256 of this chunk's content>",
  "contains_table": false,
  "created_at": "2026-08-25T10:00:00Z"
}
```

`status` vocabulary: `authoritative` / `current` / `historical` / `obsolete` (defined, unused this pass — nothing in the corpus is tagged obsolete, see `docs/RAG_INGESTION.md`'s exclusion list instead) / `experimental`. `version` is the parent document's content hash — "which document version produced this chunk," directly answerable per-chunk, matching `docs/PRODUCT_ARCHITECTURE.md` §11's versioning goal applied to the RAG side. Full design rationale and the retrieval-ranking implication (`authoritative`/`status` should outrank a plain similarity tie): `docs/RAG_PHASE_3_REPORT.md` §4.

## What is NOT ready

- No live connection between STRUCTURED data and the RAG index — by design, not a gap (see above); the gap that DOES remain is a retrieval/RAG-facing tool that calls the Reporting API, not yet built (Phase 6 of the roadmap, tool calling).
- SEMANTIC and a slice of EXPERIMENTAL knowledge ARE now chunked/indexed (30 documents, 432 chunks, Phase 3) — but retrieval evaluation is a smoke test only (`docs/RAG_RETRIEVAL.md`), and several real content gaps surfaced doing it (e.g. no single clean "what is Semi-HMM" definitional chunk, no single clean "why is Test locked" chunk — see `docs/RAG_PHASE_3_REPORT.md` §11).
- `Results/evaluation/*/REPORT.md` (raw per-experiment EXPERIMENTAL knowledge) is not yet ingested — scoped out of Phase 3, see `docs/RAG_PHASE_3_REPORT.md` §16.
- No phase-level biological glossary exists yet — a real content gap, not just a technical one.
- No mechanism to guarantee a RAG answer surfaces the relevant EXPERIMENTAL caveat automatically (e.g., always mentioning the emission bottleneck when asked about t3) — this is a retrieval/prompting design problem for the future orchestrator (Phase 5), not solved by the data model or the Phase 3 retrieval prototype alone.

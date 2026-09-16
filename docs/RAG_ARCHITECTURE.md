# RAG Architecture

**Ingestion foundation IMPLEMENTED, Phase 3 of `docs/PRODUCT_ROADMAP.md`** (2026-08-25) — vector DB deployed (Chroma, `RagIndex/`), 30 documents / 432 chunks ingested, a retrieval prototype built and smoke-tested. Full build report: `docs/RAG_PHASE_3_REPORT.md`; day-to-day usage docs: `docs/RAG_INGESTION.md`, `docs/RAG_RETRIEVAL.md`.

**Update, 2026-08-26**: `retrieval.retrieve()` now has a typed orchestrator-facing adapter — `Training/orchestrator/tools.retrieve_documents()` (and `get_phase_information()`, a constrained variant) — see `docs/TOOL_CONTRACTS.md`. This is still the same raw prototype underneath (no re-ranking, no multi-collection merge, no intent detection beyond the orchestrator's own router) — the adapter only adds typed error handling (`ToolUnavailableError`/`ToolInputError`) and a stable call signature the orchestrator's context builder depends on. **Still not built**: a real Phase-4-grade retrieval pipeline (intent-aware, multi-collection, metadata-filtered beyond the flat filters already in `retrieval.retrieve()`), and any LLM answer-synthesis layer on top of retrieval — see `docs/RAG_ORCHESTRATION_PHASE_REPORT.md` for what this phase actually added (a router + context builder + LLM abstraction, not a new retrieval pipeline). Builds on `docs/RAG_DATA_MODEL.md` (which knowledge exists and where) by adding the missing *how*: storage, chunking, retrieval mechanics, evaluation.

## 1. Knowledge categories (hybrid RAG)

Three static categories, each a **separate collection** in the vector DB (not one undifferentiated pool) — allows per-category retrieval weighting and metadata filtering without cross-contamination:

| Collection | Content | Source docs |
|---|---|---|
| `scientific_knowledge` | Phase definitions, embryology background, dataset documentation, methodology | `docs/HANDOFF.md`, `docs/SCIENTIFIC_REPORT.md` §1-4, original project `README.md` — **plus a genuine content gap**: no phase-level biological glossary exists yet anywhere in this repo (flagged already in `docs/RAG_DATA_MODEL.md`, re-confirmed here, not yet closed) |
| `project_knowledge` | Architecture, experiment history, results, model comparisons, decisions | `docs/SCIENTIFIC_REPORT.md` (full), `docs/MODEL_COMPARISON.md`, `docs/GLOBAL_MODEL_COMPARISON.md`, `Results/evaluation/*/REPORT.md` |
| `methodological_knowledge` | HMM/Semi-HMM/GRU/Linear SSM math, metrics (Brier/ECE/PR-AUC/calibration), definitions | `Training/evaluation/models/hmm.py`'s and `semi_hmm.py`'s module docstrings (already extensive, genuinely reusable as source text), `docs/HMM_RESEARCH_PLAN.md`, `Results/evaluation/global_model_comparison/PRAUC_REPORT.md` |

## 2. Dynamic data: never in the vector DB

**Rule**: current predictions, probabilities, durations, entropy — anything the Reporting API can compute for a specific `(video, split, window)` — is **never embedded or stored** in the RAG index.

**Why, precisely**:
1. **Freshness**: a vector DB is optimized for slowly-changing content re-indexed occasionally. A prediction is per-window, per-video, computed on demand — indexing it would mean re-embedding on every inference, an absurd cost multiplier for zero retrieval benefit (the Reporting API is already a direct, exact lookup — no semantic search is needed to find "the prediction for window 54 of Patient_319").
2. **Single source of truth** (§27 of the request, already a hard constraint in `docs/PRODUCT_ARCHITECTURE.md` §4): if a number could come from either the RAG index or the Reporting API, the two could silently disagree (e.g. if the model is ever updated and re-run, but the RAG index wasn't re-ingested) — a correctness bug with no natural detection mechanism, since both would "look like" valid retrieval.
3. **Retrieval error surface**: embedding a number and retrieving it via semantic similarity is not exact retrieval — a vector search could return the *wrong window's* probability as "close enough," a failure mode a direct API call structurally cannot have.
4. **Scale mismatch**: 43,339 Val windows × 15 phases × several derived fields would be a enormous, constantly-stale index for information a single API call already serves in milliseconds.

**Consequence**: the LLM orchestrator's tools (`docs/LLM_ORCHESTRATION.md`) always call the Reporting API live for anything numeric; RAG retrieval is only ever invoked for the three static categories above.

**Same rule applies to any future GRU-derived signal** (`docs/PRODUCT_ARCHITECTURE.md` §4a, `docs/GRU_IDENTITY_ANALYSIS.md`, design note only, not implemented): if a GRU event/temporal signal is ever wired into the Reporting API, it is dynamic, per-window, per-video data exactly like the Semi-HMM's own outputs — never embedded or stored in the vector DB, never retrieved via semantic search, and never treated by the RAG/LLM layer as a documented fact to cite. It would be fetched live, tagged with its own provenance/uncertainty, and explicitly distinguished from the static `methodological_knowledge` collection's *description* of what the GRU is and how it was evaluated (which — unlike its live output — is exactly the kind of stable, documentary content this collection already exists to hold).

## 3. Vector database

### Comparison (only the options actually relevant at this project's scale)

| | FAISS | Chroma | Qdrant | pgvector |
|---|---|---|---|---|
| Deployment | Library, no server | Embedded or optional server | Separate server (Docker) | Postgres extension |
| Native metadata filtering | No (needs a companion store) | Yes | Yes, rich | Yes (SQL `WHERE`) |
| Ops overhead | Minimal | Minimal | Moderate (new service) | **Zero new service** if Postgres already exists |
| Fit for this project's scale (low hundreds of chunks) | Overkill precision, awkward filtering | Good fit | Overkill for now | Good fit, reuses existing infra |
| Existing project infra | None | None | None | **`WebApplication/` already runs PostgreSQL** (`Dataset_shema.sql`, `psycopg2-binary` already in `requirements.txt`) |

### Recommendation — CONFIRMED and implemented, Phase 3

**Chroma** — embedded, zero new ops burden, native metadata filtering matching the schema below, fastest to prototype and iterate on while the corpus is small and evolving alongside the docs it indexes. Re-checked against the real repository this phase (not re-derived from scratch) before implementing: still the right fit at the corpus's actual measured scale (30 documents, 432 chunks — squarely inside the "low hundreds of chunks" this recommendation was originally sized for). New dependency added to `requirements.txt`: `chromadb`.

**pgvector flagged as the natural migration target** if/when the product consolidates its backend onto Postgres (which `WebApplication/` already uses) — reusing existing database infrastructure, credentials pattern (`.env`), and ops knowledge is a real, non-trivial advantage once the system is no longer a research prototype. **This is a Decision Required** (§ below), not resolved here, because it has real infrastructure/ops implications beyond this document's scope — the near-term Chroma recommendation is made so the roadmap isn't blocked on that larger decision.

Qdrant is not recommended now — its strengths (production-grade hybrid search, horizontal scale) don't match this project's current corpus size or traffic; revisit only if the product grows well beyond a single research team's usage.

### Document structure / chunking

- **Chunk size**: target 300-500 tokens, split on markdown section boundaries (`##`/`###` headers) rather than fixed character windows — every doc in this repo is already well-sectioned (verified: `docs/SCIENTIFIC_REPORT.md` has 21 numbered `##` sections, `docs/MODEL_COMPARISON.md` has one `###` per model) — structural chunking preserves a complete thought per chunk instead of splitting mid-argument.
- **Overlap**: 1-2 sentences of trailing context from the previous chunk, to avoid losing a referent ("this gain", "the same finding") at a chunk boundary.
- **Metadata per chunk** (matches the requested schema exactly):

```json
{
  "document_type": "scientific | project | methodological",
  "source": "docs/SCIENTIFIC_REPORT.md",
  "experiment": "emission_balanced | hmm_k7_sweep | ... | null",
  "model": "semi_hmm | hmm | gru | linear_ssm | ... | null",
  "phase": "t3 | t5 | ... | null",
  "date": "2026-08-24",
  "version": "<source doc content hash>",
  "confidence": "established | preliminary | superseded",
  "scientific_status": "confirmed | partially_confirmed | rejected | inconclusive | not_applicable"
}
```

`confidence`/`scientific_status` are populated from the source document's own language where it already exists (e.g. `docs/SCIENTIFIC_REPORT.md` §12's CONFIRMED/PARTIALLY CONFIRMED/REJECTED table) — never inferred by the ingestion pipeline; a chunk with no explicit status marker in its source gets `scientific_status: not_applicable`, not a guessed value.

### Retrieval strategy

- Per-collection semantic search (top-k, k≈5 per collection for a 3-collection hybrid query), then a lightweight re-rank/merge step (recency + `scientific_status` priority — a `CONFIRMED` project-knowledge chunk should outrank a `superseded` one on a tied similarity score).
- Metadata filters applied **before** semantic search where the question implies them (e.g. a question mentioning "t5" filters to `phase: t5` chunks first) — cheaper and more precise than filtering after retrieval.
- Embedding model for the RAG corpus is a **separate decision from the ResNet18 embeddings** used by the scientific model — **decided and implemented, Phase 3**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, local CPU inference, no API key. Chosen multilingual specifically because the corpus is English prose but real usage questions (this project's own future users) are French — an English-only model was tried first and measurably failed cross-lingual retrieval before this switch (`docs/RAG_INGESTION.md`). New dependency added to `requirements.txt`: `sentence-transformers`.

## 4. Retrieval pipeline

```
Question
  → intent detection (does this need live data? static knowledge? both?)
  → retrieval planning (which collections, which metadata filters, does it need a Reporting API call)
  → [parallel] Reporting API call(s)  +  vector retrieval (per relevant collection)
  → context assembly (typed: FACT block from API, DOCUMENTATION block from RAG, each tagged with its source)
  → LLM (given both blocks, plus tool-calling ability for anything not yet fetched)
  → answer (with citations, see docs/LLM_ORCHESTRATION.md)
```

Worked example — *"Pourquoi le modèle prédit t5 ?"*:
1. Intent detection: needs current prediction (which window? — requires session/context state, see `docs/LLM_ORCHESTRATION.md` §Session context) **and** explanation.
2. Retrieval planning: (a) Reporting API `get_inference(video, window)` for the live posterior; (b) `methodological_knowledge` retrieval filtered `phase: t5` for t5's known emission difficulty; (c) `project_knowledge` retrieval for the emission-diagnosis finding (t5 accuracy 3.4%, `docs/SCIENTIFIC_REPORT.md` §13).
3. Context assembly: FACT block (P(t5)=X from the API) + DOCUMENTATION blocks (t5's definition, its known emission weakness) + INTERPRETATION space for the LLM to connect them.
4. LLM produces: "The model assigns t5 a probability of X (source: live inference). t5 is one of the phases the shared emission classifier struggles with most — only 3.4% raw accuracy in the unweighted configuration, and this project's diagnostics found the bottleneck is the emission itself, not the temporal model [source: SCIENTIFIC_REPORT.md §13]. This means the confidence in this specific prediction should be treated with caution."

## 5. RAG evaluation

Not "does the chatbot work" — explicit metrics, matching `docs/LLM_ORCHESTRATION.md` §Evaluation's system-level scenarios but focused on the retrieval step in isolation:

| Metric | Definition | Test method |
|---|---|---|
| Retrieval precision | Fraction of retrieved chunks actually relevant to the query | Hand-labeled query set (build ~30-50 queries covering all 3 collections during Phase 3/4) |
| Retrieval recall | Fraction of all relevant chunks that were retrieved | Same labeled set, requires knowing the full relevant-chunk ground truth per query |
| Faithfulness | Does the LLM's answer only state what the retrieved context supports? | Automated: NLI-style entailment check of answer sentences against retrieved chunks, or manual review for the initial evaluation set |
| Citation correctness | Does every cited source actually contain the claim attributed to it? | Manual spot-check + automated string-presence check for quoted claims |
| Hallucination rate | Fraction of answers containing an unsupported factual claim (especially numeric) | Manual review against the FACT/INTERPRETATION tagging (`docs/LLM_ORCHESTRATION.md` §Reliability) |
| Answer relevance | Does the answer address the actual question asked? | Manual rubric, small evaluation set |
| Latency | End-to-end time from question to answer | Automated timing, per scenario type (API-only vs. API+RAG vs. multi-call) |

**Test plan**: build the ~30-50 query evaluation set during Phase 4 (RAG retrieval, `docs/PRODUCT_ROADMAP.md`), covering all three collections and a range of specificity (broad "what is t5" to narrow "why does the Semi-HMM's negative binomial duration model matter for tPB2"). Re-run this fixed set after every re-ingestion to catch retrieval regressions — this is the RAG-specific analogue of this project's own frame-shuffle/reproducibility-check discipline already established for the scientific models.

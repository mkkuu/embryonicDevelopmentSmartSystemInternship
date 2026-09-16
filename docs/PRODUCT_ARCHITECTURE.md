# Product Architecture

**Update, 2026-08-26**: the "LLM ORCHESTRATOR" box in §3's diagram below now has a real (foundation-only) implementation behind it — `Training/orchestrator/`, see `docs/RAG_ORCHESTRATION_PHASE_REPORT.md` for the full build report and `docs/TOOL_CONTRACTS.md`/`docs/LLM_PROMPT_CONTRACT.md` for the new companion docs this added. §1's status table and §4a are updated accordingly; the rest of this document (§3's diagram, §4's separation-of-responsibilities table, §9-13) is unchanged and still accurate.

Planning document only for everything not listed as IMPLEMENTED in §1 below. Companion docs: `docs/RAG_ARCHITECTURE.md`, `docs/LLM_ORCHESTRATION.md`, `docs/TOOL_CONTRACTS.md`, `docs/LLM_PROMPT_CONTRACT.md`, `docs/WEBAPP_ARCHITECTURE.md`, `docs/WEBAPP_INTEGRATION_PLAN.md`, `docs/PRODUCT_ROADMAP.md`. State verified against real code this session (`Training/orchestrator/`, re-checking `Training/reporting/`), not assumed from prior documentation.

## 1. Current state (verified against code, not docs)

| Component | Status | Evidence |
|---|---|---|
| Embeddings (ResNet18) | IMPLEMENTED | `Embeddings/resnet18/{train,val,test}/`, checksum-verified provenance |
| Semi-HMM k=7 (scientific model) | IMPLEMENTED, frozen | `Results/evaluation/semi_hmm_weekend_phaseF/model/`, Val-validated |
| Reporting API (code) | IMPLEMENTED | `Training/reporting/*.py`, 58 new tests, real-data integration test passing |
| Reporting API (documentation) | IMPLEMENTED (as of this pass) | `docs/REPORTING_API.md`, `docs/INFERENCE_SCHEMA.md` |
| Reporting Cache (Phase 1.5) | IMPLEMENTED | `Training/reporting/{cache,warm_cache}.py`, 38,374x measured speedup, exact numerical equivalence — see `docs/REPORTING_CACHE.md` |
| Reporting API (production deployment) | MISSING | dev-only Flask entry point, no WSGI server, no process manager, no auth; pre-warm not yet scheduled |
| Vector DB / RAG index | IMPLEMENTED (Phase 3, foundation) | Chroma, `RagIndex/`, 30 documents / 432 chunks, idempotent ingestion — see `docs/RAG_INGESTION.md` |
| RAG retrieval | IMPLEMENTED (Phase 3, prototype only) | `Training/rag/retrieval.py`, smoke-benchmarked (mean P@5=0.44, MRR=0.56) — see `docs/RAG_RETRIEVAL.md`; no answer synthesis, no LLM |
| LLM orchestrator | FOUNDATION IMPLEMENTED (this phase), no real LLM connected | `Training/orchestrator/{router,context_builder,llm_provider,orchestrator}.py` — deterministic router (18/18 on the evaluation set), provenance-preserving context builder, `LLMProvider` abstraction with only `NullLLMProvider` registered. See `docs/RAG_ORCHESTRATION_PHASE_REPORT.md` |
| Tool calling | FOUNDATION IMPLEMENTED (minimum 5-tool set), not wired to a real LLM | `Training/orchestrator/tools.py` + `docs/TOOL_CONTRACTS.md` — `get_current_inference`, `get_trajectory`, `get_model_metadata`, `get_phase_information`, `retrieve_documents`, each a typed wrapper around an existing Reporting API/RAG function; no function-calling loop exists (no LLM to drive one) |
| Web App | MISSING | only requirements captured (`docs/WEBAPP_DATA_REQUIREMENTS.md`); `WebApplication/` is a different, unrelated Flask app (doctor/admin CRUD) |
| Data/model versioning scheme | PARTIALLY IMPLEMENTED | model config is inspectable (`model_loader.model_version_string`), embeddings have SHA-256 provenance in `manifest.json` — no unified scheme across RAG index/prompts/LLM config yet |
| Deployment infrastructure | MISSING | no Docker, no compose file, no secrets management beyond `WebApplication/`'s own unrelated `.env` |

**Update (Phase 1.5, complete)**: the ~333s/video cold-cost finding below has been **mitigated** — a disk-backed cache (`docs/REPORTING_CACHE.md`) now persists each video's forward pass across process restarts. Measured speedup on a real Val trajectory: **38,374x** (358.5s → 0.009s), with exact (`np.array_equal`) numerical equivalence confirmed between cached and uncached output. The finding itself (why it was slow) remains true and is preserved below for context; it is no longer an unmitigated blocker for Phase 6.

**One severe finding this session that shaped this plan, now addressed**: `SemiHMMModel.next_phase_distribution()` costs **~333-357 seconds per video, cold** (pure-Python O(T·K·Dmax) loop, unvectorized, in the frozen scientific model — not touched, not optimized). See §9 (Performance) for the resolution.

**Another new finding**: the raw 15-way `phase_probabilities` output is severely miscalibrated (Brier≈0.96, worse than uniform) even though ranking is fine. Any UI/LLM surface must present it as a ranking, never as a calibrated confidence — see `docs/REPORTING_API.md`'s Known Limitations and `Results/evaluation/global_model_comparison/PRAUC_REPORT.md`.

## 2. Product objective

A user (researcher/clinician) should be able to: **(A) explore** a trajectory (pick an embryo/video, scrub the timelapse, land on a window, see its timestamp and ground-truth phase if available); **(B) understand** the model's prediction at that window (current phase, full distribution, entropy, next-phase distribution, expected duration + interval, and — for a completed video — the segment-level transition chain); **(C) ask** natural-language questions spanning live predictions ("what phase are we in"), explanation ("why t5"), and project/methodology knowledge ("why Semi-HMM over GRU").

## 3. Target architecture

The proposed reference architecture is **fundamentally sound and is adopted with two refinements**, not replaced:

```
                         USER
                          │
                          ▼
                       WEB APP
                          │
                          ▼
                 APPLICATION API (BFF)
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       REPORTING API             LLM ORCHESTRATOR
      (source of truth              │
       for all numbers)    ┌────────┼────────┐
              │             │        │        │
              │             ▼        ▼        ▼
              │           RAG     TOOLS      GENERAL LLM
              │        RETRIEVAL  (wrap            │
              │             │    Reporting API)    │
              │             └────────┴─────────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
                       RESPONSE
                          │
                          ▼
                       WEB APP
```

**Refinement 1 — "Tools" are not a new data path, they are typed wrappers around the same Reporting API** the visualization panels call directly. This is the single most important structural decision in this plan (see §27's constraint: the Reporting API is the sole source of truth for predictions) — the LLM never gets a second, parallel way to obtain a number.

**Refinement 2 — the "Application API" (BFF) is a thin routing/composition layer, not a third data source.** For pure structured-data requests (the Visionneuse/Analysis panels), the Web App can call Reporting API endpoints **directly**, bypassing the BFF and the LLM entirely — lower latency, no LLM cost, no hallucination surface for numbers that don't need natural-language framing. The BFF/orchestrator path is reserved for the Chat panel. Whether the BFF is a physically separate service or folded into the orchestrator is a deployment decision, not an architectural one (see §12 Decisions Required).

## 4. Separation of responsibilities

| Layer | Owns | Never does |
|---|---|---|
| **Model** (Semi-HMM, frozen) | Predictions, probabilities, transitions, durations — via its own already-implemented methods | Is never called directly by anything except `inference_service` |
| **Reporting API** | Exposing structured outputs, trajectories, metadata; the *only* source of numeric prediction values in the whole system | Never fabricates a value; never interprets/explains |
| **RAG** | Scientific/project/methodological documentation, retrieval | Never stores or serves current prediction values (§ RAG_ARCHITECTURE.md) |
| **LLM** | Understanding the question, reasoning, selecting information, synthesizing, explaining, producing natural language | Never a source of truth for a number the Reporting API can supply; never "corrects" a model output |
| **Web App** | Visualization, interaction, timelapse, charts, chat UI | Never computes a prediction or duration itself |

## 4a. GRU as a candidate event/temporal signal (design note, not implemented)

Added 2026-08-25 following an intermediate GRU-vs-Identity-Dynamics threshold/sensitivity analysis (`docs/GRU_IDENTITY_ANALYSIS.md`, source data `Results/evaluation/gru_identity_threshold_analysis/`). **Not implemented** — the GRU (`Training/evaluation/models/gru.py`) is not wired into `Training/reporting/` or any Reporting API endpoint today; only the frozen Semi-HMM is. This section records a considered, evidence-based design option for a future phase, not a change to the current architecture.

**Finding**: on the Val split, the GRU (classification formulation) ranks positives above negatives more reliably than Identity Dynamics (AUROC 0.764 vs 0.750, PR-AUC 0.357 vs 0.325) and offers a genuine high-sensitivity regime (90.4% recall at threshold=0.059), but at that regime its false-positive rate is 56.8%, its best-F1 operating point only reaches F1=0.379, and its probability is not well calibrated (ECE=0.022). See `docs/GRU_IDENTITY_ANALYSIS.md` for the full analysis, including the explicit caveat that these GRU-vs-Identity-Dynamics differences are point estimates, not statistically tested.

**Where a GRU signal would fit, if ever built**: as an additional, clearly-labeled **event/temporal signal** — e.g. "elevated transition probability at window W" — feeding the Reporting API or the future LLM orchestrator's tool layer alongside the Semi-HMM's own outputs, never replacing them. Concretely, this would extend the **Model** row of §4's table with a second entry:

| Layer | Owns | Never does |
|---|---|---|
| **GRU (candidate, not implemented)** | A bounded-confidence event/transition signal and approximate temporal localization | Is never treated as a verified biological fact; never a substitute for the Semi-HMM's own probabilistic outputs; never presented to a user as a calibrated probability without the caveats in `docs/GRU_IDENTITY_ANALYSIS.md` §10-11 |

This mirrors §4's existing discipline exactly: the Reporting API remains the sole source of truth for any number; a GRU signal, if integrated, would be exposed as one more structured field with its own explicit provenance/uncertainty tagging (see `docs/LLM_ORCHESTRATION.md`'s FACT/MODEL OUTPUT/INTERPRETATION separation), never silently blended with the Semi-HMM's phase posterior. Building this is **not scheduled** — it is not part of the current roadmap (`docs/PRODUCT_ROADMAP.md`) and would need its own scoped phase (API contract, calibration/validation work, and likely the second/third GRU seed's results) before implementation, matching this project's established discipline for touching the Reporting layer.

## 5. Reporting API

**Already implemented and documented** — see `docs/REPORTING_API.md` for the full endpoint table, example response, error handling, and known limitations. No new endpoints are required to support the RAG/LLM layer: every tool defined in `docs/LLM_ORCHESTRATION.md` §Tools maps 1:1 onto one of the 7 existing endpoints. This is a deliberate validation of the original design, not a coincidence — the schema (`docs/INFERENCE_SCHEMA.md`) was built from the model's real capabilities, and the model's real capabilities are exactly what the product needs.

## 6. RAG

See `docs/RAG_ARCHITECTURE.md` for the full design (knowledge categories, vector DB choice, retrieval pipeline, evaluation plan). Summary: three static knowledge categories (scientific, project, methodological) live in the vector DB; **current prediction values never do** — always fetched live from the Reporting API.

## 7. LLM orchestration

See `docs/LLM_ORCHESTRATION.md` for the full design (routing logic, tool definitions, reliability guardrails, citation format, evaluation scenarios).

## 8. Web App

See `docs/WEBAPP_ARCHITECTURE.md` for panel breakdown and 3 user journeys.

## 9. Performance

Costly operations identified, all measured or reasoned from real evidence this session — nothing here is a guess:

| Operation | Cost | Mitigation |
|---|---|---|
| `SemiHMMModel.next_phase_distribution()`, cold (first request per video, ever) | **~333-357s / video** (measured) | **DONE (Phase 1.5)**: disk-backed cache (`docs/REPORTING_CACHE.md`), keyed by `(model_version, split, video)`, persists across process restarts. Measured: 38,374x speedup on repeat requests, exact numerical equivalence. First-ever request per video still pays the full cost — mitigated by `Training/reporting/warm_cache.py` (built, not yet scheduled — a full 106-video Val pre-warm is ~9.7h sequential, parallelizable, an operational Decision Required, not an architectural gap) |
| `SemiHMMModel.filtering()`, cold | ~1.3-1.7s / video (measured) | Cheap; covered by the same cache as a side effect |
| Same operation, cache hit | **~0.01s** (measured) | — |
| Vector search (RAG) | Not yet measurable (no corpus built) — expected sub-100ms for a corpus this size (low hundreds of chunks) with any of the compared vector DBs | None needed pre-emptively; measure once built |
| LLM calls (external API) | Provider-dependent, typically 1-10s per call, more for multi-tool-call chains | Stream partial responses to the Web App; cap tool-call chain depth (§ LLM_ORCHESTRATION.md) |
| Embedding generation for new RAG documents | One-time/batch, not per-request | Ingestion-time only, not a serving-path concern |

**Do not over-engineer**: no need for a distributed cache (Redis etc.) at this scale — an in-process cache (already implemented in `inference_service.py`) is sufficient for a single-instance deployment; revisit only if/when horizontal scaling of the Reporting API is actually needed.

## 10. Security and robustness

- Parameter validation: already implemented (`trajectory_service._check_split`, `ValueError`/`KeyError` handling) — extend the same discipline to any new orchestrator/RAG endpoint.
- Rate limiting: not yet needed at current scale (internal research tool, not public-facing) — flag as a pre-public-launch requirement, not now.
- Timeouts: **critical** given the 333s cold-inference cost — any synchronous tool call must have a timeout well above normal warm-cache latency but must not silently hang the whole chat turn; prefer async job + polling/streaming for cold requests (Decision Required, §12).
- Model unavailable: `model_loader.get_model()` already raises `FileNotFoundError`/`ModelIncompatibleError` explicitly — the orchestrator must catch these and report "model unavailable" rather than silently falling back to LLM invention.
- Missing documents (RAG): retrieval returning zero results must produce an explicit "no documentation found" state the LLM is instructed to surface, not paper over.
- Hallucination: addressed structurally, not just by prompting — see `docs/LLM_ORCHESTRATION.md` §Reliability.
- Logs: every inference/tool call/RAG retrieval/LLM call should log `model_version`, `sample_id`, `window`, `timestamp`, and (for LLM calls) prompt/response — needed for §13 auditability and §20 versioning.
- Model version / document version: see §11.

## 11. Versioning

| Artifact | Versioning approach |
|---|---|
| Scientific model | Already versioned implicitly via `state.json`'s hyperparameters + `model_loader.model_version_string()`; recommend freezing this as a stable string, e.g. `semi_hmm_weekend_phaseF@<git-commit-of-Training/evaluation/models/semi_hmm.py>` |
| Embeddings | Already versioned: SHA-256 of the source checkpoint, recorded in `manifest.json` |
| Data / splits | `Data/Splits/F0.csv`, currently unversioned beyond its own content — recommend a content hash recorded alongside any derived artifact |
| Results | Already versioned by directory name + date in `Results/evaluation/*` (project convention) |
| Documentation | Currently git-ignored (local-only) — if RAG-indexed, needs its own version marker (see below) |
| RAG index | **New, must design**: tag every ingested chunk with `source_doc_version` (e.g. file mtime + content hash) and `index_build_id` (timestamp of the ingestion run); a stale chunk is detectable by comparing source hash to what's indexed |
| Prompts | **New, must design**: store orchestrator/system prompts as versioned files (not inline strings), tagged `prompt_version` in every LLM call log |
| LLM config | **New, must design**: log model name + provider + temperature/params per call, so "which LLM answered this" is always reconstructable |

**Goal check**: "which model version produced this prediction" is already answerable today via `InferenceRecord.model.model_version` (see `docs/REPORTING_API.md`'s example response) — this is a real, working capability, not aspirational.

## 12. Decisions required

See the consolidated list at the end of this plan (shared across all five docs, not duplicated per-file).

## 13. Risks

- **Cold-inference latency (333s/video)** is the single largest technical risk to product usability — must be mitigated before Phase 6 (tool calling) is exposed to real users, not discovered in production.
- **LLM hallucination on numeric claims** — mitigated structurally (§ LLM_ORCHESTRATION.md), but never fully eliminable; requires ongoing evaluation (§21/§22).
- **Phase-probability miscalibration** (Brier≈0.96) — a UI/LLM presenting `phase_probabilities` as a percentage confidence would be actively misleading; must be designed around from day one of the Web App, not patched later.
- **RAG staleness** — if the scientific docs are revised (as they have been continuously this project) without re-ingestion, the RAG answers stale/contradicted content. Needs a re-ingestion trigger tied to doc changes (manual for now, given corpus size — see roadmap Phase 3).
- **Scope creep** — 28 sections of design is a lot of surface area; the roadmap (§ PRODUCT_ROADMAP.md) is deliberately incremental specifically to manage this.
- **External LLM + medical-adjacent data** — real privacy/compliance exposure if raw embryo imagery or patient identifiers ever reach an external LLM API; the architecture must guarantee only derived, already-anonymized structured facts and documentation text cross that boundary (see `docs/LLM_ORCHESTRATION.md` §Offline/Privacy) — this is a Decision Required, not a default.

## 14. Recommended implementation order

See `docs/PRODUCT_ROADMAP.md` for the full phase-by-phase plan. One-line summary: Phase 1 (Reporting API) is **already done** — the immediate next step is closing its performance gap (pre-warming), not starting RAG. See NEXT STEP at the end of this response.

# Product Roadmap

Planning only — no phase below has been started except Phase 1, which is **already complete**. Order follows the request's own 10-phase structure with one adjustment (a performance sub-step inserted before Phase 2 proper) justified by real evidence from this session, not a preference.

## Phase 1 — Reporting API — **DONE**

- **Objective**: expose the frozen Semi-HMM's outputs as a stable service, independent of the model's internals.
- **Files**: `Training/reporting/{__init__,schemas,model_loader,trajectory_service,inference_service,api}.py`, `Tests/reporting/*.py`.
- **Dependencies**: frozen model (`semi_hmm_weekend_phaseF`), embeddings cache — both already existed.
- **Tests**: 287/287 passed (229 pre-existing + 58 new).
- **Validation criteria**: model loads and validates; a real Val trajectory loads; a window can be selected; inference returns probabilities/next-phase/duration/uncertainty; timestamps exposed without inventing a unit; Test never touched — **all met**.
- **Complexity**: medium (delivered).
- **Risk realized, now known**: `next_phase_distribution()` cold cost (~333s/video) — carries forward as Phase 1.5 below.

## Phase 1.5 — Performance mitigation — **DONE**

- **Objective**: close the single largest risk to every downstream phase before building on top of it — persist the per-video forward-pass result to disk instead of paying ~333-357s on every process's first request for a video.
- **Files**: `Training/reporting/cache.py` (disk-backed cache, `fcntl.flock` concurrency, atomic writes), `Training/reporting/warm_cache.py` (pre-warm driver, reuses the same cache function), `Training/reporting/inference_service.py` (minimal, surgical change routing `_forward_pass` through the new cache), `Tests/reporting/test_cache.py` (13 new tests), `.gitignore` (new `Cache/` entry).
- **Dependencies**: Phase 1.
- **Tests**: cache mechanics (hit/miss/corruption/version-isolation/lock-release) against a `FakeModel`, isolated from model correctness; full regression suite re-run.
- **Validation criteria — all met**: cached result exactly (`np.array_equal`) matches the uncached result; cache-hit latency ~0.009s vs. ~358s before (38,374x); invalidation confirmed for version/split/video/`window_starts` mismatches and corrupted files; concurrency lock confirmed to release cleanly; no scientific artifact modified (checksums identical before/after).
- **Complexity**: low-medium, as estimated.
- **Risk retired**: the cold-inference-latency risk flagged throughout `docs/PRODUCT_ARCHITECTURE.md` is no longer unmitigated. **Remaining, smaller risk carried forward**: `warm_cache.py` is built but not scheduled anywhere — the *first* request for any not-yet-warmed video still pays the full cost once; an operational decision (deploy hook vs. cron vs. on-demand) for a later phase, not a re-opened technical risk.

## Phase 1.6 — `next_phase_distribution()` optimization — **DONE**

- **Objective**: close the remaining first-computation cost that Phase 1.5's cache could not address — precompute the duration-model `log_prob`/`log_survival` lookup tables inside `next_phase_distribution()`, mirroring the fix `_explicit_duration_forward_optimized()` already applies to its sibling recursion.
- **Files**: `Training/evaluation/models/semi_hmm.py` (the pre-Phase-1.6 body preserved verbatim as `_next_phase_distribution_reference()`, a new `_next_phase_distribution_optimized()`, public `next_phase_distribution()` now a thin dispatcher to the optimized path), `Tests/evaluation/models/test_semi_hmm.py` (9 new equivalence tests).
- **Dependencies**: Phase 1.5.
- **Tests**: 9/9 new reference-vs-optimized equivalence tests (various T/dmax/n_states/duration family/transition smoothing/edge cases), full suite 308/308.
- **Validation criteria — all met**: bit-identical (`np.array_equal`) output on synthetic fixtures and on a real Val trajectory (`Patient_319`); cache miss/hit re-verified equivalent against the reference oracle; no scientific artifact (frozen model, embeddings, historical results) modified; Test never touched.
- **Complexity**: low (table precomputation only, no algorithm change, direct precedent in the same file).
- **Measured result**: `next_phase_distribution()` 326.22s → 8.71s (**37.4x**); total forward pass (filtering + next_phase_distribution) 327.75s → 10.25s (**32.0x**). Does not reach `filtering()`'s own ~1.5s — the remaining cost is Python-level loop overhead, not duration-model calls; further vectorization was deliberately out of scope for this phase. Full detail: `docs/SEMI_HMM_PHASE_1_6_REPORT.md`.
- **Risk retired**: the *first-request-per-video* cold cost (the one risk Phase 1.5 explicitly could not address) is now ~32x smaller. `warm_cache.py`'s ~9.7h full-Val pre-warm estimate (Phase 1.5) is expected to drop proportionally but was not re-benchmarked end-to-end this phase — a real re-measurement, not this ratio, should be used before relying on a new number operationally.

## Phase 2 — Data schemas / validation (hardening, not rebuilding) — **DONE (foundation, minimum tool set)**

- **Objective**: harden `Training/reporting/schemas.py`'s existing dataclasses into a formal tool-calling contract (JSON Schema or equivalent) the LLM orchestrator can rely on for structured tool outputs, and add request-parameter validation (video/window/split) at the tool layer, not just the API layer.
- **Files**: `docs/TOOL_CONTRACTS.md` (the formal contract, prose + schema per tool), `Training/orchestrator/tools.py` (the executable half — typed wrappers, request-parameter validation via a 3-way `ToolError` taxonomy, `split="test"` rejected a third time at this layer).
- **Dependencies**: Phase 1.
- **Tests**: `Tests/orchestrator/test_tools_reporting.py`/`test_tools_rag.py` (skip-guarded integration, not yet run — see `docs/RAG_ORCHESTRATION_PHASE_REPORT.md` §5), `Tests/orchestrator/test_router_evaluation.py` (asserts no tool crosses the RAG/Reporting boundary).
- **Validation criteria — met for the 5-tool minimum set** (`get_current_inference`, `get_trajectory`, `get_model_metadata`, `get_phase_information`, `retrieve_documents`): each has a machine-checkable input/output contract in `docs/TOOL_CONTRACTS.md`, matched 1:1 by `tools.py`'s real signature. **Not done**: the remaining tools listed in `docs/LLM_ORCHESTRATION.md` §2 (`get_frame`, `get_phase_probabilities`, `get_next_phase_distribution`, `get_duration_distribution`, `get_transition_chain`, `search_knowledge`, `get_experiment_result`) — deliberately deferred, see `docs/TOOL_CONTRACTS.md`'s closing section.
- **Complexity**: low (builds directly on already-tested code).
- **Risk**: low.

## Phase 3 — Vector DB + ingestion — **DONE (foundation)**

- **Objective**: stand up Chroma (recommendation, `docs/RAG_ARCHITECTURE.md`), chunk and ingest a curated set of the project's documentary knowledge, build a retrieval prototype, and evaluate it with a smoke benchmark.
- **Files**: `Training/rag/{__init__,schemas,inventory,chunker,hashing,embeddings,vectorstore,ingest,retrieval,eval_benchmark}.py`, `Tests/rag/*.py` (43 new tests), `RagIndex/` (new gitignored top-level dir, the persisted Chroma store).
- **Dependencies**: Phase 1 (Reporting API, for the architectural separation this phase enforces), the existing `docs/*.md` corpus. New pip dependencies added to `requirements.txt`: `chromadb`, `sentence-transformers`.
- **Scope decision, made explicitly not by default**: source inventory is `docs/*.md` only, hand-curated (30 of 39 files included, 9 excluded with stated reasons — session logs, repo-hygiene meta-docs, session-resume scratch notes) — the RAG does **not** blindly ingest the whole repository. `Results/evaluation/*/REPORT.md` (raw per-experiment knowledge) is scoped out of this pass, a documented future follow-up, not an oversight.
- **Tests**: 43 new (hashing, schemas, chunking — synthetic fixtures, always run; inventory-vs-disk drift check; ingestion idempotence/hash-change/prune, real chromadb+sentence-transformers, skip-guarded; retrieval correctness + metadata filtering, same). Full suite **351/351** passed (308 existing + 43 new), server-verified.
- **Validation criteria — all met**: real ingestion run against the full curated corpus (30 docs → 432 chunks, ~22s); idempotence confirmed empirically (identical immediate re-run: 0 ingested, 30 skipped, chunk count unchanged); retrieval prototype tested against this task's 6 required real questions plus 4 more (10-question hand-labeled smoke benchmark: mean Precision@5=0.44, mean Recall@5=0.64, MRR=0.56 — explicitly not claimed as a robust evaluation); data/RAG separation verified directly (a live-prediction question returns zero fabricated numeric chunks, because none exist in the corpus to retrieve — a structural guarantee, not a policy).
- **Complexity**: medium-high (chunking quality mattered in practice — an early English-only embedding model measurably failed French queries against this English corpus, caught by the benchmark and fixed by switching to a multilingual model, not assumed correct).
- **Risk retired**: "the RAG might ingest the whole repository indiscriminately" — closed by the hand-curated inventory. **Risk surfaced, not yet closed**: several real content gaps found by the smoke benchmark (e.g. no single clean "what is Semi-HMM"/"why is Test locked" chunk) — a documentation-content problem, not a retrieval-code problem; `docs/RAG_PHASE_3_REPORT.md` §11/§16 has the detail and a recommended next step.
- Full build report: `docs/RAG_PHASE_3_REPORT.md`. Day-to-day docs: `docs/RAG_INGESTION.md`, `docs/RAG_RETRIEVAL.md`.

## Phase 4 — RAG retrieval

- **Objective**: implement the retrieval pipeline (`docs/RAG_ARCHITECTURE.md` §4) — intent-aware, multi-collection, metadata-filtered.
- **Files**: `Training/rag/retrieval.py` or similar.
- **Dependencies**: Phase 3.
- **Tests**: the ~30-50 query evaluation set (`docs/RAG_ARCHITECTURE.md` §5) — precision/recall against hand labels.
- **Validation criteria**: retrieval precision/recall meet a bar set once the first evaluation run establishes a baseline (no target number invented here without data).
- **Complexity**: medium.
- **Risk**: low-quality chunking from Phase 3 would surface here as poor retrieval — expect at least one iteration back to Phase 3.

## Phase 5 — LLM orchestrator — **FOUNDATION DONE, no real LLM connected**

- **Objective**: routing logic (`docs/LLM_ORCHESTRATION.md` §1), context assembly, the FACT/DOCUMENTED/INTERPRETATION separation.
- **Files**: `Training/orchestrator/{router,tools,context_builder,llm_provider,orchestrator,eval_questions,run_orchestration_evaluation}.py`, `Tests/orchestrator/*.py` (50 new tests).
- **Dependencies**: Phase 2 (tool contracts — done, minimum set), Phase 3 (RAG retrieval prototype — done, foundation). Phase 4's fuller retrieval pipeline (intent detection, multi-collection merge) is still not built — this phase's router/context-builder work directly against Phase 3's existing prototype instead of waiting on it. An LLM choice remains a Decision Required, unresolved — `NullLLMProvider` is the only registered provider.
- **What's done**: a deterministic 4-category router (`DOCUMENTARY`/`DYNAMIC_DATA`/`HYBRID`/`UNKNOWN`, collapsing `docs/LLM_ORCHESTRATION.md` §1's 5-way table per the task's own scope), deterministic tool selection per category (`orchestrator.plan_tools()`), a context builder enforcing the FACT/DOCUMENTED physical separation structurally (not by prompt convention), and an `LLMProvider` abstraction with a working `NullLLMProvider`. See `docs/RAG_ORCHESTRATION_PHASE_REPORT.md` for the full build report.
- **What's NOT done (unchanged as of the original foundation pass)**: no real LLM wired in; no session/context-state resolution (`video_id`/`window` must be passed explicitly, `docs/WEBAPP_INTEGRATION_PLAN.md`); no multi-call/cross-trajectory-aggregate planning.
- **Update, 2026-08-26 (later session)**: 401/401 full-suite pytest confirmed on the GPU server (real run, not just direct execution — see `docs/PROJECT_CHECKPOINT.md`'s "VALIDATED" checkpoint), 18/18 routing/tool-selection confirmed against the real Reporting API + real `RagIndex/` (`Results/evaluation/orchestration_foundation_eval/`). The grounding check originally deferred to Phase 6 below is now also implemented — see that entry.
- **Tests**: 18-question routing-category evaluation (`Tests/orchestrator/test_router_evaluation.py`), **18/18 correct**, confirmed via real pytest on the GPU server; context-assembly unit tests confirming `document_context`/`dynamic_context` are never merged.
- **Validation criteria**: all 4 routing categories correctly classified on an 18-question labeled set (>= task's 15-question minimum) — **met and confirmed for real**.
- **Complexity**: high (first genuinely new orchestration logic) — delivered at foundation scope; wiring a real LLM remains its own, larger future step.
- **Risk**: routing misclassification silently degrades answer quality without an obvious error — the evaluation harness (`run_orchestration_evaluation.py`) has now been run against the real system, risk substantially retired for routing/tool-selection specifically (not for a real LLM's own answer quality, which no provider has been evaluated on yet).

## Phase 6 — Tool calling — **DONE (grounding check + Grounded Answer Contract), no real LLM connected**

- **Objective**: wire the tool definitions (`docs/LLM_ORCHESTRATION.md` §2) into the orchestrator's function-calling loop, plus the grounding check (§3).
- **Files**: `Training/orchestrator/tools.py` (Phase 5), `Training/orchestrator/grounding_check.py` (this phase, 2026-08-26), `Training/orchestrator/{llm_provider,prompt,orchestrator}.py` (extended), `Training/orchestrator/run_llm_benchmark.py`.
- **Dependencies**: Phase 5 (done), Phase 1.5 (cold-cost mitigation — done, `docs/REPORTING_CACHE.md`).
- **What's done**: the "function-calling loop" this phase asked for is `orchestrator.execute_plan()` (built in Phase 5, calling the real tool contracts) — this phase adds the piece Phase 5 explicitly deferred: `grounding_check.py`, applied to **every** provider's output by `orchestrator.generate_grounded_response()`, never just trusting a provider's own `grounded` self-report (both downgrading a false claim and upgrading an overly conservative true one). The Grounded Answer Contract (`LLMResponse`: `sources`/`dynamic_data`/`used_tools`/`confidence`), the first system prompt (`prompt.py`), and timeout/provider-failure handling are also new this phase. Full detail: `docs/GROUNDED_GENERATION.md`.
- **What's NOT done**: no real (external/local) LLM connected — `docs/LLM_PROVIDER_DECISION.md` is an open Decision Required; only `TemplateLLMProvider` (deterministic, zero-network) is registered as a real provider.
- **Tests**: grounding-check tests with deliberately-injected wrong numbers (`Tests/orchestrator/test_grounded_generation.py::test_hallucinated_number_is_caught_even_when_provider_claims_grounded`) confirm detection; 33 new dependency-free tests total across `test_grounding_check.py`/`test_prompt.py`/`test_grounded_generation.py`.
- **Validation criteria**: "LLM never returns a number that doesn't match a tool output" — **mechanically enforced and tested** for the registered providers (`TemplateLLMProvider` by construction, a hand-written lying fake provider caught and corrected); **not yet evaluated for a real LLM's actual tendency to hallucinate**, since none is connected (`docs/LLM_EVALUATION.md`).
- **Complexity**: high, as estimated.
- **Risk**: hallucination risk from a REAL LLM remains untested until Phase 5/6's provider decision is made — the mechanism that would catch it is built and tested against synthetic failures, but has never seen a real LLM's actual failure modes yet.

## Phase 7 — Web App visionneuse

- **Objective**: the Visionneuse + Analyse panels (`docs/WEBAPP_ARCHITECTURE.md` §2-4), talking directly to the Reporting API (no LLM needed for this phase).
- **Files**: new frontend project (framework: Decision Required).
- **Dependencies**: Phase 1 only — deliberately does not depend on Phases 3-6, so it can be built and demoed in parallel with the RAG/LLM work.
- **Tests**: component tests, one end-to-end test against a real running Reporting API instance.
- **Validation criteria**: matches the 14-point success criteria list from the original Reporting API task, now at the UI layer (probabilities exposed, next phase shown, duration shown when available, timestamps shown without an invented unit, etc.).
- **Complexity**: medium (frontend engineering, not novel design — the requirements are already fully specified in `docs/WEBAPP_DATA_REQUIREMENTS.md`).
- **Risk**: frame image serving is a real, currently-unaddressed gap (§ WEBAPP_ARCHITECTURE.md §2) — needs its own small scoped task, not assumed to fall out of existing endpoints.

## Phase 8 — Chat integration

- **Objective**: wire the Chat panel to the Application API / LLM Orchestrator.
- **Files**: frontend chat component; `Application API` BFF routes (new, thin).
- **Dependencies**: Phase 5-7.
- **Tests**: the 7 scenarios (A-G, `docs/LLM_ORCHESTRATION.md` §5) run end-to-end through the real UI.
- **Validation criteria**: citations visible, FACT vs. INTERPRETATION visually distinguishable, "data used" disclosure present.
- **Complexity**: medium.
- **Risk**: UX risk of over-trusting the chat's numeric answers if the visual FACT/INTERPRETATION distinction isn't strong enough — worth a dedicated design pass, not an afterthought.

## Phase 9 — Évaluation complète

- **Objective**: run the full RAG (`docs/RAG_ARCHITECTURE.md` §5) and system-level (`docs/LLM_ORCHESTRATION.md` §5) evaluation suites for real, establish baselines.
- **Files**: `Tests/rag/`, `Tests/orchestrator/`, an evaluation report doc.
- **Dependencies**: Phases 3-8 all functional.
- **Tests**: the metrics themselves (retrieval precision/recall, faithfulness, citation correctness, hallucination rate, answer relevance, latency).
- **Validation criteria**: every metric has a real, measured baseline (not a target invented in advance) and a documented plan for re-running after any future change to models/docs/prompts.
- **Complexity**: medium (mostly harness-building, since the metrics are already defined).
- **Risk**: evaluation is exactly the phase most likely to get cut under time pressure — flagged explicitly as not optional, per the original request's own emphasis ("ne pas se contenter de tester si le chatbot fonctionne").

## Phase 10 — Déploiement

- **Objective**: package and deploy per `docs/PRODUCT_ARCHITECTURE.md` §Deployment (once decided — see Decisions Required).
- **Files**: `Dockerfile`s, `docker-compose.yml`, deployment docs.
- **Dependencies**: everything above, plus the Decisions Required list resolved (framework, vector DB, LLM, hosting).
- **Tests**: deployment smoke test (health checks across all services).
- **Validation criteria**: a fresh environment can be stood up from the repo + documented secrets, reproducibly.
- **Complexity**: medium-high, mostly ops rather than novel logic.
- **Risk**: secrets/credentials management for an external LLM API key, vector DB credentials, etc. — needs the same discipline already established for `WebApplication/`'s `.env` pattern, not a new ad hoc scheme.

## Why this order, not another

Phase 7 (Web App visionneuse) is placed after Phase 6 in the request's own ordering but is **deliberately independent of Phases 3-6** here — it only needs Phase 1 (already done). A team could build Phase 7 in parallel with Phases 3-6, delivering visible product value (a working visualization tool) well before the RAG/LLM stack is ready, rather than waiting for the full chain. This isn't a reordering of the numbered phases, just a note that the dependency graph is not strictly linear — worth exploiting if parallel work is possible.

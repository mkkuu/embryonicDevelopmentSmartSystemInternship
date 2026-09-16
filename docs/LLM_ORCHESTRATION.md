# LLM Orchestration

**Update, 2026-08-26 (Phase 6)**: this document's design is now real, tested code — `Training/orchestrator/{router,tools,context_builder,llm_provider,grounding_check,prompt,orchestrator}.py`. §1's routing table, §2's tool table, §3's reliability guardrails (including the point-2 grounding check), and §4's citation format are all implemented; see `docs/GROUNDED_GENERATION.md` for the executable detail and `docs/TOOL_CONTRACTS.md` for the tool contracts. **Still no EXTERNAL LLM API is called anywhere in this repository** — `docs/LLM_PROVIDER_DECISION.md` is the current, still-open Decision Required for which real provider to connect; only a deterministic, zero-network `TemplateLLMProvider` is registered as a real (non-null) provider so far.

Originally planning only — no LLM API called, no orchestrator code written. Defines routing, tools, reliability guardrails, citations, and evaluation scenarios for the layer between the Application API and the general-purpose LLM.

## 1. Routing logic

The orchestrator classifies each question into one of five categories **before** doing any retrieval or generation — this classification itself can be a cheap LLM call (function-calling / structured output) or a lighter classifier; either way it is a discrete, inspectable step, not folded invisibly into one giant prompt.

| Category | Example | Needs |
|---|---|---|
| Reporting API only | "Quelle est la probabilité de t6 ?" | One or more tool calls, no RAG |
| RAG only | "Que signifie t6 ?" | Vector retrieval (`scientific_knowledge`), no tool calls |
| API + RAG | "Pourquoi le modèle prédit t6 ?" | Both — live number + documented explanation of why that phase is hard/easy |
| Comparison / methodology | "Pourquoi avons-nous choisi le Semi-HMM ?" | RAG only (`project_knowledge`/`methodological_knowledge`), potentially multi-chunk synthesis, no live data |
| Multi-call / comparative-with-data | "Compare cette trajectoire à la trajectoire moyenne." | Multiple tool calls (this trajectory + some aggregate/reference computation) — **flagged as not fully supported by the current Reporting API**, see §Gaps below |

## 2. Tools

Each tool is a **typed wrapper around an existing Reporting API endpoint** (`docs/REPORTING_API.md`) — no tool introduces a new data path.

| Tool | Input | Output | Source endpoint | When to use |
|---|---|---|---|---|
| `get_trajectory(video_id, split)` | video_id, split (val/train only) | `TrajectorySummary` | `GET /videos/{id}/trajectory` | Question needs the full video's segment history / transition chain |
| `get_frame(video_id, split, window_start)` | as above + window | window metadata (timing, availability) | `GET /videos/{id}/windows` (filtered) | Question is about timing/timestamp for a specific window |
| `get_inference(video_id, split, window_start)` | as above | full `InferenceRecord` | `GET /videos/{id}/inference/{window}` | Question needs current phase, probabilities, next phase, or duration together — the common case |
| `get_phase_probabilities(video_id, split, window_start)` | as above | `phase_probabilities` + `entropy` only | same endpoint, narrowed response | Question is specifically about current-phase confidence, not the full record |
| `get_next_phase_distribution(video_id, split, window_start)` | as above | `next_phase_distribution` only | same endpoint, narrowed | Question is specifically "what's next" |
| `get_duration_distribution(video_id, split, window_start)` | as above | `duration` block only | same endpoint, narrowed | Question is about expected duration/timing |
| `get_transition_chain(video_id, split)` | video_id, split | `transition_chain` | `GET /videos/{id}/trajectory` | Question is retrospective ("how did this trajectory actually progress") — **never** for a live/in-progress framing |
| `search_knowledge(query, collections, filters)` | free text + optional collection/metadata filters | ranked chunks with metadata | RAG retrieval (`docs/RAG_ARCHITECTURE.md`) | Question needs definitions, methodology, or project history |
| `get_model_info()` | — | model name/version/config | `GET /model` | Question is "what model/version produced this" |
| `get_experiment_result(experiment_id)` | experiment name/id | metric summary | `project_knowledge` retrieval, filtered `experiment:<id>` (not a live endpoint — experiment results are static, RAG-appropriate, not Reporting-API-appropriate) | Question asks about a specific past experiment's numbers (e.g. "what was the PR-AUC for Semi-HMM") |

**Security**: every tool validates `split` and rejects `"test"` at the tool layer too — a third, redundant guard on top of the two already in the Reporting API itself (defense in depth, matching the discipline already used in `Training/reporting/`). No tool accepts a raw file path, raw SQL, or arbitrary code — every parameter is a bounded, typed value (video id validated against `list_videos()`, window against `get_windows()`, split against a fixed enum).

**Hard rule** (§27 of the request, restated as an implementation constraint): the LLM **must** call a tool for any numeric claim it is capable of retrieving, and **must not** answer from parametric memory or interpolation when a tool exists for that exact quantity. This is enforced by (a) system-prompt instruction, and (b) the grounding check described in §Reliability below — belt and braces, since a prompt instruction alone is not a reliable guarantee.

### Gaps this surfaces (not solved here, flagged for the roadmap)

- No tool/endpoint currently computes a **cross-trajectory aggregate** ("average behavior," "typical duration across videos") — the Reporting API has no such endpoint today. "Compare this trajectory to the average" cannot be honestly answered without either (a) a new Reporting API aggregate endpoint (real, scoped, non-trivial work — needs its own decision on what "average" means statistically) or (b) the LLM being explicitly told this isn't available and should say so, never approximate it from memory.

## 3. Reliability / guardrails

Every piece of information in an answer is tagged into one of five categories, and the tagging is enforced structurally, not just by prompt wording:

| Tag | Example | Source |
|---|---|---|
| **FACT** | `P(t6)=0.72` | Direct tool-call output, verbatim |
| **MODEL OUTPUT** | "current phase: t6" | Direct tool-call output, verbatim |
| **DOCUMENTED KNOWLEDGE** | "t6 corresponds to [...]" | RAG chunk, with citation |
| **MODEL INTERPRETATION** | "the model favors t6 over t5" | LLM synthesis *derived from* a FACT, must reference the underlying FACT |
| **LLM INFERENCE** | "this suggests the embryo is progressing normally" | LLM reasoning beyond what any tool/document states directly — must be explicitly flagged as such, never presented with the same confidence as a FACT |

**Structural enforcement, not just prompting**:
1. The orchestrator's context assembly (§ RAG_ARCHITECTURE.md §4) keeps FACT/DOCUMENTED blocks physically separate from the LLM's own generation — the system prompt instructs the model to only ever restate FACT block contents verbatim for numbers, never recompute or round differently.
2. **Post-hoc grounding check**: after the LLM produces an answer, a lightweight validation pass extracts every number/phase-name mentioned and checks it against the actual tool outputs from that turn. A number that doesn't match any tool output is flagged (logged, and — for a first version — the answer is regenerated or the ungrounded claim is stripped before returning to the user). This is the single concrete mechanism preventing "the LLM quietly changed 0.72 to roughly 0.7 and then to 'about 75%'" drift.
3. The LLM is never given raw numeric authority to "correct" a model output — if a user says "that seems wrong," the LLM's only valid moves are to re-fetch the tool output, cite the model's documented limitations (`docs/SCIENTIFIC_REPORT.md` §17), or say it cannot resolve the discrepancy — never to substitute its own estimate.

## 4. Citations

Two source families, distinguished explicitly in every answer:

- **Documentation-derived**: `[Source: docs/SCIENTIFIC_REPORT.md §13]` or inline "According to the emission diagnostic (`semi_hmm_next_step_analysis/emission_diagnostic`)..." — always resolvable to an actual file/section, never a vague "the documentation says."
- **Live-data-derived**: `[Live inference: semi_hmm_weekend_phaseF, Patient_319, window 54, 2026-08-24T13:10:00Z]` — always carries the model version, sample, window, and timestamp (§13 Observability, already implemented in `InferenceRecord.model`/`.window`/`.sample`).

Every answer containing a FACT or DOCUMENTED KNOWLEDGE claim must carry at least one citation of the matching family. An answer with zero citations is only acceptable for pure LLM INFERENCE, and must self-identify as such ("this is my own reasoning, not a documented result").

## 5. System evaluation scenarios

| Scenario | Data needed | Tools called | RAG sources | Expected answer shape | Hallucination risk |
|---|---|---|---|---|---|
| A. "Quelle est la phase actuelle ?" | Current window's inference | `get_inference` | none | Direct FACT, one sentence | Low — pure lookup |
| B. "Quelle sera probablement la prochaine phase ?" | Current window's next-phase dist. | `get_inference` or `get_next_phase_distribution` | none | FACT + argmax framing | Low |
| C. "Pourquoi ?" (follow-up to A/B) | Same window + emission difficulty for that phase | `get_inference` (already have) | `methodological_knowledge`, `project_knowledge` filtered by phase | FACT + DOCUMENTED explanation, clearly separated | **Medium** — the LLM must not invent a causal story beyond what §13's emission diagnostic actually documents |
| D. "Combien de temps avant la prochaine phase ?" | Current phase's duration distribution | `get_duration_distribution` | none, unless phase is tPB2/tEB → must surface `duration_available=false` | FACT with explicit uncertainty (quantile range, not a point estimate) | Medium — must not present a point number as certain, must not fabricate a duration when `duration_available=False` |
| E. "Explique-moi le Semi-HMM." | None live | none | `methodological_knowledge` | Synthesized DOCUMENTED KNOWLEDGE, citations | Medium — long-form synthesis risk of drifting from source |
| F. "Compare HMM et Semi-HMM." | None live (or optionally the current video's HMM vs Semi-HMM prediction, if in scope) | possibly `get_inference` twice (once conceptually per model) — **not supported by current Reporting API, which only serves Semi-HMM**; otherwise RAG only | `project_knowledge` (`docs/MODEL_COMPARISON.md`, `docs/GLOBAL_MODEL_COMPARISON.md`) | DOCUMENTED comparison, citing the real metric table | **High** if asked to compare live predictions (not supported) vs. **Low** if asked about the documented historical comparison |
| G. "Pourquoi le modèle est-il incertain ?" | Current window's entropy/phase_probabilities | `get_inference` | `project_knowledge` (the PR-AUC/calibration finding) | FACT (entropy value) + DOCUMENTED (known miscalibration, `PRAUC_REPORT.md`) | **Medium-high** — must not claim the probability itself is a trustworthy confidence measure, given the newly-discovered miscalibration; this scenario is a direct test of whether the guardrails in §3 actually work |

Scenario F exposes a real, current product gap: the Reporting API only ever serves the Semi-HMM (the `CURRENT_PIPELINE_CANDIDATE`) — it has no live HMM-vs-Semi-HMM comparison endpoint. A user asking to compare *live* predictions between architectures cannot be honestly answered today; the system must route this to the *documented* (historical, Val-set) comparison instead of fabricating a live one, and should say so explicitly.

## 6. Offline / privacy

| | A. External LLM API | B. Local LLM | C. Hybrid |
|---|---|---|---|
| Data sent externally | Tool outputs + RAG chunks + user question (never raw imagery/embeddings) | None | Structured facts/RAG text only, question stays local until routed |
| Confidentiality | Depends entirely on provider's data-handling terms; embryology data is health-adjacent — real regulatory exposure if raw data ever crosses this boundary (it shouldn't, given the architecture, but the *policy* must say so explicitly, not rely on the architecture alone) | Full data sovereignty | Depends on which calls stay local vs. external |
| Cost | Per-token, scales with usage | Infrastructure (GPU) cost, fixed regardless of usage | Mixed |
| Latency | Network-dependent, typically 1-10s | Depends on local hardware, can be faster or much slower | Mixed |
| Quality | Best available (frontier models) | Typically behind frontier models, though closing | Best-of-both if routed well (e.g. local for simple FACT framing, external for complex synthesis) |
| Complexity | Lowest to integrate | Requires hosting/serving infrastructure, model updates | Highest — two systems to maintain |

**Not resolved here** — this is explicitly a Decision Required (§ below), because it depends on factors this document cannot settle alone: institutional data policy for embryology data, budget, and whether the deployment target is a private research environment or something more exposed. The architecture already minimizes what *could* cross an external boundary (never raw imagery, never patient identifiers, only derived structured facts + documentation text) — but "minimizes" is not the same as "the decision has been made that this is acceptable," which is a human call.

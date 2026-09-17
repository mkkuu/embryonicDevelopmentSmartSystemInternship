# docs/ — map

Restructured on 2026-09-15; successor documents added under `handover/` on 2026-09-17. Read in
this order: the root `README.md`, then `handover/HANDOVER.md`, then the five guides, then
`reference/`, and only then `archive/`. Note that only part of this directory is in Git (see
"Git status of this directory" at the end): a fresh clone has the guides, `handover/`, the 30
RAG documents and `corpus/`, but **not** `reference/` or `archive/`.

## Guides (current, maintained)

| File | Content |
|---|---|
| `ARCHITECTURE.md` | what runs, layer by layer, verified in code; fragile couplings; legacy |
| `SCIENTIFIC_BACKGROUND.md` | question, data, models, established results with sources, limits |
| `REPRODUCIBILITY_GUIDE.md` | inputs and their identity, rebuild commands, reference numbers, GPU rules |
| `BENCHMARK.md` | Q1–Q15, protocol, score history, three-condition experiment, Validator, v1.3 status, artefact registry |
| `DEVELOPMENT.md` | machines, environment, tests, conventions, pitfalls |

## `handover/` — successor documents (2026-09-17, in Git, not in the RAG index)

| File | Content |
|---|---|
| `handover/HANDOVER.md` | first hour / day / week, checklists before touching code, experiments, benchmark, corpus, server; how to read the current state; validity of a run; recovery; source of truth |
| `handover/GPU_SERVER.md` | shared-server rules, read-only diagnostics, valid vs CPU-offload condition, the gated v1.3b launcher, post-run checks, server tree vs HEAD, restoring inputs |
| `handover/RAG_OPERATIONS.md` | corpus and INCLUDED/EXCLUDED semantics, embedding model, Chroma index and manifest, retrieval and query enrichment, provenance, why the server index is older than the versioned corpus |
| `handover/WEBAPP_GUIDE.md` | current BFF + frontend, endpoints and payloads, running it, security posture, the legacy `WebApplication/` |

They live in a subdirectory on purpose: `Training/rag/inventory.py` must classify every
top-level `docs/*.md` and is hashed into the benchmark run identity (see the last section).

## `reference/` — stable specifications and retained reports (not in the RAG index; local-only, not in Git)

`RAG_FIXED_QUESTION_BENCHMARK.md` (the frozen question specification),
`VALIDATOR_V1_IMPLEMENTATION.md`, `COMPACT_V2_AND_VALIDATOR_INTEGRATION.md`,
`GROUNDED_GENERATION.md`, `LLM_PROMPT_CONTRACT.md`, `TOOL_CONTRACTS.md`,
`WEBAPP_SECURITY.md`, `PHASE_7_FINAL_REPORT.md` (web app as built),
`RAG_LLM_QUALITY_REPORT.md` (28-question quality evaluation).

## The 30 RAG documents — frozen names, do not move, rename or edit

These files stay at the top of `docs/` because `Training/rag/inventory.py` lists them by path,
`Training/orchestrator/compact_context.py` names 17 of them in its curation rules, and the
vector index (`RagIndex/`) hashes their content. Changing any of them changes what the
language model reads. Several are historical (plans, roadmaps, implementation reports) and
would otherwise belong in `archive/`; they are indexed on purpose (retrieval-scope decision of
2026-08-25) and may only be re-classified together with a code change and a re-ingestion.

`HANDOFF.md`, `HANDOFF_SEMI_HMM.md`, `HANDOFF_EMISSION_BALANCED.md`,
`HANDOFF_EMISSION_WEIGHTED.md`, `RESEARCH_BLUEPRINT.md`, `BLUEPRINT_SUMMARY.md`,
`HMM_RESEARCH_PLAN.md`, `HMM_PHASE0_EXECUTION_PLAN.md`, `SCIENTIFIC_REPORT.md`,
`SCIENTIFIC_RESULTS.md`, `MODEL_COMPARISON.md`, `GLOBAL_MODEL_COMPARISON.md`,
`GRU_IDENTITY_ANALYSIS.md`, `GRU_IDENTITY_EXECUTIVE_SUMMARY.md`, `SEMI_HMM_PHASE_1_6_REPORT.md`,
`REPRODUCIBILITY.md` (2026-08-24 version; superseded in practice by `REPRODUCIBILITY_GUIDE.md`),
`DATA_LINEAGE.md`, `INFERENCE_SCHEMA.md`, `REPORTING_API.md`,
`REPORTING_API_IMPLEMENTATION_REPORT.md`, `REPORTING_CACHE.md`,
`REPORTING_CACHE_IMPLEMENTATION_REPORT.md`, `PRODUCT_ARCHITECTURE.md`, `PRODUCT_ROADMAP.md`,
`PROJECT_TO_PRODUCT.md`, `RAG_ARCHITECTURE.md`, `RAG_DATA_MODEL.md`, `LLM_ORCHESTRATION.md`,
`WEBAPP_ARCHITECTURE.md`, `WEBAPP_DATA_REQUIREMENTS.md`.

Authoritative among them: `HANDOFF.md` (mathematical formulation), `RESEARCH_BLUEPRINT.md`
(hypotheses), `SCIENTIFIC_REPORT.md` (results; supersedes `SCIENTIFIC_RESULTS.md`),
`HANDOFF_SEMI_HMM.md` (Semi-HMM), `DATA_LINEAGE.md`, `INFERENCE_SCHEMA.md`.

## `corpus/` — the Istanbul Consensus 2025 transcription (runtime dependency of the Validator)

`ISTANBUL_CONSENSUS_2025.md` (138 page-referenced blocks) and
`ISTANBUL_CONSENSUS_2025_TRACEABILITY.md`. Read directly by `Training/validator/corpus.py`;
not in the vector index. Never edit; its hash is recorded in every benchmark artefact.

## `archive/` — history, kept verbatim (local-only, not in Git)

`archive/sessions/` (day-by-day logs and checkpoints, including the 560 KB
`PROJECT_CHECKPOINT.md`), `archive/audits/` (the 2026-09-15 repository audits),
`archive/reports/` (finished experiment reports and design notes, Aug–Sep 2026). Index in
`archive/README.md`.

## Git status of this directory

Since 2026-09-16 (commits `621f458` and `5358604`), `docs/` is **partly versioned**. The
`.gitignore` rule is `docs/*` with three negations, `!docs/corpus/`, `!docs/*.md` and
`!docs/handover/` (the last one added 2026-09-17):

| Part | In Git? | Why |
|---|---|---|
| the five guides + `README.md` (top-level `*.md`) | **yes** | successor documentation |
| `handover/` | **yes** (`!docs/handover/`, 2026-09-17) | successor documentation, outside the `docs/*.md` glob of `rag/inventory.py` |
| the 30 RAG documents (top-level `*.md`) | **yes** | runtime dependency of `Training/rag/ingest.py` (`inventory.INCLUDED`) |
| `corpus/` | **yes** | runtime dependency of `Training/validator/corpus.py` |
| `reference/` | **no** — local-only | retained reports; no code path reads them |
| `archive/` | **no** — local-only | session logs, audits, historical reports; no code path reads them |

`reference/` and `archive/` exist only in the local checkout, in the 2026-09-15 snapshot
(kept by the project owner) and, as an older flat tree, on the GPU server.
This is deliberate: they are history, not inputs. Whether to version them later is an open,
non-blocking decision; nothing in the application depends on it.

Consequence of versioning the top-level `*.md`: adding or removing any `docs/*.md` now
requires classifying it in `Training/rag/inventory.py` (`INCLUDED` or `EXCLUDED`), or
`Tests/rag/test_rag_inventory.py` fails. `inventory.py` is one of the modules hashed by
`orchestrator/capture_run_identity.py`, so a new top-level document changes the recorded run
identity; put new documentation in `handover/` (or another subdirectory re-included in
`.gitignore`) instead, until the pending v1.3b comparison is done.

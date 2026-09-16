# ARCHITECTURE — what runs, how it is wired, what is legacy

Verified in the code on 2026-09-15 (static import graph of the 114 modules of `Training/`,
route tables, file paths). File references are given so you can check every statement.
Design documents with the reasoning behind these choices are listed in `docs/README.md`.

## 1. One page

```
 Browser  ──────────────  Training/webapp_api  (Flask, port 8001, the only network-facing surface)
                          │  GET /            index.html + app.js  (patient list · timeline · frame viewer · Q1–Q15 panel · answer)
                          │  GET /videos, /videos/<id>, /timeline, /window/<w>, /window/<w>/frame, /questions
                          │  POST /chat  {question | question_id, video_id, window, split=val}
                          │
                          ├── Training/reporting  (in-process; also a standalone read-only API on port 8000)
                          │     model_loader ─ Results/evaluation/semi_hmm_weekend_phaseF/model/   (frozen Semi-HMM, validated once)
                          │     trajectory_service ─ Embeddings/resnet18/{train,val}/            (test refused: 403)
                          │     inference_service + cache ─ Cache/reporting/…/inference.json     (regenerable)
                          │     transition_events ─ OBSERVED transitions from the annotations
                          │     frame_mapping ─ Data/embryo_dataset_F0 + _annotations             (real JPEG frames)
                          │
                          └── Training/orchestrator.answer_question()
                                router.classify()      keyword tables → DOCUMENTARY | DYNAMIC_DATA | HYBRID | UNKNOWN
                                tools.*                get_current_inference · get_trajectory · get_transition_events ·
                                                       get_inference_history · get_model_metadata · retrieve_documents · get_phase_information
                                   ├─ reporting.*      (above)
                                   └─ rag.retrieval ─ RagIndex/ (Chroma, collection docs_v1, 30 docs, multilingual MiniLM)
                                temporal_context       multi-window series, offsets, stability, model-vs-observed lag
                                context_builder        document / dynamic / scientific / audit blocks kept separate
                                compact_context + compact_v2_render   curated retrieval, labelled blocks (OBSERVED / MODEL-DERIVED / …)
                                scientific_reference ─ docs/corpus/ISTANBUL_CONSENSUS_2025.md   ([SCIENTIFIC] block, biology questions only)
                                prompt                 system prompt v2 (a reading method, never a value)
                                llm_provider           Ollama at http://localhost:11434  (llm_config: LLM_MODEL, default llama3.2:latest)
                                grounding_check        every number / phase / transition claim present in the context?  (presence, not truth)
                                scientific_validation ─ Training/validator  (claims → provenance → Istanbul corpus → rules R1…R14 → qualification)
```

Production import closure: 55 modules. Nothing in `Training/` imports `WebApplication/`, and
nothing in `WebApplication/` imports `Training/`.

## 2. Layers, bottom-up

### 2.1 Original classifier pipeline (`Training/*.py`, 2026-07, unchanged by the extension)
`preProcess.py` (splits, CSVs) → `DataSet.py` (`Embryo_Transition_Dataset`, sliding windows,
modes `image_seq` for ResNet18 / `video` for TimeSformer, target `consistency_flag`) →
`ModelBuilder.get_model()` → `train.py` / `train_balanced.py` → `Results/{model}/best_model.pth`
(a `state_dict`). Run from inside `Training/`. Known bugs: `Configs/config.ini` never read on
Linux; `ConfigArgs()` parses `sys.argv` at import (see `DEVELOPMENT.md` §5).

### 2.2 Embeddings (`Training/embeddings/`)
`build_cache.py` runs the frozen encoder once per (model, split) and writes
`Embeddings/resnet18/manifest.json` (model, checkpoint path and SHA-256, window, stride, focal
type, image size, extractor version) plus per split `embeddings.pt` (N × 512) and
`metadata.csv`. `validate.py` runs six integrity checks. `cache.py`/`dataset.py` are the
readers used by everything downstream.

### 2.3 Evaluation framework and scientific models (`Training/evaluation/`)
A `Model` interface (`fit`, `predict`, `save`, `load`; JSON, never pickle), a registry
(`@register_model`), `runner.run_experiment`, `metrics.py` (patient-level bootstrap,
Holm-Bonferroni), `calibration.py`. Seven registered models: `base_rate`, `identity_dynamics`,
`persistence`, `linear_ssm`, `gru` (the E1 ladder) and `hmm`, `semi_hmm` (the phase-as-hidden-state
branch). Only `hmm`/`semi_hmm` are loaded by the application; the others exist for the
scientific results (`SCIENTIFIC_BACKGROUND.md`). Since 2026-09-15 `evaluation/` holds only the
framework and the models; every runner lives under `Training/experiments/` (see its README). Caveat: `evaluation/__init__.py` imports
`runner`, which imports `plots` (matplotlib) — the web app therefore needs matplotlib installed.

### 2.4 Reporting (`Training/reporting/`)
Read-only serving of **one frozen model**: `model_loader.py` loads
`Results/evaluation/semi_hmm_weekend_phaseF/model/` once, checks `n_states = 15`,
`dmax = 268`, `duration_family = negative_binomial`, `embedding_dim = 512`, unweighted emission,
and raises `ModelIncompatibleError` otherwise — never a silent substitute. `inference_service`
computes filtering posteriors and next-phase distributions per video (cached on disk, atomic
writes, one lock per key); `trajectory_service` loads embeddings for `train`/`val` and refuses
`test`; `transition_events` derives the OBSERVED transitions from the annotations using the
HMM's own segmentation code; `frame_mapping` maps a window back to the raw frame files.
Records (`schemas.py`): sample, window (with `time_unit = "unknown/unverified"`), current
state (`current_phase`, `phase_probabilities`, entropy), next phase, duration (`duration_unit
= "windows"`), model info, ground truth, transition events (`observed = True`, `is_skip`,
`phase_distance`). Standalone API: `python -m reporting.api` on port 8000 (`/health`, `/model`,
`/videos`, `/videos/<id>`, `/windows`, `/inference/<w>`, `/inference_history/<w>`, `/trajectory`).

### 2.5 RAG (`Training/rag/`)
`ingest.py` parses the 30 documents listed in `inventory.py` (all under `docs/`), chunks them
(`chunker.py`: never across a heading, code block or table; ≈ 1 800 chars), embeds them with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, chosen so French
questions can hit an English corpus) and stores them in Chroma (`RagIndex/`, collection
`docs_v1`). A manifest of content hashes makes ingestion idempotent. `retrieval.py` is the only
consumer (top-k 5 by default). The Istanbul corpus is **not** in this index.

### 2.6 Orchestrator (`Training/orchestrator/`)
`answer_question(question, video_id, window, split="val", llm, llm_timeout_seconds,
context_format, validate)` returns `question, route, tool_plan, context_policy, context,
response, validation, final_answer`. There is no session or context resolution: the caller
must name the video and window. Steps:
1. `router.classify()` — deterministic keyword tables (accent-normalised), explicitly not a model.
2. A tool plan per route; `tools.py` wraps `reporting.*` and `rag.retrieval` with typed errors.
3. `temporal_context.derive_series_analysis()` — the multi-window series around the window,
   offsets from the centre, a `stability` block (returns, skips, regressions counted on the
   model's own sequence) and `model_vs_observed_lag_windows` (the only model-vs-annotation lag,
   `null` when the compared events differ).
4. `context_builder.build_context()` keeps `document_context`, `dynamic_context`,
   `scientific_context` and `retrieval_audit` physically separate, each with its provenance.
5. Context format (`prompt.py`, `EMBRYO_LLM_CONTEXT_FORMAT`, default `v2`; also `compact` and
   `compact_v2`): `compact_v2` renders labelled blocks — `[OBSERVED — …]`, `[MODEL-DERIVED — …]`,
   `[DERIVED — …]`, `[SCIENTIFIC — Istanbul Consensus 2025 …]`, `[DOCUMENTATION PROJET — …]`,
   `[LIMITATIONS — …]` — and curates retrieval with `SOURCE_RULES` D1/D2 and section rules
   (`compact_context.py`), which name 17 documents by path.
6. `scientific_reference.py` adds the `[SCIENTIFIC]` block only for `compact_v2` and only
   when the question carries a biology marker (never for timing/skip-only questions).
7. The system prompt (`prompt.py`) is a reading method: which block carries which quantity,
   three provenance classes never mixed, three never-interchangeable quantities (offset, time,
   lag), plus the prohibitions (no external knowledge, no invented value). A test pins that it
   contains no value, phase name or expected answer.
8. `llm_provider.py`: `NullLLMProvider` (default, never writes prose), `TemplateLLMProvider`
   (deterministic), `OllamaLLMProvider` (real; `num_ctx` from `EMBRYO_OLLAMA_NUM_CTX`, default
   16384; seed/temperature only if set). `llm_config.py` is the single decision point:
   `LLM_MODEL` (default `llama3.2:latest`), `LLM_TIMEOUT_SECONDS` (default 120 in the web app).
9. `grounding_check.py`: `grounded` (every number, phase token and transition pair of the
   answer is present in the context; locale-aware numerics; phase names only count for the top-2
   of a probability dict), `provenance_consistent` (a separate property), and per-claim
   `transition_claim_provenance` ∈ {observed, model, both, none}. Presence, never correctness.
10. `scientific_validation.py` (`EMBRYO_SCIENTIFIC_VALIDATOR`, default on) runs the
    Validator and returns `final_answer` = raw answer + a "QUALIFICATION SCIENTIFIQUE" block;
    the raw answer is never rewritten; a Validator error degrades to the raw answer.

### 2.7 Scientific Validator (`Training/validator/`)
`schema` (claim contract and provenance classes OBSERVED / MODEL-DERIVED / DERIVED-FROM-* /
SCIENTIFIC / …) → `extraction` (verbatim claim spans) → `provenance` (structural resolution of
each claim against the context blocks, including restated `key: value` fields) → `retrieval`
(lexical search in the 138 blocks of the Istanbul corpus, `corpus.py`) → `rules` (one function
per rule R1…R14, listed in `BENCHMARK.md` §6) → `validator` (verdicts SUPPORTED /
PARTIALLY_SUPPORTED / NOT_SUPPORTED / NOT_ASSESSABLE and the qualification text). Current
engine: `scientific_validator v1.3`.

### 2.8 Web application / BFF (`Training/webapp_api/`)
`app.py` (Flask, port 8001, `debug=False`) composes reporting and orchestrator in-process; the
frontend is one page (`templates/index.html`, `static/app.js`, `static/style.css`, self-hosted
Inter fonts, LabISEN logo) with no build tooling. Every backend string is rendered through
`textContent`/`createElement`, never `innerHTML` (pinned by `Tests/webapp_api/test_chat_dom_safety.mjs`
and a static-source audit). `/questions` serves the frozen Q1–Q15 from
`orchestrator.fixed_question_benchmark` (the app holds no copy); `/chat` accepts either
`question_id` or free `question` (both → 400). `split=test` → 403 here as well.

## 3. External services and data the running system needs

| Need | Where | Regenerable |
|---|---|---|
| Ollama service with the model named by `LLM_MODEL` | `http://localhost:11434`, GPU0 | system service, models never re-pulled by the project |
| Frozen Semi-HMM | `Results/evaluation/semi_hmm_weekend_phaseF/model/` | no (fit script: `Training/experiments/archive/semi_hmm_construction/phase_f_full_train_val.py`) |
| Embeddings (train, val) | `Embeddings/resnet18/` | yes, from `Results/resnet18_balanced/best_model.pth` |
| Raw images and annotations | `Data/embryo_dataset_F0`, `Data/embryo_dataset_annotations` | dataset download, NAS copy |
| Inference cache | `Cache/reporting/` | yes (`python -m reporting.warm_cache --split val`) |
| Vector index | `RagIndex/` | yes (`python -m rag.ingest`) from the 30 `docs/` files |
| Istanbul corpus | `docs/corpus/ISTANBUL_CONSENSUS_2025.md` | no (manual transcription of `Ressources/*.pdf`) |

Environment variables read at runtime: `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`,
`EMBRYO_OLLAMA_NUM_CTX`, `EMBRYO_LLM_CONTEXT_FORMAT`, `EMBRYO_SCIENTIFIC_VALIDATOR`,
`CUDA_VISIBLE_DEVICES`. `Configs/config.ini` is not read by the application.

## 4. Fragile couplings to know before refactoring

- `rag/inventory.py` (30 paths) and `orchestrator/compact_context.py` (17 paths) hard-wire
  document names under `docs/`: renaming or moving those files changes retrieval and the
  curated context, i.e. the answers. That is why they were left in place in `docs/`.
- `orchestrator/capture_run_identity.py` hashes a fixed list of modules by path: moving a
  module changes the run identity used to compare benchmark runs.
- `evaluation/__init__.py` eagerly imports `runner`, `plots` (matplotlib), `report`.
- Single importers in production: `rag.embeddings`/`vectorstore` ← `rag.retrieval`;
  `reporting.transition_events` ← `orchestrator.tools`; `validator.rules` ← `validator.validator`;
  `orchestrator.{context_builder, scientific_validation, tools}` ← `orchestrator.orchestrator`.
- `experiments/llm_regression/eval_questions.py` and `rag_llm_quality_questions.py` are used only by
  benchmark runners but pinned by active tests.

## 5. Legacy: `WebApplication/`

The original Flask + PostgreSQL application (doctor/admin accounts, embryo upload, a
prediction endpoint). It is **not part of the current product**: no import in either
direction, no test, its own `.env` (gitignored; template `.env.example`), `debug=True`, plain-text
password comparison, and `Classes/Doctor.py::predictTransitions` loads
`../Results/resnet18/best_model.pth` with `torch.load` expecting a whole model while the
pipeline saves a `state_dict` — and that path no longer exists — so it silently returns
random predictions flagged `is_random = true`. Launch: `cd WebApplication && flask run`
(port 5000). Kept until the PostgreSQL side is reviewed; see the root README §11.

## 6. Things that do not exist (so nobody looks for them)

No session/context memory in the chat; no multi-video aggregate tool; no GRU served; no
retraining or refitting path in the services; no external LLM API; no authentication on the
BFF; no automated grading of answers.

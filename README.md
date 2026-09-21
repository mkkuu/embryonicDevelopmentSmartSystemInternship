# Embryonic Development — SciML Extension

Detection of developmental-stage transitions in human embryo time-lapse videos, extended with
a scientific-machine-learning study of the temporal dynamics behind those transitions and a
local, grounded question-answering application over the resulting models.

This README is the entry point and the handover document of the repository. It explains the
project; `docs/handover/HANDOVER.md` explains how a successor takes ownership; the session log
`docs/archive/sessions/PROJECT_CHECKPOINT.md` (internal, not distributed — §5.2) records the state at each
session. Written in English so the repository can be transferred; the application's user
interface and the benchmark questions are in French.

**STATUS (2026-09-17): `WAITING_FOR_VALIDATOR_V1_3B_REBENCHMARK`** — see §11.

---

## 1. Project Overview

### 1.1 Purpose

Two things live in this repository:

1. **The original classifier** (`Training/*.py`, `WebApplication/`, July 2026): ResNet18 /
   TimeSformer models that detect, in an 8-frame window of a time-lapse video, whether a
   developmental transition occurs, plus a legacy Flask + PostgreSQL application for doctors
   and administrators (§4.8, §5). Paper reference in §15.
2. **The SciML extension** (`Training/{embeddings, evaluation, rag, orchestrator, validator,
   reporting, webapp_api}/`, `Training/experiments/`, `Tests/`, `docs/`, July–September 2026):
   it caches the classifier's embeddings, tests whether *explicit temporal-dynamics modelling*
   adds anything to frame-independent classification, builds a hidden-Markov / semi-Markov
   model of the developmental phase on top of the embeddings, serves that model through a
   read-only API, and lets a local language model answer questions about one embryo's
   trajectory with every number checked against the served data and every scientific claim
   qualified against one external reference (the Istanbul Consensus 2025).

### 1.2 Project origin and attribution

This repository did not start with the SciML extension. Two bodies of work coexist in it, and
the successor must not read the whole as created during the 2026 extension.

**Original project (2025)**

| Aspect | What the repository establishes | Source |
|---|---|---|
| Original author and originator of the codebase and of the research direction | **Aissa Benfettoume Souda**, first author of the reference paper, LSL team, LabISEN / ISEN Ouest | paper citation in the original README (`git show d268761:README.md`) and in §15 |
| Copyright holder of the original code | "Aissa BenFettoume Souda", 2025, MIT | `LICENSE` |
| Reference paper | *Spatio-Temporal Transformers for High-Accuracy Detection of Embryo Developmental Transitions* (2025), eight authors (§15) | original README (BibTeX), §15 |
| Initial code | `Training/*.py` (pre-processing, sliding-window dataset, ResNet18 / TimeSformer builder, training loop, configuration), `Configs/config.ini` | module headers `Author: LSL Team, Version 1.0, Last Updated: 2025-10-04` (13 of the 14 original modules; the 14th, `train_balanced.py`, was added by the extension) |
| Initial application | `WebApplication/` (Flask + PostgreSQL, doctor/admin accounts, embryo upload, prediction endpoint), `Dataset_shema.sql` | same headers; original README |
| Initial research direction | detection of developmental transitions in 8- or 32-frame windows of time-lapse videos, with `consistency_flag` as target; model selection by frame count (8 → ResNet18, 32 → TimeSformer) | original code and README |
| Original repository and contact | GitHub `AissaStory/Spatio-Temporal-Transformers-for-High-Accuracy-Detection-of-Embryo-Developmental-Transitions`; contact e-mail in the original README | original README |
| Author's published checkpoints | on the laboratory NAS; they solve a **per-frame** classification task (phase or embryo quality), not the windowed transition task, so the extension had to train its own classifier | `docs/archive/sessions/POINT_STAGE_2026-07-23.md` |

The original README (recoverable with `git show d268761:README.md`) documents the training CLI,
`config.ini`, the PostgreSQL set-up and the legacy application in full; it remains the
reference for that part of the project.

**SciML extension (July–September 2026)**

| Aspect | Fact |
|---|---|
| Who | the current repository owner (Git author `mkkuu`, every commit), during an internship, on the laboratory's GPU server, with an AI coding agent driven by written missions (`docs/DEVELOPMENT.md` §7); every session is logged in `docs/archive/sessions/` (internal, §5.2) |
| Added | `Training/{embeddings, evaluation, rag, orchestrator, validator, reporting, webapp_api, experiments}/`, `Training/train_balanced.py`, `Tests/`, `docs/` (except the original README), `docs/corpus/`, the Istanbul PDF under `Ressources/` |
| Kept and used from the original code | `preProcess.py`, `DataSet.py`, `ModelBuilder.py`, `Load_data.py`, `config_args.py`, `train.py`, `train_val_test_pipline.py`: the extension imports them (composition) and trained its own ResNet18 with them |
| Not modified | none of the original modules was edited by the extension (§3.1 rule "composition, never edition"); `WebApplication/` is untouched and now legacy (§4.8); the original `README.md` was replaced by the successor README on 2026-09-15 and is kept in Git history |
| Dataset | the Human embryo time-lapse video dataset (Zenodo, DOI 10.5281/zenodo.7912264; creators and licence in §15), used unchanged by both parts |

Known divergence, not resolved here: `LICENSE` spells the original author's name
"BenFettoume" while the paper citation, the original README and the session notes spell it
"Benfettoume"; the original README's BibTeX key says 2024 while its `year` field says 2025.
The repository does not record who else in the LSL team worked on the original code beyond
the paper's author list.

### 1.3 Scientific question

Verbatim from `docs/RESEARCH_BLUEPRINT.md` (Part I):

> Is human embryo development, as captured by time-lapse imaging, governed by a shared,
> low-dimensional dynamical law — and do the discrete clinical categories (phase, arrest)
> correspond to that law's structure, or to something else?

### 1.4 Project scope

- **In scope, done**: embedding cache; the E1 ablation ladder (H1) with a frame-shuffle
  control; a GRU branch; an HMM and a Semi-HMM of the phase; a read-only Reporting API with a
  disk cache; a RAG index over the project's own documents; a deterministic orchestrator with
  a local LLM (Ollama), a grounding check and a Scientific Validator; a chat web application;
  a frozen 15-question benchmark with several paired experiments.
- **In scope, pending**: the Validator v1.3 rebenchmark (§11).
- **Out of scope / never started**: hypotheses H2–H4 (§2.4); retraining or refitting any
  model from the application; an external LLM API; authentication on the new web app; any
  clinical use.

### 1.5 Current status

| Item | Status on 2026-09-17 |
|---|---|
| Repository | public release prepared on 2026-09-21 from the internal repository; the state described in this README is that of the 2026-09-17 documentation pass |
| Scientific models | frozen; the served Semi-HMM has not changed since 2026-08-23 |
| Benchmark | Q1–Q15 frozen since 2026-09-01; last valid run = Validator v1.2 OFF/ON (2026-09-12) |
| Validator | v1.3 implemented, frozen (hashes in the gated launcher), tests pass; **not yet rebenchmarked** |
| Blocking condition | GPU0 of the shared server is occupied by an external job; the gated launcher waits |
| GPU server code | **older layout than this repository** (not synced since the 2026-09-15 restructure; §5.3) |

### 1.6 What has been demonstrated

Every statement below is measured on this dataset, this encoder checkpoint and this target;
sources and confidence intervals are in `docs/SCIENTIFIC_BACKGROUND.md` and `docs/BENCHMARK.md`.

| Statement | Evidence |
|---|---|
| The embedding content alone carries most of the transition signal (`identity_dynamics`, Test AUROC 0.7384 [0.7189, 0.7563]) | E1 ladder, patient-level bootstrap n = 1000 |
| Fixed or learned **linear** dynamics do not add to it (persistence 0.5922, linear SSM 0.5858, both below identity) | E1 ladder + frame-shuffle control |
| **Non-linear** dynamics (GRU) do not add to it once the frame-shuffle confound is accounted for | GRU steps 6a/6b/6c |
| A phase-as-hidden-state model (HMM k = 7) reaches Val AUROC 0.63735 with poor calibration | HMM sweep |
| Explicit phase-duration modelling (Semi-HMM) gives a small, directionally consistent, non-significant gain (Val AUROC 0.64271, +0.0054; p = 0.29 at trajectory level, n = 106) | Semi-HMM phase F |
| Per-phase calibration error correlates with phase duration (r ≈ −0.7, 13 phases) | duration diagnostic |
| Emission class re-weighting does not help HMM or Semi-HMM | emission branch (closed) |
| The LLM, fed with model outputs, presents them as observations; a presence-only grounding check cannot see it | three-condition experiment (§10) |
| Validator v1.2 produced 5 false positives on non-hallucinated answers; v1.3 removes them on a read-only replay of the 45 stored answers without losing a correct detection | v1.3 replay + frozen test cases |

### 1.7 What has not been demonstrated

- That a shared low-dimensional dynamical law exists (H1 not supported; H2–H4 never tested).
- That the Semi-HMM is significantly better than the HMM at trajectory level.
- That any HMM/Semi-HMM configuration is calibrated: none beats the constant-rate Brier
  baseline, and the 15-way phase posteriors are worse than uniform on Brier.
- That the Scientific Validator improves the benchmark score (v1.2: OFF 18.5/45 vs ON 19.0/45,
  n = 3, one grader, no blinding; v1.3 not yet run on a valid GPU).
- That the compact context format is better than v2 (+1.17/15, not significant, Q3 regressed).
- That the language model is reliable: run-to-run σ = 0.75/15 with byte-identical contexts.
- Anything about biology beyond the dataset's annotations: the time unit is unverified in every
  artefact, and the corpus contains no morphokinetic reference times.

---

## Scientific Integrity — DO NOT VIOLATE

These rules are load-bearing; several of them are enforced in code (403 on the test split,
Validator rules R1–R14, tests that pin the prompt and the questions). Breaking one silently
invalidates every comparison made so far.

- **Never modify dataset annotations.** The dataset, the annotations and the splits are locked.
- **Never tune on the test split.** It was consumed twice (E1 ladder, GRU) for pre-registered
  reports and is refused with HTTP 403 at every serving layer. Benchmark development uses `val`.
- **Never present an annotated skip as a predicted skip**, nor the reverse (Validator R3).
- **Never confuse model-derived information with observations** (R2, the reason the Validator
  exists). Every context block is labelled OBSERVED / MODEL-DERIVED / DERIVED / SCIENTIFIC /
  LIMITATION; keep the labels.
- **Never claim a temporal lag when the compared transition events are not the same event.**
  `model_vs_observed_lag_windows = null` plus an explanation is the correct output (R11).
- **Never infer direct cleavage solely from a phase skip**, reverse cleavage solely from a
  return in the predicted phase, or chaotic cleavage from phase labels alone. These terms are
  neither annotated nor defined in the corpus (Q15 stays NOT_ASSESSABLE).
- **Do not infer biological atypicality solely from model predictions** (R14: the Consensus
  never validates a prediction).
- **Never invent a morphokinetic fact.** If the corpus lacks it, the correct answer says so.
- **If the data cannot support a biological conclusion, the verdict is NOT_ASSESSABLE.**
- **Never name a time unit where it is not sourced.** Every artefact says
  `time_unit = unknown/unverified`; durations are in windows (R9/R9c).
- **Do not silently change benchmark conditions** (questions, anchor, model, `num_ctx`,
  timeout, context format, corpus, prompt). Record a run identity for every run.
- **Do not treat stochastic LLM variation as a regression without replication** (n ≥ 3,
  paired designs; differences below ≈ 1.5/15 are not interpretable).
- **Preserve provenance of all benchmark artefacts.** Never delete or overwrite one; an invalid
  run is kept and marked with a sidecar `.INVALID.txt`.
- **Never run a benchmark while a foreign process occupies GPU0**, and never touch another
  user's job (§3.4).

---

## 2. Scientific Context

### 2.1 Problem formulation

Time-lapse image sequences of human embryos are annotated frame by frame with 15
chronological developmental phases (`tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM,
tSB, tB, tEB`). The original classifier looks at an 8-frame window and predicts
`consistency_flag`: 0 if the first and last frame carry the same phase, 1 if a transition
occurs inside the window (positive rate ≈ 13–14 %). This binary event is the supervision
target of the classifier and of every model in the extension.

### 2.2 Dynamical-system perspective

`docs/HANDOFF.md` §4 (a design document written before any experiment) formulates the embryo
as a hidden state `x(t)` with static individual parameters `θ`, a discrete regime `r(t)` with a
sojourn time `τ(t)`, dynamics `dx = f_r(x;θ)dt + σ(x;θ)dW + jumps`, regime switching through a
generator `Q(x,τ;θ)`, and observations `y_k = h(x(t_k), r(t_k)) + ε_k`. The simplification
chain (hybrid jump-diffusion → SDE → ODE → linear state space → *or* a semi-Markov model on
`r(t)` alone) is what the experiments walk down: where one stops in this chain is the actual
hypothesis under test.

### 2.3 Partial observation

The observation map `h` is generically non-injective, so some components of the state are
structurally unobservable. In practice the observations of every downstream model are the
512-dimensional ResNet18 embeddings of overlapping 8-frame windows (stride 1, 7 of 8 frames
shared between neighbours), which violates the emission-independence assumption of the
HMM/Semi-HMM. This is documented, not corrected; it affects sequence log-likelihoods, not the
k-step event that is actually scored. The first and last phases (`tPB2`, `tEB`) are
structurally censored under this recording protocol.

### 2.4 Scientific hypotheses

| Hypothesis | Claim | Status |
|---|---|---|
| **H1** (gates everything) | explicit dynamics modelling beats frame/window-independent classification on temporal-coherence tasks | **not supported**: no linear or non-linear dynamics model beats the order-invariant `identity_dynamics` once the frame-shuffle control is applied |
| H2 | a discrete regime variable is necessary beyond a continuous state | never tested as pre-registered; the HMM/Semi-HMM branch stops at "the phase as the regime" |
| H3 | biologically derived constraints beat a placebo/scrambled control | never tested |
| H4 | natural coordinates differ from hand-designed biological ones | deferred, never started |

### 2.5 Model-derived vs observed information

Five kinds of statement are kept apart in code, context and documentation:

| Label | Meaning | Example |
|---|---|---|
| OBSERVED | an annotation of the dataset | "the annotated phase at window 156 is t4" |
| MODEL-DERIVED | a prediction or estimate of the Semi-HMM | "the model's most likely phase is t7" |
| DERIVED | a deterministic computation on observations or predictions | "last annotated transition before this window: t4 → t6" |
| SCIENTIFIC | external knowledge (Istanbul Consensus 2025) | phase order, morphology vocabulary |
| LIMITATION | a field of the context that bounds the answer | `time_unit = unknown/unverified` |

A model output presented as an observation is the central failure mode this project measures
and guards against.

---

## 3. Experimental Philosophy

### 3.1 Scientific integrity
See the banner above. Comparisons are pre-registered where possible (E1 ladder, Holm-Bonferroni
over three tests), negative results are kept and reported, superseded documents are marked
rather than deleted, and every benchmark artefact records the conditions it was produced under.

### 3.2 Dataset / split policy
Split by video, seed 42, 70/15/15: Train 492 videos / 196 934 windows, Val 106 / 43 339,
Test 106 / 42 519. `test` is locked (used twice, pre-registered, never again for exploration);
`val` is the development and benchmark split. Annotations are never edited.

### 3.3 Reproducibility
Every input has an identity (checkpoint SHA-256, embedding manifest, corpus hashes, index
manifest, run identity JSON). The rebuild order and the reference numbers to reproduce are in
`docs/REPRODUCIBILITY_GUIDE.md`; the generic workflow is in §13. LLM outputs are not
reproducible run to run; contexts, grounding verdicts and Validator verdicts are.

### 3.4 GPU constraint
The project uses **at most one GPU, GPU0**, of a shared laboratory server. GPU1–GPU7 belong to
other users. No external job is ever interrupted, signalled, reniced or inspected beyond
read-only commands. A benchmark is valid only if the LLM sits entirely in GPU0 memory
(`gpu_fraction = 1.0`). The full rules and commands are internal GPU-server procedures, handed
over separately to the successor (not part of this repository).

### 3.5 Frozen benchmark policy
The 15 questions, their anchor (`Patient_319`, window 156, split `val`), their success criteria
and the grading conventions are frozen (`Training/orchestrator/fixed_question_benchmark.py`,
`docs/reference/RAG_FIXED_QUESTION_BENCHMARK.md`). No experimental change is adopted without a
comparison on this benchmark, with replication.

---

## 4. System Architecture

### 4.1 High-level architecture

```
Browser (French UI)
  → Training/webapp_api        Flask BFF, port 8001 — the only network-facing surface
      ├─ Training/reporting    read-only inference on the frozen Semi-HMM (in-process; also standalone on :8000)
      └─ Training/orchestrator.answer_question()
            router.classify()  keyword tables → DOCUMENTARY | DYNAMIC_DATA | HYBRID | UNKNOWN
            tools.*            typed calls over reporting.* and rag.retrieval (Chroma, 30 project documents)
            temporal_context   multi-window series, offsets, stability, model-vs-observed lag
            context_builder    document / dynamic / scientific / audit blocks, kept separate, labelled
            prompt (v2)        a reading method — never a value, phase name or expected answer
            llm_provider       Ollama, http://localhost:11434 (model from llm_config.LLM_MODEL)
            grounding_check    is every number / phase / transition pair PRESENT in the context? (presence, not truth)
            scientific_validation → Training/validator   claims → provenance → Istanbul corpus → rules R1…R14 → qualification
```

Verified in code on 2026-09-15 (`docs/ARCHITECTURE.md`, with file references for every box).

### 4.2 Scientific pipeline

```
Data/ (images + phase annotations)
  → Training/preProcess.py            video-level splits 492/106/106 → Data/Splits/F0.csv
  → Training/DataSet.py               8-frame windows, stride 1, target consistency_flag
  → Training/train_balanced.py        ResNet18, class-balanced → Results/resnet18_balanced/best_model.pth
  → Training/embeddings.build_cache   512-d embeddings per split → Embeddings/resnet18/{train,val,test}/
  → Training/experiments/*            E1 ladder, GRU, HMM k=7 sweep, Semi-HMM → Results/evaluation/*
  → Training/reporting                frozen Semi-HMM: filtering posteriors, next-phase distributions, observed transitions, frames
```

### 4.3 Temporal context
`orchestrator/temporal_context.py` builds, around the requested window, a multi-window series
of the model's phase posteriors with offsets from the centre, a `stability` block (returns,
skips, regressions counted on the model's own sequence) and `model_vs_observed_lag_windows`,
the only model-vs-annotation lag, which is `null` whenever the compared events differ. Three
quantities are never interchangeable: offset (position in the series), time (unverified unit),
lag (model vs annotation of the *same* event).

### 4.4 RAG
`Training/rag/` ingests the 30 project documents listed in `rag/inventory.py` into a Chroma
collection (`RagIndex/`, `docs_v1`) with a multilingual sentence embedding, and `retrieval.py`
returns the top 5 chunks. Dynamic data (predictions, per-window records) is structurally never
indexed. Details and operating rules: `docs/handover/RAG_OPERATIONS.md`.

### 4.5 LLM
One decision point, `Training/orchestrator/llm_config.py`: `LLM_MODEL` (default
`llama3.2:latest`, the production default since the web app was built) and
`LLM_TIMEOUT_SECONDS` (default 120 in the web app). The benchmark model is `mistral-nemo:12b`
(§9). Three providers exist: `NullLLMProvider` (default in code, never writes prose),
`TemplateLLMProvider` (deterministic), `OllamaLLMProvider` (real). No external API is called.

### 4.6 Grounding
`orchestrator/grounding_check.py` checks that every number, phase token and transition pair of
the answer is present in the context (locale-aware numerics; phase names reachable only as
keys of a probability dict count for the top 2). It reports presence, never correctness or
provenance; its rate is not a quality score.

### 4.7 Validator
`Training/validator/` extracts verbatim claims, resolves each claim's provenance against the
context blocks, searches the Istanbul corpus lexically, applies rules R1…R14 and returns
verdicts SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / NOT_ASSESSABLE. The raw answer is
never rewritten; a "QUALIFICATION SCIENTIFIQUE" block is appended. It is a qualification and
grounding layer, **not a biological oracle** and not a scientific model. Engine: v1.3.

### 4.8 Web application
`Training/webapp_api/` is the current application (patient list, timeline with phase bands,
frame viewer, the frozen Q1–Q15 panel, free-text chat). `WebApplication/` is the original,
legacy application, not on the current path. Both: `docs/handover/WEBAPP_GUIDE.md`.

---

## 5. Repository Structure

### 5.1 Tracked (in Git)

| Path | Role | Kind |
|---|---|---|
| `Training/*.py` | original pipeline: `preProcess.py`, `DataSet.py`, `ModelBuilder.py`, `Load_data.py`, `config_args.py`, `train.py`, `train_balanced.py`, `train_val_test_pipline.py` | active code (legacy pipeline, unchanged by the extension) |
| `Training/embeddings/` | embedding cache builder, readers, validator | active code |
| `Training/evaluation/` | `Model` interface, registry, bootstrap metrics, calibration, the seven models (`models/`) | active code (only `hmm`/`semi_hmm` are loaded by the application) |
| `Training/rag/` | inventory, chunker, embeddings, Chroma store, ingest, retrieval | active code |
| `Training/orchestrator/` | router, tools, temporal context, context builder, prompt, providers, grounding, the frozen questions, the benchmark runners, run identity | active code |
| `Training/validator/` | Scientific Validator v1.3 (schema, extraction, provenance, corpus, retrieval, rules, validator) | active code (frozen) |
| `Training/reporting/` | frozen-model loader, inference/trajectory services, cache, transition events, frame mapping, standalone API | active code |
| `Training/webapp_api/` | BFF Flask app, `templates/index.html`, `static/{app.js, style.css, fonts, logo}` | active code |
| `Training/experiments/` | every experiment and benchmark runner, grouped by experiment (`e1_ladder/`, `gru/`, `hmm_semi_hmm/`, `llm_regression/`, `context_ablations/`, `rag_llm_diagnostics/`, `embeddings_exploration/`, `rag_retrieval_smoke/`) with an index README; `archive/` = 37 one-off scripts salvaged from the server's `/tmp` | experiments (nothing here is imported by the application) |
| `Tests/` | 68 `test_*.py` files + 4 Node files, synthetic data / mocked dependencies only (last full run 2026-09-15: 1 218 passed, 40 skipped, 5 environment-only failures on the local machine; all pass on the server) | tests |
| `docs/*.md` (top level) | the 6 guides + `README.md` map, and the **30 RAG documents** (frozen names) | documentation, of which 30 are a runtime input |
| `docs/corpus/` | Istanbul Consensus 2025 transcription + traceability | runtime input of the Validator |
| `docs/handover/` | successor documents (HANDOVER, RAG_OPERATIONS, WEBAPP_GUIDE); the GPU-server procedures are internal and handed over separately | documentation |
| `Ressources/` | `ISEN.jpg`, `LabISEN.png`, the Istanbul Consensus 2025 PDF (source of the corpus, referenced by `rag/validate_istanbul_corpus.py`) | resources |
| `WebApplication/` | legacy Flask + PostgreSQL app, `.env.example`, `Dataset_shema.sql` | legacy code |
| `Configs/config.ini` | training configuration of the original pipeline; **not read on Linux** (§5.5) | kept for the record |
| `requirements.txt`, `LICENSE`, `.gitignore` | infrastructure | — |

### 5.2 Internal, not distributed (in `.gitignore`; kept on the project owner's checkout and in the 2026-09-15 snapshot)

| Path | Content | Where it exists |
|---|---|---|
| `docs/archive/` | `sessions/` (the 560 KB `PROJECT_CHECKPOINT.md` day-by-day log and earlier logs), `audits/` (2026-09-15 audits), `reports/` (24 finished experiment reports) | original checkout, snapshot; an older flat copy of some files on the server |
| `docs/reference/` | 9 stable specifications and retained reports (frozen question spec, Validator v1 implementation, prompt contract, tool contracts, web-app security, …) | same |
| `CLAUDE.md`, `.claude/` | operating guide of the AI coding agent that built the extension | original checkout |
| `graphify-out/` | regenerable code knowledge graph | regenerated by hooks |

A fresh clone therefore has the code, the tests, the guides, the RAG corpus and the Istanbul
corpus, but **not the session history**. `docs/handover/HANDOVER.md` says where to get it.
Every `docs/archive/…` or `docs/reference/…` path cited in this README and in the guides is a
provenance pointer to these internal documents; it does not resolve in a public clone.

### 5.3 Server-only (gitignored, never in Git, exist only on the GPU server + NAS backups)

`Data/` (dataset, splits), `Results/` (checkpoints, the frozen HMM/Semi-HMM, every experiment
and benchmark artefact), `Embeddings/`, `RagIndex/` (vector index, regenerable), `Cache/`
(inference cache, regenerable). Paths, identities and backups: `docs/REPRODUCIBILITY_GUIDE.md` §2.

**The server's code tree is not this repository's HEAD.** The server is fed by `rsync` only
(its `.git` is stuck at the Init commit) and was last synced before the 2026-09-15 restructure:
it still has the runners inside the packages (no `Training/experiments/`), the pre-2026-09-16
`rag/inventory.py`, and a flat `docs/` of 39 files (30 RAG documents + 9 old session files, no
`archive/`, `reference/` or `handover/`). This is deliberate while the v1.3 rebenchmark is
pending: the gated launcher only needs modules that were not moved. Read
the internal GPU-server procedures before syncing anything.

### 5.4 Untracked local files on the project owner's checkout (not in Git, not needed by anything)

`.gitattributes` (a `merge=graphify` driver line for `graphify-out/graph.json`),
`.graphifyignore` (local tooling), `new-logo-isen.png` and `Ressources/logo-LabISEN_2024.png`
(both byte-identical to the tracked `Training/webapp_api/static/logo-LabISEN_2024.png`),
`Ressources/logo-Isen_2024.jpg` (referenced nowhere). None is required to operate the project;
deleting or versioning them is an open, non-blocking decision.

### 5.5 Known non-blocking technical debt (confirmed by inspection)

- `Configs/config.ini` is never read on Linux (`ConfigArgs` default path uses a backslash);
  `ConfigArgs()` also parses `sys.argv` at import time — see `docs/DEVELOPMENT.md` §5.
- `Training/evaluation/models/README.md` is stale (only the original linear ladder) and says so.
- `Training/orchestrator/llm_config.py` and a few frozen modules cite old documentation paths
  (`docs/RAG_FIXED_QUESTION_BENCHMARK.md`, `docs/PROJECT_CHECKPOINT.md`) in docstrings; the
  files now live under `docs/reference/` and `docs/archive/sessions/`. Left unchanged because
  those modules are hashed into the benchmark run identity.
- `evaluation/__init__.py` eagerly imports `plots` (matplotlib), so the web app needs matplotlib.
- Lost and not recoverable: the original `Results/resnet18/` checkpoint (destroyed 2026-07-24)
  and the script that produced `Results/evaluation/gru_identity_threshold_analysis/` (never
  found; the report is kept). The whole data side lives on one server, with NAS copies.

---

## 6. Data

| Aspect | Fact | Provenance class |
|---|---|---|
| Dataset | public human-embryo time-lapse dataset (Zenodo, §15), focal plane F0, grayscale frames | — |
| Patients / videos | one video per embryo, identified as `Patient_<n>`; 704 videos after splitting | — |
| Splits | by video, seed 42: Train 492 / Val 106 / Test 106 (`Data/Splits/F0.csv`) | — |
| Annotations | one phase per frame among 15 chronological phases (`tHB` excluded); ≈ 11.5 % of annotated transitions skip a phase | **OBSERVED / ANNOTATION** |
| Windows | 8 frames, stride 1; windows spanning > 2 phases or 2 non-adjacent phases are dropped | DERIVED |
| Target | `consistency_flag` (transition inside the window) | DERIVED from annotations |
| Time representation | window indices. `time_unit = "unknown/unverified"` and `duration_unit = "windows"` in every record. The raw elapsed-time files exist in the dataset but were never joined into the pipeline; the owner has stated outside the repository that the column is in hours, which is **not** traced in any artefact | LIMITATION |
| Observed transitions | derived from the annotations by the HMM's own segmentation code (`reporting/transition_events.py`): `observed = True`, `is_skip`, `phase_distance` | DERIVED (from OBSERVED) |
| Model predictions | Semi-HMM filtering posterior over 15 phases (`current_phase`, `phase_probabilities`, entropy), next-phase distribution, model transition sequence, stability counts | **MODEL-DERIVED** |
| Embeddings | 512-d ResNet18 features per window (`Embeddings/resnet18/`), the observations of every downstream model | MODEL-DERIVED (encoder) |

The `phase_probabilities` served by the API are a ranking signal, not calibrated probabilities.

---

## 7. Models and Scientific Components

| Component | Status | Where | Notes |
|---|---|---|---|
| ResNet18 classifier (class-balanced) | **production encoder**; sole surviving checkpoint (SHA-256 `be5460d0…`) | `Results/resnet18_balanced/best_model.pth`, `Training/train_balanced.py` | the original `Results/resnet18/` was destroyed on 2026-07-24 |
| TimeSformer | original pipeline option, no checkpoint on the server | `Training/ModelBuilder.py` | never used by the extension |
| E1 ladder: `base_rate`, `identity_dynamics`, `persistence`, `linear_ssm` | **benchmark models, negative result for dynamics**; not served | `Training/evaluation/models/`, `Results/evaluation/e1_*` | Test split, pre-registered |
| GRU (`GRUDynamicsModel`, residual and direct modes) | **exploratory, negative result**; not served; role as a future "signal" documented but never implemented | `evaluation/models/gru.py`, `Results/evaluation/e1_gru_*` | its apparent +0.01 AUROC gain disappears under frame shuffling |
| HMM k = 7 (α 2.0, ρ 0.5) | reference for the Semi-HMM comparison; its emission model is reused | `Results/evaluation/e1_hmm_k7_sweep/best_model/` | transitions relaxed to allow skips |
| **Semi-HMM** (dmax 268, negative binomial, unweighted emission) | **production / reference model, served by `reporting`**; frozen since 2026-08-23; never refit | `Results/evaluation/semi_hmm_weekend_phaseF/model/`; fit script `Training/experiments/archive/semi_hmm_construction/phase_f_full_train_val.py` | `model_loader.py` validates the configuration and refuses anything else |
| Emission re-weighting variants | closed branch, negative | `Results/evaluation/emission_*` | — |
| Dynamics-related components (linear SSM forecast/imputation, k-step alignment check, duration diagnostics) | diagnostics, finished | `Training/experiments/{e1_ladder, hmm_semi_hmm}`, `experiments/archive/hmm_diagnostics/` | — |
| SciML directions H2–H4 | **never started** | `docs/RESEARCH_BLUEPRINT.md` | — |
| LLMs | `llama3.2:latest` production default; `mistral-nemo:12b` benchmark model | Ollama on the server, `orchestrator/llm_config.py` | not scientific models |

Do not rank the E1/GRU block and the HMM/Semi-HMM block on one AUROC scale: they are different
task formulations scoring the same label.

---

## 8. LLM / RAG System

Chain: **Web App → Flask BFF → Orchestrator → Router → Tools / RAG → Context Builder → Ollama
LLM → Grounding → Validator** (§4.1).

| Element | What it is today |
|---|---|
| Ollama | system service on the GPU server (`http://localhost:11434`), pinned to GPU0; models are never pulled or re-pulled by the project |
| Model choice / history | `llama3.2:latest` scored 1.0/15 on the frozen benchmark (10 refusals); `llama3.1:8b` 4.0/15; `mistral-small3.1:24b` was A/B-tested earlier (`docs/archive/reports/RAG_LLM_MODEL_AB_TEST.md`); `mistral-nemo:12b` (Q4_0, ≈ 11.7 GB at `num_ctx = 16384`, fits GPU0's 12 GB alone) scored 5.5/15 in the single-variable swap and is the benchmark model since 2026-09-02. Production default was left at `llama3.2:latest` pending a decision |
| Retrieval | `rag/retrieval.py`, top-k 5, no re-ranking; `orchestrator.build_retrieval_query()` appends the phase names already present in the fetched tool outputs for two questions (Q11/Q12) and invents nothing |
| Chroma | persistent embedded store in `RagIndex/`, collection `docs_v1`; only `rag/vectorstore.py` talks to it |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims; chosen after an English-only model failed on a French query) |
| Corpus | the 30 `docs/*.md` in `rag/inventory.py` `INCLUDED` (hand-curated, each with a reason); 6 `EXCLUDED` guides. The Istanbul corpus is **not** in the index |
| Context construction | `context_builder.build_context()` keeps `document_context`, `dynamic_context`, `scientific_context`, `retrieval_audit` separate; format `compact_v2` (used by the Validator experiments) renders labelled blocks and curates retrieval with source/section rules that name 17 documents by path |
| Temporal context | §4.3 |
| Grounding | §4.6 |
| Validator | §4.7; rules listed in `docs/BENCHMARK.md` §6 |
| Istanbul Consensus 2025 | `docs/corpus/ISTANBUL_CONSENSUS_2025.md`, 138 page-referenced blocks transcribed from `Ressources/Istanbul Consensus 2025 (2).pdf`; read directly by the Validator and injected as a `[SCIENTIFIC]` block for biology questions only. The consensus states it is for information and education and sets no standard of care; the project uses it for vocabulary and phase order, never to validate a prediction |

Operating rules for the corpus and the index: `docs/handover/RAG_OPERATIONS.md`.

---

## 9. Benchmark Protocol

- **Why**: to measure the *answer layer* (not the scientific models): does the LLM, fed with
  the Reporting API's outputs and the retrieved documents, answer correctly, admit gaps, and
  never turn a prediction into an observation?
- **Frozen**: 15 French questions Q1–Q15 (`Training/orchestrator/fixed_question_benchmark.py`),
  anchor `Patient_319`, split `val`, window 156 (an annotated `t4 → t6` skip where the model
  believes `t7`), success criteria, expected sources, grading conventions (since 2026-09-09:
  Q7 "non calculable" = PASS; a Q9 answer covering only the model side = PARTIAL).
- **Conditions**: `mistral-nemo:12b`, `num_ctx = 16384`, `seed = None`, `temperature = None`,
  timeout 240 s, model entirely in GPU0 VRAM, one generation per question and replication,
  n ≥ 3 replications.
- **raw / OFF / ON**: in the paired Validator design, one generation gives the raw answer; OFF
  = the raw answer; ON = the same raw answer + the qualification block. Never two generations.
- **Scoring**: manual, PASS = 1, PARTIAL = 0.5, FAIL = 0, over 15 questions (or 45 answers for
  3 replications); one grader, no blinding possible. Nothing is graded automatically.
- **Valid vs invalid**: a run is valid only if `complete = true`, 15/15 questions, 0
  errors/timeouts, the run identity matches the reference conditions and `gpu_fraction = 1.0`.
  A run under CPU offload is invalidated with a `.INVALID.txt` sidecar and never scored.
- **Stochasticity**: σ = 0.75/15 between replications with byte-identical contexts (n = 4);
  single runs are read for invariant failures, not for score differences.
- **Run identity**: `python -m orchestrator.capture_run_identity --label <run>` records model
  digest, Ollama version, `num_ctx`, timeout, prompt and question hashes, module hashes, RAG
  index identity and the placement of every process on every GPU.

Full protocol, per-question table and artefact registry: `docs/BENCHMARK.md`.

---

## 10. Results

Scientific-model results are in §1.6 and `docs/SCIENTIFIC_BACKGROUND.md`. Answer-layer results
(all `mistral-nemo:12b` unless stated; sources in `docs/BENCHMARK.md` §4–6):

| Experiment | Condition | Measured result | Interpretation (bounded) |
|---|---|---|---|
| LLM model comparison (2026-09-02d) | `llama3.2:latest` vs `mistral-nemo:12b`, context v2, single-variable swap | 1.0/15 (10 refusals) vs **5.5/15** | Nemo is the only model that completes 15/15 with a usable score on this GPU; 5.5 is the reference baseline |
| Temporal-context restructuring (2026-09-03) | after the multi-window context | 4.5 (corrected Q9 text) / 3.5 (frozen text) | within the noise floor; no demonstrated regression from the restructuring itself |
| Replication (2026-09-07b/c) | prompt v2, n = 4 | mean 5.62, σ 0.75, range 4.5–6.0 | run-to-run variation is of the same order as the interventions being tested |
| Grounding fix (2026-09-07d) | numeric/phase presence check | grounded count 8 → 11/15 while the score fell | grounding rate is not a quality score |
| FULL vs OBSERVED_ONLY (2026-09-09→11) | paired, n = 3 | 6.83 (σ 2.52) vs 4.50 (σ 1.00), Δ −2.33, t(2) = −1.32, n.s. | removing model data costs on model-derived questions; different coverage, not a proven difference |
| PREDICTION_ONLY (2026-09-11) | n = 3, unpaired | 4.67 (σ 1.44); on observed-answer questions 0.0/6 vs 5.5/6 for FULL, while staying "grounded" (Q3: model `t7` stated as observed 3/3) | the LLM propagates model outputs as observations and can add unsupported claims; motivated the Validator |
| Context v2 vs compact_v1 (2026-09-12) | paired, n = 3 | 5.00 vs 6.17, Δ +1.17, t(2) = 1.07, n.s.; Q3 3/3 → 1/3 | not adopted as a demonstrated improvement |
| Validator v1.2 OFF vs ON (2026-09-12) | compact_v2, paired, n = 3 | OFF 18.5/45, ON 19.0/45; 21/45 raw answers hallucinated; 12 targeted correctly, 4 wrongly, 5 false positives | valid benchmark; gain not demonstrated; not comparable with the v2-context history |
| Validator v1.3 (2026-09-14) | read-only replay of the 45 stored answers | v1.2 reproduced bit-for-bit before the change; after it, 0 false positives, all correct detections kept, qualification on 16/45 instead of 27/45 | precision fix; awaiting its valid rebenchmark |
| Istanbul Consensus grounding | `[SCIENTIFIC]` block + rules R7/R8/R14 | provides phase order and vocabulary; corpus silent on morphokinetic reference times and on direct/reverse/chaotic cleavage definitions | useful reference; never justifies an unsupported biological interpretation |

Structural findings: Q2 is a real data gap (no segment start instant in the context); Q13/Q14
are gap tests (no reference times, unverified time unit → NOT_ASSESSABLE is the honest
answer); Q15 (direct / reverse / chaotic cleavage) must stay NOT_ASSESSABLE from phase labels
alone unless supporting evidence is introduced.

---

## 11. Current Open Work

**STATUS: `WAITING_FOR_VALIDATOR_V1_3B_REBENCHMARK`** (as of 2026-09-17).

- Validator v1.3 is implemented and frozen: the six functional files are byte-identical to the
  hashes hard-coded in the gated launcher, local and server (re-verified read-only on
  2026-09-17); 184 validator/integration tests passed on the server on 2026-09-15.
- The rebenchmark has **not** been executed. The one attempt (2026-09-14, label
  `validator_v1_3`) was invalidated: an external hyper-parameter search held 6.3 GB of GPU0,
  Ollama placed 48 % of the model in VRAM, Q1 took 97 s instead of 15.8 s and Q5 timed out. The
  partial artefact is kept with `INVALID_REASON = GPU_CONTENTION_CPU_OFFLOAD` and is never scored.
- The external job observed on 2026-09-13→16 finished; **a new external job of the same kind
  started on 2026-09-16 and occupies GPU0 again**. Its duration cannot be predicted. It must
  not be interrupted. (Process ids and memory figures are transient; check live with the
  read-only commands of the internal GPU-server procedures.)
- The gated launcher (`Training/experiments/archive/validator_benchmark/launch_v1_3_gated.sh`,
  active copy `/tmp/launch_v1_3_gated.sh` on the server) refuses to start unless GPU0 carries
  no foreign process and `gpu_fraction == 1.0`; it writes label `validator_v1_3b`.

Immediate next scientific action:

```
WAIT FOR GPU0 TO BECOME VALID
→ RUN THE VALIDATOR V1.3B GATE            bash /tmp/launch_v1_3_gated.sh validator_v1_3b   (on the server)
→ POST-RUN VALIDATION                      complete, 15/15, 45/45, 0 errors/timeouts, raw == OFF 45/45, pairing, GPU0 only
→ RAW / OFF / ON MANUAL SCORING            frozen rubric, same conventions as v1.2
→ COMPARISON WITH THE FROZEN BASELINE      v1.2 OFF 18.5 / ON 19.0 /45 (compact_v2); v2-context history as secondary
```

Deferred until after that comparison, because they change the run identity or the benchmark
conditions: updating the paths cited by `rag/inventory.py`'s excluded list beyond what was
committed, re-classifying the six guides into the corpus, re-ingesting the index, moving any
module under `orchestrator/`, `validator/` or `rag/`, syncing the restructured tree to the
server, changing the production `LLM_MODEL`, adopting the compact context. Non-blocking open
decisions: versioning `docs/archive/` and `docs/reference/`, the five untracked files,
archiving `WebApplication/`, rotating the legacy PostgreSQL credential (§14).

---

## 12. How to Start

First day, in order:

1. `git status && git log --oneline -5` — expect a clean tree at or after `f8e64cd`.
2. Read this README, then `docs/handover/HANDOVER.md`.
3. Get the session history: `docs/archive/sessions/PROJECT_CHECKPOINT.md` is not in Git; ask
   for the original checkout or the 2026-09-15 snapshot (`HANDOVER.md` §"Source of truth").
4. Walk the tree with §5 open; read `Training/experiments/README.md`.
5. Environment: `conda create -n embryo_env python=3.10 && pip install -r requirements.txt`
   (`docs/DEVELOPMENT.md` §2). Ollama is a system service on the server, not a pip dependency.
6. Tests, from the repository root: `pytest Tests/` and the four `node Tests/webapp_api/*.mjs`.
   Expected: everything passes on the server; on a machine without the data, tests that need
   `Embeddings/`, `Cache/`, `RagIndex/` or `docs/corpus/` skip, and a few float-tolerance
   tests may fail (`docs/DEVELOPMENT.md` §3).
7. Inspect the current experiment state on the server, read-only (internal GPU-server
   procedures): is GPU0 free, does `validator_v1_3b` exist?
8. Do not modify anything under `Training/orchestrator/`, `validator/`, `rag/` or the RAG
   documents before the v1.3b comparison exists.

---

## 13. Reproducing Experiments

Generic workflow, applied to every experiment in this project:

1. Identify the exact experiment and its runner (`Training/experiments/README.md`).
2. Identify the commit (`git log`, and the run identity JSON of the reference run if any).
3. Identify the dataset split (`val` for anything exploratory; `test` only for the two
   pre-registered reports, never again).
4. Identify the configuration (CLI flags; `Configs/config.ini` is not read).
5. Identify the model (checkpoint SHA-256, frozen Semi-HMM directory).
6. Identify the corpus version (the 30 RAG documents' content hashes in `RagIndex/manifest.json`;
   `istanbul_corpus_sha256` in every benchmark artefact).
7. Identify the benchmark version (question hashes, prompt hash, context format).
8. Verify the GPU conditions (GPU0 free of foreign processes, `gpu_fraction = 1.0`).
9. Execute from inside `Training/` with the validated commands (`docs/REPRODUCIBILITY_GUIDE.md` §3).
10. Verify the artefact (complete, expected counts, hashes).
11. Record provenance (`capture_run_identity`, the artefact's own provider block).
12. Compare only against runs with a compatible identity.

Results and logs live under `Results/evaluation/<experiment>/` on the server (copies in the
2026-09-15 snapshot; SHA-256 of the benchmark artefacts in `docs/BENCHMARK.md` §8). Reference
numbers to reproduce exactly: `docs/REPRODUCIBILITY_GUIDE.md` §4.

---

## 14. Safe Modification Rules

**SAFE** (no effect on any result):
- documentation changes outside the 30 RAG documents and `docs/corpus/`;
- tests that do not alter scientific behaviour;
- isolated tooling (deploy script, CI, linting), provided nothing under `Training/` changes.

**REQUIRES CARE** (changes what the system computes or what the LLM reads; needs a comparison
on the frozen benchmark and a new run identity):
- model code (`Training/evaluation/models/`, `Training/reporting/`);
- temporal context, context builder, compact rendering;
- prompts (`orchestrator/prompt.py`; a test pins that it contains no value);
- the RAG corpus (any of the 30 documents), `rag/inventory.py`, chunking, embedding model;
- retrieval (`rag/retrieval.py`, `build_retrieval_query`);
- the Validator (`Training/validator/`, `scientific_validation.py`);
- the benchmark runners; dataset handling.

**INVALIDATES COMPARISON** (a run produced after this cannot be compared with any earlier run):
- changing the benchmark questions, anchor or grading conventions;
- changing the evaluation protocol (replications, pairing, timeout, `num_ctx`, seed/temperature);
- changing the dataset, the annotations or the splits;
- silently changing the corpus or re-ingesting the index;
- silently changing the model or context configuration (`LLM_MODEL`, context format, Validator on/off);
- mixing incompatible run identities (different module hashes, Ollama version or digest, GPU placement).

### Security

The legacy `WebApplication/.env` (a real PostgreSQL credential) was tracked and pushed in
July 2026. On 2026-09-15 it was untracked, ignored (`.env`, `*.env`, `WebApplication/.env`),
replaced by `WebApplication/.env.example`, and purged from the Git history of the internal
repository (filter-repo); this public repository was created afterwards from the cleaned history
and never contained it. **The credential itself has not been rotated**: the database host is reachable neither
from the local machine nor from the GPU server, so rotation is a human action on the database
machine (`docs/archive/audits/SECURITY_HARDENING_2026-09-15.md`). Local environment
configuration must stay outside Git; never write a secret into a document, a commit or a chat.
Nothing under `Training/` uses PostgreSQL. The BFF has no authentication and binds to localhost
by default; do not expose it.

---

## 15. Reference, licence

Dataset: *Human embryo time-lapse video dataset*, Zenodo, DOI 10.5281/zenodo.7912264,
creators Gomez Tristan, Feyeux Magalie, Boulant Justine, Normand Nicolas, Paul-Gilloteaux
Perrine, David Laurent, Fréour Thomas, Mouchère Harold; licence CC BY-NC-SA 4.0
(non-commercial, share-alike) — it binds every use of the data and of models trained on it.

Original work: *Spatio-Temporal Transformers for High-Accuracy Detection of Embryo
Developmental Transitions*, Aissa Benfettoume Souda, Mohammed El Amine Bechar, Souaad
Hamza-Cherif, Jean-Marie Guyader, Marwa Elbouz, Fréderic Morel, Aurore Perrin, Nesma Settouti
(2025, LSL team — Light Scatter Learning, LabISEN / ISEN Ouest; author list as in the original
README's BibTeX, whose URL field says "will be updated"). Original author and copyright holder:
§1.2.

External scientific reference used by the Validator: Coticchio et al., *The Istanbul consensus
update: a revised ESHRE/ALPHA consensus on oocyte and embryo static and dynamic morphological
assessment*, Human Reproduction 2025, DOI 10.1093/humrep/deaf021 (PDF under `Ressources/`,
transcription under `docs/corpus/`).

Licence of the code: MIT (`LICENSE`, copyright 2025 Aissa BenFettoume Souda). Licence of the
dataset: CC BY-NC-SA 4.0.

### Documentation map

| Need | Read |
|---|---|
| take ownership | `docs/handover/HANDOVER.md` |
| what runs and how | `docs/ARCHITECTURE.md` |
| the science, established results, limits | `docs/SCIENTIFIC_BACKGROUND.md` (then the RAG documents `HANDOFF.md`, `RESEARCH_BLUEPRINT.md`, `SCIENTIFIC_REPORT.md`) |
| inputs, rebuild commands, reference numbers | `docs/REPRODUCIBILITY_GUIDE.md` |
| the 15 questions, protocol, results, Validator | `docs/BENCHMARK.md`, `docs/reference/RAG_FIXED_QUESTION_BENCHMARK.md` |
| environment, tests, conventions, pitfalls | `docs/DEVELOPMENT.md` |
| GPU server | internal procedures, handed over separately (not in this repository) |
| RAG corpus and index | `docs/handover/RAG_OPERATIONS.md` |
| web application | `docs/handover/WEBAPP_GUIDE.md` |
| everything in `docs/` | `docs/README.md` |

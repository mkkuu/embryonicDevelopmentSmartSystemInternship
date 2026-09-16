# embryonicDevelopmentSciMLExtension

Detection of developmental-stage transitions in human embryo time-lapse videos, extended with
a scientific-machine-learning study of the temporal dynamics behind those transitions and a
local, grounded question-answering application over the resulting models.

This README is the entry point for a newcomer. It says what exists, what has been established,
what is legacy, and where to go next. Detailed guides live in `docs/` (see §12).

## 1. Purpose

Two things live in this repository:

1. **The original classifier** (`Training/*.py`, `WebApplication/`, July 2026): ResNet18 /
   TimeSformer models that detect, in an 8-frame window of a time-lapse video, whether a
   developmental transition occurs, plus a Flask + PostgreSQL application for doctors and
   administrators. Paper reference in §13.
2. **The SciML extension** (everything under `Training/{embeddings, evaluation, rag,
   orchestrator, validator, reporting, webapp_api}/`, `Tests/`, `docs/`, July–September 2026):
   it caches the classifier's embeddings, asks whether *explicit temporal-dynamics modelling*
   adds anything to frame-independent classification, builds a hidden-Markov / semi-Markov
   model of the developmental phase on top of the embeddings, serves that model through a
   read-only API, and lets a local language model answer questions about one embryo's
   trajectory with every number grounded in the served data and every scientific claim
   checked against one external reference (the Istanbul Consensus 2025).

## 2. Scientific objective

The programme's question (`docs/SCIENTIFIC_BACKGROUND.md` §1): is embryo development, as seen
in time-lapse imaging, governed by a shared low-dimensional dynamical law, and do the discrete
clinical phases reflect that law? The gating hypothesis **H1** — "explicit dynamics beats
frame-independent classification" — was tested first and is **not supported** on this dataset:
the embedding content alone reaches AUROC 0.738 on the transition target, and neither fixed,
learned-linear nor GRU dynamics add to it once a frame-shuffle control is applied. The
phase-as-hidden-state branch (HMM, then Semi-HMM with explicit phase durations) reaches AUROC
≈ 0.64 on validation with poor calibration; the Semi-HMM is the model served today.

Throughout the code and the documentation, five kinds of statement are kept apart and labelled:

| Label | Meaning | Example |
|---|---|---|
| OBSERVED | an annotation of the dataset | "the annotated phase at window 156 is t4" |
| MODEL-DERIVED | a prediction or estimate of the Semi-HMM | "the model's most likely phase is t7" |
| DERIVED | a deterministic computation on observations or predictions | "last annotated transition before this window: t4 → t6" |
| SCIENTIFIC | external knowledge (Istanbul Consensus 2025) | phase order, morphology vocabulary |
| LIMITATION | a field of the context that bounds the answer | `time_unit = unknown/unverified` |

A model output presented as an observation is the central failure mode this project measures
and guards against.

## 3. Current architecture

```
Browser → Training/webapp_api (BFF, Flask :8001)
            ├─ Training/reporting   read-only inference on the frozen Semi-HMM (in-process; also standalone on :8000)
            └─ Training/orchestrator.answer_question()
                 router (keywords) → tools (reporting + RAG retrieval over 30 project docs, Chroma)
                 → temporal context → context builder (labelled blocks) → local LLM via Ollama
                 → grounding check (presence of every value) → Scientific Validator (Istanbul corpus, rules R1–R14)
```

Scientific components behind it: `Training/embeddings/` (cached ResNet18 embeddings),
`Training/evaluation/` (model interface, bootstrap harness, the seven models),
`Training/experiments/` (the runners that produced every result), the frozen artefacts under
`Results/` on the GPU server. Full description: `docs/ARCHITECTURE.md`.

## 4. Repository structure

| Path | Role |
|---|---|
| `Training/` | original pipeline (`preProcess.py`, `train.py`, `train_balanced.py`, `DataSet.py`, `ModelBuilder.py`, …) and the SciML packages listed above (production and scientific libraries only) |
| `Training/experiments/` | every experiment and benchmark runner, grouped by experiment (`e1_ladder/`, `gru/`, `hmm_semi_hmm/`, `llm_regression/`, `context_ablations/`, `rag_llm_diagnostics/`, …) with an index README; `archive/` holds the one-off scripts salvaged from the server. Nothing here is used by the application |
| `Tests/` | 74 test files / 1 211 tests mirroring the SciML packages, synthetic data only, plus 4 Node tests for the frontend |
| `Configs/config.ini` | training configuration of the original pipeline; **not read on Linux** because of a Windows path bug, kept for the record |
| `Results/` | gitignored; exists only on the GPU server: checkpoints, frozen HMM/Semi-HMM, every experiment and benchmark artefact |
| `Ressources/` | logos and the Istanbul Consensus 2025 PDF from which the corpus was transcribed |
| `docs/` | the guides below, the 30 documents that form the RAG corpus (frozen names) and `corpus/` (Istanbul transcription) are versioned; `reference/` and `archive/` are **local-only** (not in Git, present on the original checkout and in the 2026-09-15 snapshot) — see `docs/README.md` |
| `WebApplication/` | the original doctor/admin Flask application — **legacy**, see §11 |
| `deploy2GPUServ.sh` | rsync of the code to the GPU server (excludes data, results, `docs/`) |

Also gitignored and server-only: `Data/` (dataset), `Embeddings/`, `RagIndex/` (vector index),
`Cache/` (inference cache).

## 5. Current application

```bash
cd Training
python -m reporting.warm_cache --split val      # optional, pre-fills the inference cache
python -m webapp_api.app                        # http://localhost:8001
```

Endpoints of the BFF (verified in `Training/webapp_api/app.py`): `GET /`, `/health`, `/videos`,
`/videos/<id>`, `/videos/<id>/timeline`, `/videos/<id>/window/<w>`,
`/videos/<id>/window/<w>/frame?which=first|last`, `/questions`, `POST /chat` with
`{question | question_id, video_id, window, split}`. `split` defaults to `val`; `test` is
refused with 403 everywhere. Requires the frozen Semi-HMM, the embeddings, the images, the
vector index and an Ollama service with the model named by `LLM_MODEL` (default
`llama3.2:latest`). The standalone Reporting API (`python -m reporting.api`, port 8000) exposes
the same inference without the language model.

## 6. Scientific pipeline

```
Data/ (images + phase annotations)
  → preProcess.py (video-level splits 492/106/106) → DataSet.py (8-frame windows, stride 1, target consistency_flag)
  → train_balanced.py → Results/resnet18_balanced/best_model.pth
  → embeddings.build_cache → Embeddings/resnet18/{train,val,test}/embeddings.pt (512-d)
  → experiments/* runners on the evaluation/ models (E1 ladder, GRU, HMM k=7 sweep, Semi-HMM) → Results/evaluation/*
  → reporting (frozen Semi-HMM: filtering posteriors, next-phase distributions, observed transitions, frame mapping)
  → orchestrator (temporal context, tools, RAG, LLM, grounding) → validator (Istanbul corpus, rules) → answer
```

## 7. Models

| Model | Status | Where |
|---|---|---|
| ResNet18 (class-balanced) | production encoder; **sole surviving checkpoint** | `Results/resnet18_balanced/best_model.pth` |
| Semi-HMM (dmax 268, negative binomial, unweighted emission) | production / reference, served by `reporting` | `Results/evaluation/semi_hmm_weekend_phaseF/model/` |
| HMM k=7 (α 2.0, ρ 0.5) | reference for the Semi-HMM comparison | `Results/evaluation/e1_hmm_k7_sweep/best_model/` |
| E1 ladder (`base_rate`, `identity_dynamics`, `persistence`, `linear_ssm`) and GRU | benchmark / negative results, not served | `Training/evaluation/models/`, `Results/evaluation/e1_*` |
| Emission re-weighting variants | closed branch, negative result | `Results/evaluation/emission_*` |
| TimeSformer | original pipeline option; no checkpoint on the server | `Training/ModelBuilder.py` |
| LLMs (Ollama) | `llama3.2:latest` production default; `mistral-nemo:12b` benchmark model | `Training/orchestrator/llm_config.py` |

## 8. Benchmarks

Fifteen frozen questions (Q1–Q15, French) about one anchor: patient `Patient_319`, window
156, validation split, where the annotation shows a `t4 → t6` skip and the model believes
`t7`. Manual grading (PASS/PARTIAL/FAIL), replication n ≥ 3 because the run-to-run noise is
0.75/15 with identical contexts. Reference results: `mistral-nemo:12b` 5.5/15 baseline; the
three-condition experiment (FULL 6.83 vs OBSERVED_ONLY 4.50 vs PREDICTION_ONLY 4.67) showed
that the language model presents model outputs as observations, which motivated the
Scientific Validator; the paired Validator OFF vs ON run (v1.2) gave 18.5 vs 19.0 /45 with 5
false positives, fixed in v1.3 without a new benchmark yet.

**Validator v1.3 rebenchmark status: WAITING_FOR_VALID_GPU.** The one attempt was invalidated
by an external job on GPU0 (CPU offload) and must never be scored. Everything, including the
artefact registry with hashes, is in `docs/BENCHMARK.md`.

## 9. Reproducibility

`docs/REPRODUCIBILITY_GUIDE.md` lists every input with its identity and backup, the validated
command sequence (data → classifier → embeddings → evaluation → index → serving → benchmark),
the reference numbers to reproduce, and what silently breaks comparability. In short:
Python 3.10 conda env from `requirements.txt`; the dataset; the server's GPU0 only; an Ollama
service; the 30 RAG documents and the Istanbul corpus with unchanged content. Tests:
`pytest Tests/` from the repository root plus four `node` tests (`docs/DEVELOPMENT.md`).

## 10. Known limitations

- The corpus the application reads at runtime (`docs/corpus/` and the 30 RAG documents) is
  **ignored by Git** (`docs/` is in `.gitignore`); it exists in the local checkout, on the
  server and in the 2026-09-15 snapshot. Versioning it is a pending decision.
- The whole data side lives on one server; the classifier checkpoint has one NAS copy.
- The original `Results/resnet18/` checkpoint was destroyed on 2026-07-24; the script that
  produced `gru_identity_threshold_analysis` was never found.
- Time unit: every artefact says `unknown/unverified`; durations are in windows.
- Test split consumed twice (E1, GRU) and locked; calibration of the HMM/Semi-HMM never beats
  a constant-rate baseline; overlapping windows violate emission independence (documented).
- The Istanbul Consensus is a reference for vocabulary and good-practice statements, explicitly
  **not a standard of care**, and the Validator's rule R14 never lets it validate a prediction.
- The language model is non-deterministic run to run; single benchmark runs are not evidence.
- `Configs/config.ini` is not read; the legacy PREDICT endpoint returns random predictions.
- GPU0 is shared with other users' jobs; benchmarks wait for it.

## 11. Legacy: `WebApplication/`

The original Flask + PostgreSQL application (login, doctor/admin CRUD, embryo upload, a
prediction endpoint). It is **not on the current production path**: no import to or from
`Training/`, no tests, plain-text password check, `debug=True`, and its prediction endpoint
loads a checkpoint path that no longer exists with the wrong `torch.load` contract, so it
silently answers with random predictions (`is_random: true`). Launch `cd WebApplication &&
flask run` (port 5000) with a `.env` built from `WebApplication/.env.example` and the schema in
`Dataset_shema.sql`. Its future (archive or separate repository) is decided after the
PostgreSQL side is reviewed.

## 12. Handover — if you take over the project, start here

1. This README.
2. `docs/ARCHITECTURE.md` — what runs and how it is wired.
3. `docs/SCIENTIFIC_BACKGROUND.md` — the question, the data, what is established and what is not.
4. `docs/REPRODUCIBILITY_GUIDE.md` — inputs, commands, reference numbers, what not to touch.
5. `docs/BENCHMARK.md` — the 15 questions, the protocol, the results, the Validator.
6. Check the state of the Validator v1.3 rebenchmark (`docs/BENCHMARK.md` §6; the gated
   launcher waits for GPU0) before changing anything under `Training/orchestrator/`,
   `validator/`, `rag/` or the RAG documents.
7. Only then, `docs/DEVELOPMENT.md` for the environment and conventions, and
   `docs/archive/` (session logs, experiment reports, audits) for the history — note that
   `docs/archive/` and `docs/reference/` are not in Git; a fresh clone does not have them
   (see `docs/README.md`).

## 13. Reference, licence

Dataset: Human embryo time-lapse video dataset, Zenodo, DOI 10.5281/zenodo.7912264.
Original work: *Spatio-Temporal Transformers for High-Accuracy Detection of Embryo
Developmental Transitions*, A. Benfettoume Souda, M. E. A. Bechar, S. Hamza-Cherif,
J.-M. Guyader, M. Elbouz, F. Morel, A. Perrin, N. Settouti (2025, LSL team, LabISEN / ISEN
Ouest). External scientific reference used by the Validator: Coticchio et al., *The Istanbul
consensus update: a revised ESHRE/ALPHA consensus on oocyte and embryo static and dynamic
morphological assessment*, Human Reproduction 2025, DOI 10.1093/humrep/deaf021.
Licence: MIT (`LICENSE`).

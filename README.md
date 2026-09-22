# Embryonic Development — SciML Extension

Detection of developmental-stage transitions in human embryo time-lapse videos, extended with
a scientific-machine-learning (SciML) study of the temporal dynamics behind those transitions.
A local, grounded question-answering application is built on top of the resulting models.
This is research code; it has no clinical use.

## Overview

The **original project (2025)** is a ResNet18 / TimeSformer classifier that looks at a short
window of time-lapse frames and predicts whether a developmental transition occurs inside it,
together with a Flask + PostgreSQL application for doctors and administrators.

The **SciML extension (July–September 2026)** is built on top of that code by composition: the
original modules are imported, never edited. It asks whether explicit modelling of temporal
dynamics adds anything to frame-independent classification, fits a hidden-Markov and a
semi-Markov model of the developmental phase, and serves the result through a read-only API.

On top of these models, a retrieval-augmented (RAG) application lets a local language model
answer questions about one embryo's trajectory. Every number is checked against the served
data, and every scientific claim is qualified against one external reference, the Istanbul
Consensus 2025. The user interface and the benchmark questions are in French.

## Scientific question

> Is human embryo development, as captured by time-lapse imaging, governed by a shared,
> low-dimensional dynamical law — and do the discrete clinical categories (phase, arrest)
> correspond to that law's structure, or to something else?

The embryo is treated as a partially observed dynamical system: a hidden state, a discrete
developmental regime with its own sojourn time, and observations given by the classifier's
embeddings of overlapping frame windows. The experiments walk down a chain of simplifications,
from non-linear and linear state-space models to a semi-Markov model of the phase alone. Where
one can stop in that chain is the hypothesis under test. The formulation is in
`docs/HANDOFF.md`, the hypotheses in `docs/RESEARCH_BLUEPRINT.md`.

## What the project contains

- **Original classifier** — ResNet18 / TimeSformer transition detector and its training pipeline.
- **Embedding pipeline** — caches the classifier's embeddings once per split.
- **E1 temporal-dynamics experiments** — an ablation ladder from a base rate to linear and
  recurrent (GRU) dynamics, with a frame-shuffle control and patient-level bootstrap.
- **HMM / Semi-HMM** — the developmental phase as a hidden state, with explicit phase durations.
- **Reporting API** — read-only inference on the frozen Semi-HMM.
- **RAG** — a vector index over the project's own documents.
- **Orchestrator** — deterministic question router, typed tools, a local LLM (Ollama) and a
  grounding check that verifies presence, not correctness.
- **Scientific Validator** — qualifies each claim of an answer against its context and the
  Istanbul Consensus 2025; it never rewrites the answer and is not a biological oracle.
- **Web application** — chat interface over the components above.
- **Benchmark** — 15 frozen questions on one anchored window, graded manually.

## Dataset

*Human embryo time-lapse video dataset*, Zenodo, DOI
[10.5281/zenodo.7912264](https://doi.org/10.5281/zenodo.7912264), licence **CC BY-NC-SA 4.0**.

Time-lapse image sequences of human embryos, annotated frame by frame with 15 chronological
developmental phases. Both parts of the project use it unchanged; it is not distributed here.

## Scientific status

- **H1 is not supported**: no linear or non-linear dynamics model beats frame-independent
  classification once the frame-shuffle control is applied.
- **H2–H4 were never tested.**
- The **Semi-HMM is frozen**. It is the reference model served by the application. Its gain
  over the plain HMM is small and not significant, and no configuration is well calibrated.
- The language model tends to present model outputs as observations; this motivated the
  Scientific Validator. **Validator v1.3 exists, but its valid rebenchmark is still pending**,
  so no benefit of the Validator on the benchmark is demonstrated.

Numbers, confidence intervals and limits: `docs/SCIENTIFIC_BACKGROUND.md` and `docs/BENCHMARK.md`.

## Scientific integrity

- Never modify the dataset, the annotations or the splits.
- Never use the `test` split for development; every serving layer refuses it.
- Keep observed information and model-derived information apart, in code and in answers.
- Never invent a biological conclusion; when the data cannot support one, say so.
- Never name a time unit that is not sourced; durations are expressed in windows.
- Do not change benchmark conditions silently, and replicate before reading a difference.
- Preserve the provenance of benchmark artefacts; an invalid run is kept, marked, never scored.
- On shared hardware, never interfere with another user's job.

## Repository structure

- `Training/` — the original pipeline (`*.py`) and the extension packages: `embeddings/`,
  `evaluation/`, `reporting/`, `rag/`, `orchestrator/`, `validator/`, `webapp_api/`, and
  `experiments/` (one folder per experiment).
- `Tests/` — unit tests on synthetic data and mocked dependencies.
- `docs/` — guides and the project documents indexed by the RAG layer; `docs/corpus/` holds the
  Istanbul Consensus 2025 transcription; `docs/handover/` holds the successor documents.
- `Configs/` — configuration file of the original pipeline.
- `Ressources/` — logos and the Istanbul Consensus 2025 PDF.
- `WebApplication/` — the original Flask + PostgreSQL application (legacy).

Data, trained models, embeddings, the vector index and caches are not distributed.

## Getting started

### Prerequisites

- Python 3.10 and the packages of `requirements.txt` (PyTorch, Flask, ChromaDB,
  sentence-transformers, …). Tests need nothing else.
- To run the application: a local [Ollama](https://ollama.com) service on
  `http://localhost:11434` with the model named by `LLM_MODEL` already pulled (default
  `llama3.2:latest`), and the inputs that are not distributed, placed at the repository root:
  the dataset under `Data/`, the trained classifier and the frozen Semi-HMM under `Results/`,
  the embedding cache under `Embeddings/`, the vector index under `RagIndex/`. They are rebuilt
  in the order given by `docs/REPRODUCIBILITY_GUIDE.md` §3 (dataset, classifier, embeddings,
  RAG index); training the classifier and extracting the embeddings need a GPU, the rest
  runs on CPU.

### Environment

```bash
conda create -n embryo_env python=3.10
conda activate embryo_env
pip install -r requirements.txt
pytest Tests/                    # from the repository root, not Training/
```

### Running the application

Every command runs from `Training/`: the modules resolve `Data/`, `Results/`, `Embeddings/`
and `RagIndex/` relative to it.

```bash
cd Training
python -m reporting.warm_cache --split val   # optional: pre-fills the inference cache
python -m webapp_api.app                     # http://localhost:8001
```

`LLM_MODEL` and `LLM_TIMEOUT_SECONDS` (default 120 s) select the generative model and its
per-question budget; the other variables are listed in `docs/ARCHITECTURE.md` §3. The
standalone Reporting API (`python -m reporting.api`, port 8000) is optional: the application
calls the same functions in-process.

### Using the interface

Open `http://localhost:8001`. The page lists the patients of the `val` split; selecting one
shows the phase timeline, the frame viewer, the 15 frozen questions and a free-text chat. The
`test` split is refused by every layer. If Ollama is unreachable, the chat returns an explicit
fallback answer instead of failing.

### Where to start

1. `docs/handover/HANDOVER.md` — the first hour, day and week of a newcomer.
2. `docs/ARCHITECTURE.md` — what runs, how the layers are wired, what is legacy.
3. `docs/SCIENTIFIC_BACKGROUND.md` — the established results and their limits.
4. `docs/DEVELOPMENT.md` — conventions and known pitfalls before changing code.
5. `docs/handover/WEBAPP_GUIDE.md` and `docs/handover/RAG_OPERATIONS.md` — operating the
   application and its index.

## Reproducibility

- `docs/REPRODUCIBILITY_GUIDE.md` — inputs, rebuild order, reference numbers.
- `docs/DEVELOPMENT.md` — environment, tests, conventions, known pitfalls.
- `Training/experiments/README.md` — the runner of each experiment.

Tests run without data, GPU or language model; experiments need the dataset, a GPU and Ollama.

## Web application

`Training/webapp_api/` is the current application: patient list, phase timeline, frame viewer,
the frozen question panel and a free-text chat. It has no authentication and is meant to stay
on localhost. `WebApplication/` is the original doctor / administrator application, kept as
legacy; its database settings come from a local `.env` (template `WebApplication/.env.example`).
No credential is stored in this repository. Details: `docs/handover/WEBAPP_GUIDE.md`.

## Documentation

- `docs/README.md` — map of the documentation.
- `docs/SCIENTIFIC_BACKGROUND.md` — the science, the established results and their limits.
- `docs/RESEARCH_BLUEPRINT.md` — the hypothesis programme.
- `docs/BENCHMARK.md` — the 15 questions, the protocol, the results, the Validator.
- `docs/handover/HANDOVER.md` — taking over the project.
- `docs/handover/RAG_OPERATIONS.md` — the RAG corpus and index.
- `docs/handover/WEBAPP_GUIDE.md` — the web applications.

Some guides cite internal session logs that are not distributed here.

## License & attribution

The **code** of this repository is under the **MIT** licence (`LICENSE`, copyright 2025 Aissa
BenFettoume Souda).

The original code and research direction come from Aissa Benfettoume Souda, LSL team (Light
Scatter Learning), LabISEN / ISEN Ouest: *Spatio-Temporal Transformers for High-Accuracy
Detection of Embryo Developmental Transitions*, Aissa Benfettoume Souda, Mohammed El Amine
Bechar, Souaad Hamza-Cherif, Jean-Marie Guyader, Marwa Elbouz, Fréderic Morel, Aurore Perrin,
Nesma Settouti (2025). The SciML extension was added in 2026 by the repository owner during an
internship.

Two resources keep their own licence and are **not relicensed** under MIT:

- the **dataset** — CC BY-NC-SA 4.0 (Gomez, Feyeux, Boulant, Normand, Paul-Gilloteaux, David,
  Fréour, Mouchère; DOI above);
- the **Istanbul Consensus 2025** PDF (`Ressources/`) and its transcription (`docs/corpus/`) —
  CC BY-NC 4.0: Coticchio et al., *The Istanbul consensus update: a revised ESHRE/ALPHA
  consensus on oocyte and embryo static and dynamic morphological assessment*, Human
  Reproduction 2025, DOI [10.1093/humrep/deaf021](https://doi.org/10.1093/humrep/deaf021).

## AI-assisted development

The 2026 extension was developed with the assistance of an AI coding agent driven by written
missions.

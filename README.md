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

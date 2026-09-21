# Handover — taking ownership of this repository

Written 2026-09-17 from the verified state of the repository, the GPU server (read-only) and
the session log. This document is procedural: what to read, what to run, what to check before
each kind of action. The project itself is explained in the root `README.md`; the current
checkpoint is `README.md` §11 (mirrored in `docs/archive/sessions/PROJECT_CHECKPOINT.md`,
which is not in Git — see "Source of truth" below).

STATUS at hand-over: **`WAITING_FOR_VALIDATOR_V1_3B_REBENCHMARK`**.

The GPU-server procedures (access, read-only diagnostics, gated launcher, restoring inputs)
are internal to the laboratory. They are handed over separately to the successor and are not
part of the public repository; this document calls them "the internal GPU procedures".

Project origin: the original codebase and research direction (transition detection on
time-lapse windows, the ResNet18 / TimeSformer classifier, the doctor/admin application) come
from the 2025 paper whose first author, Aissa Benfettoume Souda (LSL team, LabISEN / ISEN Ouest),
is the copyright holder named in `LICENSE`; the SciML extension was built on top of it from
July 2026 by the current repository owner. Details and sources: `README.md` §1.2.

---

## If you have just joined the project

### First hour — read and inspect, change nothing

1. `git status`, `git log --oneline --decorate -10`, `git remote -v`. Expect a clean tree at
   `f8e64cd` or later, branch `main`, one remote `origin`. Five untracked local files are normal
   (`README.md` §5.4).
2. Read `README.md` in full, including the "Scientific Integrity — DO NOT VIOLATE" banner.
3. Read `README.md` §11 (current open work) and `docs/BENCHMARK.md` §6 (the Validator and the
   invalidated attempt).
4. Skim `docs/ARCHITECTURE.md` §1 (the one-page diagram) and `Training/experiments/README.md`.
5. Locate the session history (`PROJECT_CHECKPOINT.md`) — see "Source of truth". If you only
   have a fresh clone, ask for the original checkout or the 2026-09-15 snapshot before going on.

### First day — run things that cannot change any result

```bash
# environment (local machine or server; docs/DEVELOPMENT.md §2)
conda create -n embryo_env python=3.10 && conda activate embryo_env
pip install -r requirements.txt

# tests, from the repository ROOT
pytest Tests/
node Tests/webapp_api/test_chat_dom_safety.mjs
node Tests/webapp_api/test_patient_selection.mjs
node Tests/webapp_api/test_viewer_navigation.mjs
node Tests/webapp_api/test_fixed_questions.mjs
```

Expected: on the server everything passes; on a machine without `Embeddings/`, `Cache/`,
`RagIndex/` or `Results/`, the tests that need them skip, and a handful of float-tolerance
tests may fail (`docs/DEVELOPMENT.md` §3 lists the known ones). Then, on the server and
read-only, check the experiment state (the internal GPU procedures): is GPU0 free of foreign processes,
does a `validator_v1_3b` artefact exist?

Optionally start the application on the server (`WEBAPP_GUIDE.md` §2) and ask one of the
frozen questions through the panel; it requires the frozen Semi-HMM, the embeddings, the
images, the vector index and a running Ollama.

### First week — understand before touching

- The science: `docs/SCIENTIFIC_BACKGROUND.md`, then the RAG documents `HANDOFF.md`
  (mathematical formulation), `RESEARCH_BLUEPRINT.md` (hypotheses), `SCIENTIFIC_REPORT.md`
  (results), `HANDOFF_SEMI_HMM.md` (the served model).
- The answer layer: `docs/BENCHMARK.md` end to end, `docs/reference/RAG_FIXED_QUESTION_BENCHMARK.md`
  (the frozen specification), `docs/reference/VALIDATOR_V1_IMPLEMENTATION.md`,
  `docs/reference/COMPACT_V2_AND_VALIDATOR_INTEGRATION.md`.
- The code: follow one question through `orchestrator/orchestrator.py::answer_question` with
  `Tests/orchestrator/` open; read `validator/rules.py` (one function per rule).
- The history: `docs/archive/sessions/PROJECT_CHECKPOINT.md`, newest section first, then the
  reports in `docs/archive/reports/` that `docs/archive/README.md` indexes.
- The operational constraints: the internal GPU procedures, `RAG_OPERATIONS.md`,
  `docs/REPRODUCIBILITY_GUIDE.md` §5 ("things that will silently break reproducibility").

---

## Before touching scientific code

Scientific code = `Training/evaluation/`, `Training/reporting/`, `Training/embeddings/`,
`Training/experiments/`, and the original pipeline files.

- [ ] The v1.3b rebenchmark comparison exists, or the change is provably outside every hashed
      module (`orchestrator/run_validator_experiment.py::_HASHED_MODULES`,
      `orchestrator/capture_run_identity.py::MODULES`).
- [ ] The frozen Semi-HMM directory, the classifier checkpoint and `Embeddings/` are not
      written to by the change (`reporting/model_loader.py` never refits; keep it so).
- [ ] The `test` split remains refused with 403 at every layer (`trajectory_service`,
      `reporting/api.py`, `webapp_api/app.py`).
- [ ] The original pipeline files (`Load_data.py`, `config_args.py`, `train.py`,
      `train_val_test_pipline.py`, `DataSet.py`) are composed, not edited.
- [ ] Tests added or extended; `pytest Tests/` run from the repository root.
- [ ] The change is written down: what, why, expected effect, and which earlier results it does
      or does not affect (session log entry).

## Before launching an experiment

- [ ] Pre-register: the question, the comparison, the metric, the number of replications
      (n ≥ 3 for anything involving the LLM), the split (`val`).
- [ ] Capture the run identity first: `cd Training && python -m orchestrator.capture_run_identity --label <label>`.
- [ ] GPU conditions verified read-only (the internal GPU procedures): no foreign process on GPU0; for
      an LLM benchmark, `gpu_fraction = 1.0` after loading the model.
- [ ] A new, unused `--run-label`; the runner refuses to overwrite an existing artefact, do not
      work around it.
- [ ] The conditions match the run you intend to compare with (model tag and digest, Ollama
      version, `num_ctx`, timeout, context format, Validator on/off, corpus hashes).
- [ ] Long runs in `screen` (no `tmux` on the server), Python called by full path
      (`~/miniconda3/envs/embryo_env/bin/python`), Python-side work with `CUDA_VISIBLE_DEVICES=""`.
- [ ] Output goes to a new directory or a new label under `Results/evaluation/`; nothing is
      overwritten.

## Before modifying the benchmark

Modifying = any change to `Training/orchestrator/fixed_question_benchmark.py`, the grading
conventions, the anchor, the protocol constants in `run_validator_experiment.py` /
`run_fixed_question_benchmark.py`, or the reference model.

- [ ] Understand that this **ends comparability** with every artefact in `docs/BENCHMARK.md` §8.
- [ ] The pending v1.3b comparison has been made, or explicitly abandoned in writing.
- [ ] The new version is frozen the same way: wording, anchor, success criteria, conventions in
      code and in a specification document; a new version identifier; the old one kept.
- [ ] A baseline is re-established with the new version before any intervention is measured.

## Before changing the RAG corpus

The corpus = the 30 `docs/*.md` in `Training/rag/inventory.py::INCLUDED` and the vector index
built from them; also `docs/corpus/` for the Validator. Details: `RAG_OPERATIONS.md`.

- [ ] The v1.3b comparison exists (a corpus change alters `document_context` for the benchmark).
- [ ] The change is versioned: edited documents committed, `inventory.py` updated (it is hashed
      into the run identity), re-ingestion run deliberately and logged (`RagIndex/manifest.json`
      records content hashes and `last_run_at`).
- [ ] `Tests/rag/` passes (`test_rag_inventory` checks that every top-level `docs/*.md` is
      classified INCLUDED or EXCLUDED).
- [ ] The 17 documents named by `orchestrator/compact_context.py` still exist at their paths.
- [ ] No dynamic data (predictions, per-window records, `Results/evaluation/*/analysis.json`)
      is ever added to the corpus.
- [ ] `docs/corpus/ISTANBUL_CONSENSUS_2025.md` is never edited; a new transcription is a new
      file with a new hash and a new traceability report.

## Before using the GPU server

- [ ] You have read the internal GPU procedures (rules and read-only diagnostics).
- [ ] You know that GPU1–GPU7 are not yours, that GPU0 is shared with other users' jobs, and
      that no external job is ever interrupted, signalled, reniced or environment-modified.
- [ ] You know that the server's code tree is older than Git HEAD and why (the internal GPU procedures).
- [ ] Any `rsync` toward the server is dry-run first (`rsync -n`), one source path per call.
- [ ] Nothing under `Results/`, `Data/`, `Embeddings/` on the server is deleted or overwritten.

---

## How to determine the current project state

Look, in this order, and trust the most specific evidence:

1. `README.md` §1.4 and §11 — the state as of the last documentation pass (dated).
2. `docs/archive/sessions/PROJECT_CHECKPOINT.md` — the newest section at the top is the last
   session's checkpoint with its own "next action". Local-only file.
3. The server, read-only: `ls -la Results/evaluation/validator_experiment/` (does a
   `validator_v1_3b` artefact, `.gate.txt`, `.log` or `.pid` exist?), `nvidia-smi -i 0`,
   `ollama ps` (the internal GPU procedures).
4. `git log` on the local checkout, and `git status` for uncommitted work.

If these disagree, the artefacts on the server are the fact; the documents describe intent.

## How to know whether an experiment is valid

- The artefact has `complete = true`, the expected number of rows (15 questions × replications)
  and 0 errors / timeouts.
- No sidecar `*.INVALID.txt` next to it. If one exists, the run is never scored, whatever its
  content.
- Its provider block matches the reference conditions (`docs/BENCHMARK.md` §3): model tag and
  digest, Ollama version, `num_ctx = 16384`, `seed = None`, `temperature = None`, timeout 240,
  context format; and its run identity (`run_identity_<label>.json` or the gate's `.gate.txt`)
  shows GPU0 only, `gpu_fraction = 1.0`, no foreign process.
- Its module hashes match those of the run it is compared with, except the modules the
  intervention intentionally changed.
- The split is `val`, the anchor is `Patient_319` / 156.
- For a paired Validator run: `raw == OFF` for every row, and `ON` starts with the raw answer.

## How to recover from a failed experiment

Never overwrite, never delete.

1. Stop your own process cleanly (SIGTERM to your PID only; never a foreign one).
2. Keep the partial artefact where it is. Write a sidecar `<artefact>.INVALID.txt` next to it
   with `INVALID_REASON`, the cause, the PID, the times and what was verified (model:
   `Results/evaluation/validator_experiment/…validator_v1_3.INVALID.txt`).
3. Record the event in the session log with the status it leaves the project in (for GPU
   contention: `WAITING_FOR_VALID_GPU`).
4. Re-run later under a **new label** (the runners refuse to overwrite; the gated launcher
   already uses `validator_v1_3b` for this reason).
5. Never score, summarise or cite a partial or invalid artefact.

## Who / what is the source of truth

| Question | Source of truth |
|---|---|
| What does the code do? | the code at `origin/main`, not any document |
| What was measured? | the artefacts under `Results/evaluation/` on the server (SHA-256 registry in `docs/BENCHMARK.md` §8), copied in the 2026-09-15 snapshot (a local copy kept by the project owner and a copy on the laboratory NAS; locations handed over separately) |
| What is the protocol? | `Training/orchestrator/fixed_question_benchmark.py` + `docs/reference/RAG_FIXED_QUESTION_BENCHMARK.md` + `docs/BENCHMARK.md` §3 |
| What is the state and why? | `docs/archive/sessions/PROJECT_CHECKPOINT.md` (newest first). **Not in Git**: it lives in the original checkout and in both snapshots. Without it you have the state summarised in `README.md` §11 (dated 2026-09-17) and the commit messages, which are detailed |
| Who did what, originally? | `LICENSE`, `git show d268761:README.md` (the original author's README: paper authors, contact, legacy app and CLI documented in full), `README.md` §1.2 |
| What is the mathematics? | `docs/HANDOFF.md` (stable, never fitted to data) |
| What are the hypotheses? | `docs/RESEARCH_BLUEPRINT.md` |
| Which document is current? | `docs/README.md` (the map); anything under `docs/archive/` was correct on its date only |
| Which LLM answers? | `Training/orchestrator/llm_config.py` and the artefact's provider block, never a doc |

Two documents deserve a warning: `docs/WEBAPP_ARCHITECTURE.md` and `docs/LLM_ORCHESTRATION.md`
are RAG documents frozen at an early planning stage (the first still says "not implemented");
they are indexed on purpose and must not be edited. The current description of the application
is `docs/ARCHITECTURE.md` and `WEBAPP_GUIDE.md`.

## When you leave

Add a dated section at the top of `PROJECT_CHECKPOINT.md` with: what was run, what changed,
the status label, the exact next action; update `README.md` §1.4 and §11; commit the code and
the guides (not the session log unless the versioning decision is taken); do not push a
history rewrite; leave the server without a running job of yours.

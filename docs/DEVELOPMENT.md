# DEVELOPMENT — environment, server, tests, conventions

Practical guide for whoever works on this repository. Everything here was verified on
2026-09-15 (see `archive/audits/`). For what the system *is*, read `ARCHITECTURE.md`; for
how to reproduce results, `REPRODUCIBILITY_GUIDE.md`.

## 1. Two machines

| | Local checkout | GPU server |
|---|---|---|
| Path | `~/Projects/embryonicDevelopmentSciMLExtension` | the laboratory's GPU server, under the project account: `~/projects/embryonicDevelopmentSciMLExtension` |
| Git | the real repository (`origin` = private GitHub) | **not a working Git clone** (stuck at the Init commit); it is fed by `rsync` only |
| Data / results | none (`Data/`, `Results/`, `Embeddings/`, `RagIndex/`, `Cache/` are gitignored) | all of them live here, only here (backups: the laboratory NAS) |
| Python | no GPU, compiled wheels need `LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib` on this NixOS machine | `~/miniconda3/envs/embryo_env/bin/python` (torch 2.6 + CUDA), call it by full path: non-interactive SSH does not source conda |
| LLM | none | Ollama system service (`ollama serve`, `http://localhost:11434`), pinned to GPU0 |

Deploy local → server with `./deploy2GPUServ.sh` (rsync `--delete` with explicit excludes for
`Data/ Results/ Embeddings/ Models/ RagIndex/ Cache/` and a `.gitignore` filter). Two consequences
to know: `docs/` is gitignored, therefore **never synced** by the script (the server holds an
older `docs/` tree, which is exactly the RAG corpus it indexed); and always dry-run
(`rsync -n`) before trusting a `--delete` invocation — a wrong exclude destroyed the server's
`Results/` and `Data/` once (2026-07-24). One path per rsync call, never `a/ b/ c/ dest/`
(a multi-source call once flattened `Tests/` on the server).

Long jobs on the server: `screen` (no `tmux`). Never launch anything that touches GPU1–GPU7,
never signal, renice or kill another user's job, and never run a benchmark while a foreign
process sits on GPU0 (see `REPRODUCIBILITY_GUIDE.md` §GPU).

## 2. Environment

```bash
conda create -n embryo_env python=3.10
conda activate embryo_env
pip install -r requirements.txt        # unpinned; lists scikit-learn twice (harmless)
```

`requirements.txt` also brings `chromadb` and `sentence-transformers` (RAG). The embedding
model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is downloaded from
Hugging Face on first use. Ollama and its models are **not** pip dependencies and are never
reinstalled or re-pulled by this project.

## 3. Running the tests

```bash
pytest Tests/                                   # from the repository ROOT, not Training/
node Tests/webapp_api/test_chat_dom_safety.mjs   # the 4 Node tests are not collected by pytest
node Tests/webapp_api/test_patient_selection.mjs
node Tests/webapp_api/test_viewer_navigation.mjs
node Tests/webapp_api/test_fixed_questions.mjs
```

`Tests/conftest.py` puts `Training/` on `sys.path` and truncates `sys.argv` before any import
reaches `ModelBuilder`/`Load_data` (their `ConfigArgs()` runs `argparse` at import time and
would `sys.exit(2)` on pytest's own flags). Suite size on 2026-09-15: 74 files, 1 211 tests.

Expected local result on the NixOS machine: **1 218 passed, 40 skipped, 5 failed** — the five
failures are environment-only (float tolerances in `test_hmm`/`test_semi_hmm`, the
`docs/` inventory drift in `test_rag_inventory`, a real-cache check in `test_app`) and pass on
the server. Skipped tests need the real `Embeddings/`, `Cache/`, `RagIndex/` or `docs/corpus/`.

All tests are synthetic-data / mocked-dependency: no GPU, no vector DB, no live Ollama call.

## 4. Conventions that are load-bearing

- **Composition, never edition.** The SciML extension (`embeddings/ evaluation/ rag/ orchestrator/
  validator/ reporting/ webapp_api/`) imports from the original pipeline (`DataSet.py`,
  `ModelBuilder.py`, …) and from each other, and never edits `Load_data.py`, `config_args.py`,
  `train.py`, `train_val_test_pipline.py` or `DataSet.py`. Later layers compose earlier ones
  (`webapp_api` calls `reporting`/`orchestrator` in-process; `semi_hmm.py` reuses `HMMModel`).
- **The `test` split is locked.** It is used only for pre-registered final reporting. The
  Reporting API, the BFF and the tooling reject `split=test` with 403 at several independent
  layers; keep those guards.
- **Q1–Q15 are frozen** (`Training/orchestrator/fixed_question_benchmark.py`): wording, anchor
  (Patient_319, window 156, split val), scoring criteria. The web app serves them from that
  module and holds no copy.
- **Honesty rules** in every answer surface: never invent a biological fact; always distinguish
  MODEL-DERIVED from OBSERVED; never present an annotated skip as a predicted skip; never state
  a lag when the compared events are not the same event; never name a time unit where it is
  not sourced.
- **No experimental change without a comparison** on the frozen benchmark, and replication
  (n ≥ 3) before believing a ~0.5-point difference: the benchmark's own run-to-run σ is
  0.75/15 with byte-identical contexts.
- **Never delete an artefact.** Superseded documents and invalid runs are kept and marked
  (e.g. `Results/evaluation/validator_experiment/…validator_v1_3.INVALID.txt`).
- **Which LLM answers** is decided in one place, `Training/orchestrator/llm_config.py`
  (`LLM_MODEL`, `LLM_TIMEOUT_SECONDS`). Swapping the model is a benchmark decision, not a config edit.

## 5. Known pitfalls in the original pipeline (unchanged on purpose)

- `ConfigArgs` (`Training/config_args.py`) has a Windows-style default path
  `Configs\config.ini`: on Linux `Configs/config.ini` is **never read**; every value silently
  falls back to the hardcoded defaults (`batch_size=16`, not the `1` in the file).
- `ConfigArgs()` is instantiated at import time inside `Load_data.py`/`ModelBuilder.py` and
  parses the real `sys.argv`; any script importing them with foreign CLI flags dies with
  `argparse` exit 2. `embeddings/build_cache.py` and `Tests/conftest.py` neutralise `sys.argv`
  around the import; do the same in any new script.
- Training commands run **from inside `Training/`** (paths are `../Data`, `../Results`).
- `train_val_test_pipline.py` hardcodes `device="cuda"`.

## 6. Where things are

| What | Where |
|---|---|
| Production entry points | `Training/webapp_api/app.py` (port 8001), `Training/reporting/api.py` (port 8000) |
| Scientific entry points | `Training/experiments/e1_ladder/run_e1_full_analysis.py`, `Training/experiments/gru/run_gru_evaluation.py`, `Training/experiments/hmm_semi_hmm/run_hmm_evaluation.py`, `Training/embeddings/build_cache.py` |
| Benchmark entry points | `Training/orchestrator/run_fixed_question_benchmark.py`, `run_validator_experiment.py`, `capture_run_identity.py`, `summarize_fixed_question_runs.py` |
| Finished-experiment runners | `Training/experiments/{e1_ladder, gru, hmm_semi_hmm, llm_regression, context_ablations, rag_llm_diagnostics, embeddings_exploration, rag_retrieval_smoke}/` — moved out of the packages on 2026-09-15 (imports updated, behaviour unchanged); index in `Training/experiments/README.md` |
| Salvaged one-off scripts | `Training/experiments/archive/` (37 scripts that only existed in the server's `/tmp`; see its README) |
| Documentation map | `docs/README.md` |
| Session history | `docs/archive/sessions/` (the 560 KB `PROJECT_CHECKPOINT.md` is the day-by-day log up to 2026-09-15) |

## 7. Working with an AI coding agent

This extension was built with an AI agent driven by written missions. `CLAUDE.md` at the
repository root is that agent's operating guide (gitignored, local). Keep it consistent with
this file; it must never become the only place a fact lives.

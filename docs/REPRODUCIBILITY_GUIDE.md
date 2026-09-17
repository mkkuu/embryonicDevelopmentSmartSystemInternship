# REPRODUCIBILITY GUIDE — what you need, where it is, how to rebuild, what not to touch

Written 2026-09-15 from verified state. The older `docs/REPRODUCIBILITY.md` (2026-08-24) is
part of the RAG corpus and is kept unchanged; it predates the RAG/LLM/Validator layers and
this guide supersedes it for practical purposes.

## 1. GPU and server rules (non-negotiable)

- The project uses **at most one GPU**, **GPU0**, on the laboratory's shared GPU server. Ollama is
  pinned to GPU0. GPU1–GPU7 belong to other users: never use them, never set
  `CUDA_VISIBLE_DEVICES` to move work there.
- **Never interrupt, signal, renice, suspend or modify an external job.** Read-only inspection
  only (`nvidia-smi -i 0`, `ps`). If a foreign process occupies GPU0, do not run a benchmark:
  Ollama will offload part of the model to CPU (`gpu_fraction < 1.0`), timings and possibly
  outputs change, and the run is not comparable. Wait, and end with the status
  `WAITING_FOR_VALID_GPU`. A degraded run is invalidated, never scored.
- Python-side computations (evaluation harness, RAG, validator) run on CPU:
  `CUDA_VISIBLE_DEVICES="" python -m …`. Only the encoder (`build_cache`), the original
  training and Ollama need the GPU.
- Long runs: `screen` sessions; call the env's python by full path
  (`~/miniconda3/envs/embryo_env/bin/python`).

## 2. Inputs and where they live (server only, all gitignored)

| Artefact | Path | Identity | Backup |
|---|---|---|---|
| Raw dataset | `Data/embryo_dataset_F0`, `_annotations`, `_time_elapsed`, `embryo_dataset_grades.csv` | Zenodo dataset (root README) | NAS `…/Data/` |
| Splits | `Data/Splits/F0.csv` (492 / 106 / 106 videos, seed 42) | produced by `preProcess.py` | NAS |
| Classifier checkpoint | `Results/resnet18_balanced/best_model.pth` | SHA-256 `be5460d0712d47e916589aa0a230a4132ce8f1d76c682b6e66a21dde1ffc5704` | NAS `Results_backup/` (2026-08-16) + snapshot 2026-09-15 |
| Embeddings | `Embeddings/resnet18/{manifest.json, train/, val/, test/}` | manifest records the checkpoint SHA-256 above | NAS `Embeddings_backup/` |
| Frozen Semi-HMM | `Results/evaluation/semi_hmm_weekend_phaseF/model/` | `dmax=268`, negative_binomial, unweighted | snapshot 2026-09-15 |
| Frozen HMM | `Results/evaluation/e1_hmm_k7_sweep/best_model/` | α=2.0, ρ=0.5 | snapshot |
| GRU checkpoints | `Results/evaluation/e1_gru_step6a_full/checkpoint/` | 3 seeds × 2 formulations | snapshot |
| RAG corpus | the 30 `docs/*.md` listed in `Training/rag/inventory.py` | content hashes in `RagIndex/manifest.json` | **versioned in Git since 2026-09-16** (content unchanged) + snapshot; the server holds the same 30 files in its older flat `docs/` |
| Istanbul corpus | `docs/corpus/ISTANBUL_CONSENSUS_2025.md` (+ `_TRACEABILITY.md`) | SHA-256 `9670adba…` (2026-09-11) | versioned in Git since 2026-09-16 + snapshot |
| Vector index | `RagIndex/` (Chroma, `docs_v1`) | manifest 2026-08-25 | regenerable |
| Inference cache | `Cache/reporting/semi_hmm/<config>/<split>/<video>/inference.json` | — | regenerable |
| Benchmark artefacts | `Results/evaluation/{validator_experiment, compact_context_experiment, observed_only_experiment, prediction_only_experiment, event_anomaly_rag_inventory, rag_llm_quality, llm_benchmark_*}` | SHA-256 in `BENCHMARK.md` §8 | snapshot |

Snapshot of 2026-09-15: a local copy kept by the project owner (full `Results/`, the 38 `/tmp`
scripts, docs, manifests) and a copy on the laboratory NAS (tars + `SHA256SUMS`); the exact
locations are handed over separately. Git restoration point: tag `pre-cleaning-2026-09-15` = commit `a586c20`.

**Lost and not recoverable**: the original `Results/resnet18/` checkpoint (destroyed
2026-07-24); the script that produced `Results/evaluation/gru_identity_threshold_analysis/`
(not found anywhere; the report is kept).

## 3. Rebuild order (validated commands)

All commands from inside `Training/` unless stated; `PY` = the conda env's python.

```bash
# 0. Data preparation (one-off, expects the raw dataset under ../Data)
python preProcess.py

# 1. Classifier (50 epochs, GPU). Output: ../Results/resnet18_balanced/best_model.pth
python train_balanced.py --model_name resnet18 --batch_size 16 --epochs 50 --learning_rate 0.0001

# 2. Embedding cache (once per split; explicit parameters on purpose, config.ini is not read)
python -m embeddings.build_cache --model_name resnet18 --checkpoint ../Results/resnet18_balanced/best_model.pth \
       --window_size 8 --stride 1 --focal_type F0 --image_size 224 --split Train --output_root ../Embeddings
python -m embeddings.build_cache ... --split Val      # and Test only for the locked, pre-registered uses
python -m embeddings.validate

# 3. Scientific results (CPU)
CUDA_VISIBLE_DEVICES="" python -m experiments.e1_ladder.run_e1_full_analysis        # E1 ladder, Test, bootstrap n=1000 → REPORT.md
CUDA_VISIBLE_DEVICES="" python -m experiments.e1_ladder.run_frame_shuffle_sanity_check
CUDA_VISIBLE_DEVICES="" python -m experiments.gru.run_gru_evaluation --formulation classification --seed 0   # (and forecast; seeds 0 1 2)
CUDA_VISIBLE_DEVICES="" python -m experiments.hmm_semi_hmm.run_hmm_evaluation           # k=7 sweep → e1_hmm_k7_sweep
# Semi-HMM reference: Training/experiments/archive/semi_hmm_construction/phase_f_full_train_val.py (run from Training/, verbatim copy)

# 4. RAG index (CPU; downloads the embedding model on first run)
CUDA_VISIBLE_DEVICES="" python -m rag.ingest
CUDA_VISIBLE_DEVICES="" python -m rag.retrieval "Qu'est-ce que le Semi-HMM ?" --top-k 5

# 5. Serving
python -m reporting.warm_cache --split val
python -m reporting.api          # port 8000, optional
python -m webapp_api.app         # port 8001

# 6. Benchmark (GPU0 must be free of foreign processes; see BENCHMARK.md §3)
python -m orchestrator.capture_run_identity --label <label>
python -m orchestrator.run_fixed_question_benchmark --model mistral-nemo:12b --timeout 240 ...
python -m orchestrator.run_validator_experiment --model mistral-nemo:12b --timeout 240 --replications 3 --run-label <label>
python -m orchestrator.summarize_fixed_question_runs
```

Tests (repository root): `pytest Tests/` + the four `node Tests/webapp_api/*.mjs`
(`DEVELOPMENT.md` §3).

## 4. Reference results you should be able to reproduce exactly

| Result | Expected value | Where it is checked |
|---|---|---|
| E1 ladder, Test, identity AUROC | 0.7384 [0.7189, 0.7563] | `e1_full_analysis/REPORT.md` |
| HMM k=7 reference | AUROC 0.63735, Brier 0.14072 | `e1_hmm_k7_sweep/model_selection.json` |
| Semi-HMM reference, Val | AUROC 0.64271, Brier 0.14029, ECE 0.12307 | `semi_hmm_weekend_phaseF/phaseF_report.json` |
| Semi-HMM served config | n_states 15, dmax 268, negative_binomial, unweighted | `reporting/model_loader.py` refuses anything else |
| Q1–Q15 contexts | `document_context_sha256` / `scientific_context_sha256` identical between the v1.2 run and a preflight | `capture_run_identity`, preflight script in `experiments/archive/validator_benchmark/` |
| Validator v1.2 replay | 45/45 verdicts reproduced bit-for-bit on the stored OFF/ON answers | `Tests/validator/test_validator_v13_precision.py` (frozen cases) |

Model outputs of the LLM are **not** reproducible run to run (σ = 0.75/15 with identical
contexts); only the contexts, the grounding and the validator verdicts are deterministic.

## 5. Things that will silently break reproducibility

- Editing, renaming or moving any of the 30 RAG documents or `docs/corpus/`: the index and the
  curated context change; every benchmark artefact records `istanbul_corpus_sha256` and the
  RAG identity, so later runs stop being comparable.
- Syncing a different `docs/` tree to the server with `deploy2GPUServ.sh`. Today the server's
  corpus and index are the 2026-08-25 ones and its code tree predates the 2026-09-15
  restructure; whether the script's `.gitignore` filter now syncs the versioned parts of `docs/`
  has not been verified (`handover/GPU_SERVER.md` §6). Re-ingest deliberately, then re-run the
  benchmark, never as a side effect.
- Moving modules under `Training/orchestrator/`, `validator/`, `rag/` before the pending
  Validator v1.3 rebenchmark: `capture_run_identity` hashes them by path.
- Letting `Ollama` change: the model digest (`mistral-nemo:12b` = `e7e06d107c6c…`, Ollama
  0.7.0) is part of every run identity.
- Using `Configs/config.ini` to "set" parameters: it is not read on Linux.
- Running with a foreign process on GPU0 (CPU offload).
- Any use of the `test` split outside the two pre-registered reports.

## 6. Conventions for new results

Pre-register the comparison, run n ≥ 3 replications, keep the raw artefact untouched, write the
analysis next to it, record the run identity, and never delete or overwrite a previous
artefact — mark it invalid with a sidecar file instead (as `…validator_v1_3.INVALID.txt`).

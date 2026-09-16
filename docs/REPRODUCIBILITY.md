# Reproducibility

How to reproduce or extend the SciML extension's experiments. Written so a new person can get oriented without re-deriving anything already established. Companion docs: `docs/SCIENTIFIC_RESULTS.md` (what was found), `docs/MODEL_COMPARISON.md` (per-model detail), `CLAUDE.md` (repo-wide conventions, checked into git).

## Environment

```bash
conda create -n embryo_env python=3.10
conda activate embryo_env
pip install -r requirements.txt
```

Local checkout: no `torch`/CUDA available (per `CLAUDE.md`) — all real training/embedding-extraction/HMM-fitting work happens on the remote GPU box. `Tests/` are synthetic-data-only and designed to run from the **repo root** via `pytest Tests/` — but as of 2026-08-24 this specific local checkout has **no conda/pytest installed at all** (verified: `which conda` and `python3 -m pytest` both fail). The test suite has only actually been run and verified on the remote box (`229/229 passed`, 2026-08-24) — do not assume the local checkout can run it without first setting up the conda env per the block below.

Remote GPU box: `rdelau27@gpu1.labisen.isen-ouest.fr` (4x NVIDIA Titan V, 12GB each), key-based SSH, no password. Conda env at `~/miniconda3/envs/embryo_env` — **not sourced by non-interactive SSH** (`.bashrc` skipped); always call the env's python directly: `~/miniconda3/envs/embryo_env/bin/python`. Confirmed: torch 2.6.0+cu124, CUDA available, pytest 9.1.1. Repo synced there via `./deploy2GPUServ.sh` (rsync `--delete`, with explicit `--exclude` for `Data/`/`Results/`/`Embeddings/` — **always `rsync --dry-run` first** before trusting this against remote state that matters; a 2026-07-24 incident destroyed the remote `Results/`/`Data/` before this exclude fix existed). Long-running jobs: `tmux` is not installed; use `screen` (survives SSH disconnect).

## Dependencies

`requirements.txt` (repo root, no version pinning, `scikit-learn` listed twice — harmless, unfixed): pandas, scikit-learn, tqdm, torch, torchvision, timm, transformers, matplotlib, seaborn, python-dotenv, flask, psycopg2-binary, umap-learn, scipy, pytest.

## Important paths

| Path | Location | Contents |
|---|---|---|
| `Training/` | git | Original pipeline + SciML extension (`embeddings/`, `evaluation/`) |
| `Tests/` | git | Synthetic-data unit tests, run from repo root |
| `docs/` | local only, gitignored | All planning/checkpoint/results docs, including this one |
| `Configs/config.ini` | git | Shared config (has a known-unresolved path bug on Linux, see `CLAUDE.md`) |
| `Data/` | server only, gitignored | Raw dataset |
| `Embeddings/` | server only, gitignored | Cached ResNet18 embeddings, one dir per split |
| `Results/` | server only, gitignored | All classifier checkpoints + every `evaluation/*` experiment's output |

Everything under `Training/evaluation/` and `Training/embeddings/` resolves paths relative to `Training/` as cwd (matching the rest of `Training/`'s convention) — always `cd Training` first.

## Datasets and embeddings

Raw data under `Data/embryo_dataset_F0` (+ `embryo_dataset_annotations`, `embryo_dataset_time_elapsed`, `embryo_dataset_grades.csv`), preprocessed once via `Training/preProcess.py` into `Data/Splits/F0.csv`. A full backup of `Data/` exists at `/media/nas/rdelau/embryonicDevelopmentSciMLExtension/Data/` on the lab NAS (used once already to restore after the 2026-07-24 incident).

Cached embeddings: `Embeddings/resnet18/{train,val,test}/`, built via:

```bash
cd Training
python -m embeddings.build_cache --model_name resnet18 \
  --checkpoint ../Results/resnet18_balanced/best_model.pth \
  --window_size 8 --stride 1 --focal_type F0 --image_size 224 \
  --split Train --output_root ../Embeddings   # repeat per split (Train/Val/Test)
python -m embeddings.validate
```

**Checkpoint provenance (critical, verify before assuming reproducibility)**: `Embeddings/resnet18/{train,val}/manifest.json` records `checkpoint_path=../Results/resnet18_balanced/best_model.pth`, `checkpoint_sha256=be5460d0712d47e916589aa0a230a4132ce8f1d76c682b6e66a21dde1ffc5704`. **This is the sole surviving classifier checkpoint** — the original `Results/resnet18/best_model.pth` (from the project's first 50-epoch training run) was destroyed in the 2026-07-24 incident and was never recovered (no backup was ever found for `Results/`, unlike `Data/`). If `Results/resnet18_balanced/best_model.pth` is ever lost, the currently-cached embeddings become **the only way** to reproduce anything downstream — they cannot be regenerated from scratch without either that exact checkpoint or accepting a different (non-reproducing) embedding space from a freshly retrained classifier.

Splits (`Embryo_Transition_Dataset`, window-consistency-filtered): Train 492 videos/196,934 windows, Val 106 videos/43,339 windows, Test 106 videos/42,519 windows. `phase_to_index` confirmed identical across all three splits.

## Seeds

- E1 (`run_e1_full_analysis.py`): bootstrap seed=0, n=1000 resamples.
- GRU: trained with `--seed 0/1/2` (3 independent seeds); frame-shuffle control uses seed 0 only.
- HMM/Semi-HMM: no stochastic fitting (closed-form counts + deterministic `lbfgs` LogisticRegression given fixed data/hyperparameters) — bootstrap comparison scripts use fixed seeds per comparison (documented inline in each script, e.g. `emission_balanced_experiment.py` uses seed=100/200/300 for its three pairwise bootstrap comparisons).

## Test discipline

**TEST = LOCKED** by default for every experiment in `Training/evaluation/`. The only pre-registered, explicitly authorized exception in the project's history is the GRU branch's `run_gru_test_evaluation.py` / `run_gru_frame_shuffle_sanity_check.py` (§4 of `SCIENTIFIC_RESULTS.md`) — a one-time, deliberate, final confirmatory step. Before trusting any new script that touches `Embeddings/resnet18/test/`: confirm the user explicitly authorized it for that specific script, not by analogy to GRU.

To verify Test was never touched by a given experiment: `grep -rn "test" <script>.py` (check for `Embeddings/resnet18/test` or `split="Test"` references) and check `Embeddings/resnet18/test/`'s mtime/atime against the experiment's run window — a real mtime change is unambiguous; an atime bump can be innocuous (a prior session's `ls`/`stat` audit) and should be cross-checked against the script's own code, not trusted alone.

## Main scripts (SciML extension)

| Script | Purpose |
|---|---|
| `Training/embeddings/build_cache.py` | Extract + cache embeddings per split |
| `Training/embeddings/validate.py` | Sanity-check a built cache |
| `Training/evaluation/run_e1_full_analysis.py` | E1 ladder, full bootstrap analysis (current correct entry point; `run_e1_baselines.py` is an older, statistically-degenerate seed-loop version — do not use as default) |
| `Training/evaluation/run_frame_shuffle_sanity_check.py` | E1's frame-shuffle validity control |
| `Training/evaluation/run_gru_evaluation.py` / `run_gru_test_evaluation.py` / `run_gru_frame_shuffle_sanity_check.py` | GRU train/val, Test (authorized exception), frame-shuffle control |
| `Training/evaluation/run_hmm_evaluation.py` | HMM alpha/rho sweep |
| `Training/evaluation/run_hmm_k_step_alignment_check.py` | k=1 vs k=7 alignment diagnostic |
| `Training/evaluation/models/{base_rate,identity_dynamics,persistence,linear_ssm,gru,hmm,semi_hmm}.py` | The `Model` implementations themselves |

Emission-rebalancing experiments (`emission_balanced`, `emission_weighted_sweep`) were run from ad hoc, deliberately **uncommitted** scripts placed at `/tmp/*.py` on the server (not part of the git-tracked `Training/evaluation/` package) — see `docs/CLEANUP_PLAN.md` for the decision on whether/where to promote these into the tracked codebase.

## Models / checkpoints (frozen references — do not refit)

| Path (server, under `Results/evaluation/`) | Model |
|---|---|
| `e1_hmm_k7_sweep/best_model/` | HMM, alpha=2.0/rho=0.5, k=7 — **the** HMM reference |
| `semi_hmm_weekend_phaseF/model/` | Semi-HMM, dmax=268, negative_binomial — **the** Semi-HMM reference |
| `emission_balanced/{hmm_balanced_model,semi_hmm_balanced_model}/` | `class_weight='balanced'` variants (Case D, not adopted) |
| `emission_weighted_sweep/{hmm_alpha_0p5,hmm_alpha_0p75,semi_hmm_alpha_0p5,semi_hmm_alpha_0p75}/` | Moderated reweighting variants (all closed, not adopted) |
| `emission_alpha025/{hmm_alpha025,semi_hmm_alpha025}/` | Closing experiment for the best emission-ECE candidate (closed, not adopted) — class-reweighting branch is fully closed as of this model |
| `Results/resnet18_balanced/best_model.pth` | Classifier checkpoint behind every cached embedding — see provenance note above |

## Reproducing the main results

```bash
cd Training
python -m evaluation.run_e1_full_analysis                    # E1 ladder
python -m evaluation.run_frame_shuffle_sanity_check           # E1 validity control
python evaluation/run_gru_evaluation.py --seed 0 ...           # GRU Train/Val (see script --help)
python evaluation/run_hmm_evaluation.py ...                    # HMM alpha/rho sweep
```

Emission-rebalancing experiments are not currently runnable via a single tracked command (ad hoc `/tmp` scripts, see above) — read `docs/HANDOFF_EMISSION_BALANCED.md` / the equivalent for the weighted sweep for the exact protocol if reproducing them.

## Known bugs relevant to reproducibility

- `ConfigArgs`'s default `config_path` uses a Windows-style backslash — never resolves on Linux, so `Configs/config.ini` edits silently don't take effect through any default `ConfigArgs()` call (`CLAUDE.md` has the full detail). Irrelevant to `Training/evaluation/`/`Training/embeddings/`, which deliberately never import `ConfigArgs`/`Load_data.py`/`ModelBuilder.py`.
- `ConfigArgs.__init__` runs its own `argparse` against the real `sys.argv` — any script importing `ModelBuilder`/`Load_data` while `sys.argv` holds unrecognized tokens (pytest's own CLI args, etc.) dies mid-import. `Tests/conftest.py` and `embeddings/build_cache.py` work around this by truncating `sys.argv`; apply the same pattern in any new script importing those modules outside `train.py`'s own CLI context.

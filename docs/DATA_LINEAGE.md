# Data Lineage

Full pipeline, RAW DATA → RAG/Web App, as it actually exists today (not aspirational). Companion to `docs/REPRODUCIBILITY.md` (how to run each step) and `docs/INFERENCE_SCHEMA.md` (the output shape at the end of this chain).

```
RAW DATA
  ↓  preprocessing (Training/preProcess.py)
SPLITS (Data/Splits/F0.csv)
  ↓  frame loading (Embryo_Transition_Dataset, DataSet.py)
FRAMES + WINDOWS
  ↓  classifier training (Training/train_balanced.py) → Results/resnet18_balanced/best_model.pth
  ↓  embedding extraction (Training/embeddings/build_cache.py)
EMBEDDINGS (Embeddings/resnet18/{train,val,test}/)
  ↓  trajectory grouping (Training/evaluation/trajectory.py)
METADATA (manifest.json, metadata.csv, metadata_with_time.csv)
  ↓  model fit (Training/evaluation/models/{hmm,semi_hmm}.py)
MODEL (Results/evaluation/semi_hmm_weekend_phaseF/model/, frozen)
  ↓  inference (HMMModel.explain(), SemiHMMModel.next_phase_distribution() etc.)
PREDICTION (per-window posterior, next-phase distribution)
  ↓  chain construction (SemiHMMModel.transition_chain(), duration_distribution())
PROBABILISTIC CHAIN
  ↓  [NOT BUILT]
REPORTING
  ↓  [NOT BUILT]
RAG / WEB APP
```

## Step detail

| Step | Input | Output | Script | Format | Version | Reproducibility |
|---|---|---|---|---|---|---|
| Raw data | Zenodo DOI dataset | `Data/embryo_dataset_F0`, `embryo_dataset_annotations`, `embryo_dataset_time_elapsed`, `embryo_dataset_grades.csv` | manual download | images + CSV | fixed (dataset release) | Full backup exists on lab NAS (`/media/nas/rdelau/embryonicDevelopmentSciMLExtension/Data/`) |
| Preprocessing | Raw data above | `Data/Splits/F0.csv` | `Training/preProcess.py` | CSV | one-time, run once, not versioned beyond the file itself | Deterministic given the same raw data; **not re-run since original execution** |
| Frame/window loading | `Data/Splits/F0.csv` + raw images | in-memory `Embryo_Transition_Dataset` windows (not persisted separately from embeddings) | `Training/DataSet.py` (`Embryo_Transition_Dataset`) | Python objects | tied to `window_size=8, stride=1` | Deterministic given fixed CSV + `window_size`/`stride` |
| Classifier training (imbalance-corrected) | Windows above | `Results/resnet18_balanced/best_model.pth` | `Training/train_balanced.py` | PyTorch `state_dict()` | trained 2026-07-24 | **Not exactly reproducible** without the original random seed/run — this specific checkpoint is the sole surviving artifact (original `Results/resnet18/` lost) |
| Embedding extraction | `Results/resnet18_balanced/best_model.pth` + windows | `Embeddings/resnet18/{train,val,test}/embeddings.pt` + `manifest.json` + `metadata.csv` | `Training/embeddings/build_cache.py` | `.pt` tensor + JSON/CSV | `extractor_version: "1.0"` (in manifest) | Deterministic given the checkpoint (SHA-256 verified) + fixed `--window_size 8 --stride 1 --focal_type F0 --image_size 224` |
| Trajectory grouping | Cached embeddings | `List[Trajectory]` (in-memory, per evaluation run) | `Training/evaluation/trajectory.py` (`group_into_trajectories`) | Python objects | — | Deterministic, pure function of the cache |
| Model fit | Train trajectories | `Results/evaluation/semi_hmm_weekend_phaseF/model/state.json` (+ nested `emission_hmm/state.json`) | `Training/evaluation/models/semi_hmm.py` (`SemiHMMModel.fit`) | JSON (never pickled) | `dmax=268, duration_family=negative_binomial, logreg_class_weight=None`, recorded in `state.json` | Deterministic (closed-form counts + deterministic `lbfgs` LogisticRegression given fixed data/hyperparameters) — independently bit-exact-reproducibility-checked for the sibling HMM model (`e1_hmm_step1_reproducibility_check/`) |
| Inference | Val (or any) trajectories + loaded model | Per-window posteriors, `TrajectoryPrediction` | `HMMModel.explain()`, `.predict_k_step()`; `SemiHMMModel.next_phase_distribution()` etc. | Python dict / `np.ndarray` / dataclass | tied to the loaded model's `state.json` | Deterministic given the model + input trajectories |
| Chain construction | A trajectory + loaded Semi-HMM | `List[Dict]` (`transition_chain`), duration distributions | `SemiHMMModel.transition_chain()`, `.duration_distribution()` | Python dict | — | Deterministic; **retrospective only** (needs full trajectory) |
| Reporting | Chain + predictions | — | **not built** | — | — | — |
| RAG / Web App | Reporting layer | — | **not built** | — | — | — |

## Gaps in the chain (relevant to the next implementation phase)

- No persisted, canonical "prediction record" format exists between "inference" and "chain construction" — each script computes what it needs ad hoc. `docs/INFERENCE_SCHEMA.md` proposes the target shape.
- No serving/API layer exists anywhere after model inference — everything today runs as one-off Python scripts against a local trajectory list, not a queryable service.
- `Data/embryo_dataset_time_elapsed/` is disconnected from the embeddings/model lineage — it was joined once, additively, only inside the HMM branch's own diagnostic work (`metadata_with_time.csv`), never into the canonical `metadata.csv` or any model's own pipeline.
- The classifier-training step (`Results/resnet18_balanced/best_model.pth`) is a single point of failure for the entire downstream chain — see `docs/REPRODUCIBILITY.md`'s provenance note.

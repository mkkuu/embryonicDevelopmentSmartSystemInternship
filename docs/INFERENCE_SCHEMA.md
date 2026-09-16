# Inference Schema

Canonical shape for one inference record, derived from the **actual current outputs** of `HMMModel`/`SemiHMMModel` (`Training/evaluation/models/{hmm,semi_hmm}.py`) and `Trajectory`/`TrajectoryPrediction` (`Training/evaluation/trajectory.py`) — inspected directly, not assumed. Nothing here is implemented as a serialization layer yet; this is the target shape the future reporting/web-app/RAG code should converge on, built to match what the models can already produce rather than imposed on top of them.

## What the code actually produces today

`HMMModel.explain(traj, mode)` already returns a list of per-window records (one per `Trajectory` window) shaped almost exactly like the target schema:

```python
{
    "video": str,                          # traj.video_name
    "window": int,                         # traj.window_starts[t]
    "phase": str,                          # argmax state name
    "phase_posterior": {phase_name: float, ...},   # 15 entries, sums to 1
    "next_phase_probability": {phase_name: float, ...},  # 15 entries, causal predict_next()
    "entropy": float,                      # Shannon entropy (nats) of phase_posterior
    "inference_mode": "filtering" | "smoothing",
}
```

`SemiHMMModel` does not have an equivalent single `explain()` — its capabilities are separate methods:
- `next_phase_distribution(traj)` → `np.ndarray` shape `(T, 15)`, duration-aware one-step-ahead distribution.
- `duration_distribution(state)` → `Dict[int, float]`, `d=1..dmax` (268) → `P(D=d|state)`.
- `expected_duration(state)` → `float`.
- `quantiles(qs)` (on the duration model object) → `Dict[float, int]`, e.g. `{0.5: 12, 0.9: 40}`.
- `transition_chain(traj)` → `List[Dict]`, one entry per **ground-truth ​segment** (not per window): `{"phase": str, "observed_duration_windows": int, "duration_probability_at_observed": float, "next_phase_distribution": {phase_name: float, ...}}`.
- `predict(trajectories)` / `predict_k_step(trajectories, k)` → `TrajectoryPrediction(video_name, window_starts, consistency_flag_prob)` — a single scalar per window, the binary "did/will a transition happen" probability, not a full phase distribution.

`Trajectory` (ground truth, when available): `video_name`, `window_starts: List[int]`, `embeddings: (T,512)`, `consistency_flag: (T,)`, `first_frame_phase: (T,)`, `last_frame_phase: (T,)`.

## Canonical schema (target)

```json
{
  "sample_id": "video_name + window_start, e.g. Patient_319#54",
  "current_phase": "phase_posterior argmax, e.g. t3",
  "phase_probability": {"tPB2": 0.01, "...": "...", "tEB": 0.00},
  "next_phase_distribution": {"t3": 0.02, "t4": 0.85, "...": "..."},
  "transition_chain": [
    {"phase": "t3", "observed_duration_windows": 4, "duration_probability_at_observed": 0.062,
     "next_phase_distribution": {"t4": 0.9, "...": "..."}}
  ],
  "duration_distribution": {"1": 0.003, "2": 0.006, "...": "...", "268": 0.0001},
  "expected_duration": 12.4,
  "uncertainty": 1.83,
  "timestamp": null,
  "frame_index": 54,
  "model_version": "semi_hmm_weekend_phaseF (dmax=268, negative_binomial, alpha=2.0/rho=0.5, class_weight=None)"
}
```

## Field-by-field: REQUIRED / OPTIONAL / DERIVED / UNAVAILABLE

| Field | Status | Source | Notes |
|---|---|---|---|
| `sample_id` | REQUIRED | derived from `Trajectory.video_name` + `window_starts[t]` | Not a native field anywhere — must be constructed by the reporting layer |
| `current_phase` | REQUIRED | `HMMModel.explain()`'s `"phase"` | argmax of `phase_posterior`; also derivable from Semi-HMM if a similar `explain()` is added (not implemented today) |
| `phase_probability` | REQUIRED | `HMMModel.explain()`'s `"phase_posterior"` | Native today only via the composed `HMMModel` (`filtering`/`smoothing`); Semi-HMM does not expose an equivalent per-window full-state posterior directly — it reuses the same emission but has no `explain()`-equivalent method yet |
| `next_phase_distribution` | REQUIRED | `HMMModel.explain()`'s `"next_phase_probability"` (causal, one-step) OR `SemiHMMModel.next_phase_distribution(traj)` (duration-aware, one-step) | Two different, non-identical sources — pick one explicitly and document which; do not silently mix |
| `transition_chain` | OPTIONAL | `SemiHMMModel.transition_chain(traj)` | **Segment-level, not window-level** — a list, not a per-window field; only meaningful for a completed/observed trajectory, not a live in-progress one; explicitly documented as a diagnostic over ground-truth segments, not a forward multi-step forecast |
| `duration_distribution` | OPTIONAL | `SemiHMMModel.duration_distribution(state)` | Per-*phase*, not per-window/sample — 268 entries, must be summarized (CDF/quantiles) before display, never shown as raw pmf (`docs/SCIENTIFIC_REPORT.md` §18). **UNAVAILABLE-as-informative for tPB2/tEB** (forced uniform fallback, must be flagged, not hidden) |
| `expected_duration` | OPTIONAL | `SemiHMMModel.expected_duration(state)` | Same per-phase, same tPB2/tEB caveat |
| `uncertainty` | DERIVED | `HMMModel.entropy(phase_posterior)` (Shannon entropy, nats) | No native equivalent from Semi-HMM's own outputs; if using Semi-HMM's `phase_posterior`, would need to be computed the same way once such a posterior exists for it |
| `timestamp` | UNAVAILABLE | — | `Data/embryo_dataset_time_elapsed/` exists in the raw dataset but is never joined into the cached embeddings/metadata pipeline used by evaluation (only the HMM branch's own additive `metadata_with_time.csv` has it, and only in window-index-adjacent form, not wall-clock) — real timestamps are not currently plumbed through to any model output |
| `frame_index` | REQUIRED | `Trajectory.window_starts[t]` / `TrajectoryPrediction.window_starts[t]` | Directly available, index into the split's flattened window list — NOT a per-video frame number without reconstructing via the dataset (see `docs/HANDOFF.md`/HMM branch's Phase 0 notes on `window_start` semantics) |
| `model_version` | REQUIRED | not a native field — must be constructed from the loaded model's saved `state.json` (`transition_smoothing_alpha`, `transition_decay_rho`, `dmax`, `duration_family`, `logreg_class_weight`, etc.) | Every `save()`/`load()` round-trip already preserves these hyperparameters; a version string is a trivial format layer on top, not yet built |
| ground truth (`consistency_flag`, `first_frame_phase`, `last_frame_phase`) | OPTIONAL, for display only | `Trajectory` | Only available for videos that have annotations (i.e. not a genuinely new/unlabeled inference case) — the web app's "real phase if available" requirement maps directly here |

## What this means for the web app / RAG

- A single, unified per-window record combining **all** the fields above does not exist in the code today — it requires a thin new serialization function that calls `HMMModel.explain()` for the posterior/next-phase fields and `SemiHMMModel`'s separate methods for duration/chain fields, since Semi-HMM's phase posterior itself is not yet exposed the same way HMM's is.
- `transition_chain` is fundamentally different in shape (per-segment) from everything else (per-window) — the reporting layer must not flatten it into the same array without clearly separating the two granularities.
- `timestamp` is the one field with no real source anywhere in the current pipeline — any UI requiring wall-clock time needs either a new join to `Data/embryo_dataset_time_elapsed/` (never done at the embeddings/model level) or must display window-index/frame-index only.

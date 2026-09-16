# Web App Data Requirements

Data needed for the target two-pane interface (timelapse viewer + probabilistic analysis), mapped against `docs/INFERENCE_SCHEMA.md` and the actual current pipeline. **Not implemented here** — this is the requirements list only, for the future web app work.

## LEFT — Timelapse viewer

| Data | Source | Current format | Transformation needed | Update frequency | Available now? | To generate later? |
|---|---|---|---|---|---|---|
| Sample/video identifier | `Trajectory.video_name` | string | none | Static per session | Yes | — |
| Frames | Raw dataset images (`Data/embryo_dataset_F0/<video>/`) | image files on disk (server-side, never exposed via any current API) | Needs a serving layer (static file server or API endpoint) — none exists today | Static per video | Yes, on disk (server only) | Yes — serving layer |
| Timelapse sequencing | Frame filenames + `window_starts` ordering | implicit in filename/frame order | Needs a frame-index ↔ display-order mapping | Static | Partial — frame order exists, no explicit UI-ready index | Yes |
| Current frame / window pointer | `TrajectoryPrediction.window_starts[t]` | int, index into the split's flattened window list (**not** a raw per-video frame number without dataset reconstruction — see `docs/INFERENCE_SCHEMA.md`) | Needs conversion to an actual frame number via `Embryo_Transition_Dataset` reconstruction (same mechanism `embeddings/validate.py` already uses) | Per frame, live | Partial — mechanism exists, not wired to a UI-facing endpoint | Yes |
| Timestamp | — | **UNAVAILABLE** (see `INFERENCE_SCHEMA.md`) | Would need a new join to `Data/embryo_dataset_time_elapsed/`, never done at the model-output level | — | No | Yes, real work |
| Real/annotated phase (ground truth) | `Trajectory.first_frame_phase`/`last_frame_phase` | int index into `DEFAULT_PHASE_NAMES` (15 phases) | Map index→name (`DEFAULT_PHASE_NAMES`, already defined in `hmm.py`) | Static per video | Yes, for annotated videos only | — |

## RIGHT — Probabilistic analysis

| Data | Source | Current format | Transformation needed | Update frequency | Available now? | To generate later? |
|---|---|---|---|---|---|---|
| Current phase | `HMMModel.explain()`'s `"phase"` | string (argmax of `phase_posterior`) | none | Per frame/window, live | Yes (via HMM's composed emission) | — |
| Probabilities per phase | `HMMModel.explain()`'s `"phase_posterior"` | `Dict[str,float]`, 15 entries | none for display; needs sorting/thresholding for a compact UI widget | Per frame/window | Yes | — |
| Next phase (most probable) | argmax of `next_phase_probability` (HMM, causal) or `SemiHMMModel.next_phase_distribution` (duration-aware) | `Dict[str,float]` | argmax + confidence display | Per frame/window | Yes — two non-identical sources, pick and document one | — |
| Transition distribution | `HMMModel.explain()`'s `"next_phase_probability"` or `SemiHMMModel.next_phase_distribution` | `Dict[str,float]`, 15 entries | Bar chart / ranked list | Per frame/window | Yes | — |
| Expected duration | `SemiHMMModel.expected_duration(state)` | float (windows, not clinical time) | Windows→approximate real time needs the same unresolved `time_elapsed` join as `timestamp` above | Per phase (not per frame — a property of the *current phase*, recomputed when phase changes) | Yes in window units; **no** in real time units | Yes, for real-time units |
| Interval / uncertainty | `SemiHMMModel.quantiles(qs)` on the relevant duration model, or `HMMModel.entropy(phase_posterior)` for phase-level uncertainty | `Dict[float,int]` (quantiles) or `float` (entropy, nats) | Present duration as a CDF/quantile range ("70% chance of resolving within N windows"), **never** raw pmf (`SCIENTIFIC_REPORT.md` §18) | Per phase / per frame | Yes | — |
| Probabilistic chain | `SemiHMMModel.transition_chain(traj)` | `List[Dict]`, **per ground-truth segment**, not per frame | Must be displayed as a distinct, segment-granularity view, not merged into the per-frame timeline | Per completed segment (retrospective — needs the full trajectory, not causal) | Yes, for **annotated/completed** videos only — not for a live in-progress one | For a causal, in-progress version: new work |
| tPB2/tEB duration flag | duration model's `fallback_uniform` attribute | bool | Must gate the UI: show "not estimable (structural censoring)" instead of a number | Per phase | Yes (the flag exists) — UI wiring needed | UI wiring |
| Textual justification | — | **UNAVAILABLE** | This is exactly the future RAG's job (`docs/RAG_DATA_MODEL.md`) — do not fabricate ad hoc text in the web app itself | — | No | RAG, separate workstream |
| Model version / provenance | Loaded model's `state.json` | dict of hyperparameters | Format into a short display string | Static per deployed model | Yes (data exists), format layer needed | Small |

## Cross-cutting notes

- **Causal vs. retrospective must never be silently mixed in the UI.** `filtering`/`next_phase_probability`/`predict_k_step` are causal (valid for a live/in-progress sample); `smoothing`/`transition_chain` require the full trajectory and are retrospective-only. A "live" mode of the web app must use only the causal set.
- **HMM vs. Semi-HMM outputs are not interchangeable** — Semi-HMM has no `explain()`-equivalent single per-window record yet (see `INFERENCE_SCHEMA.md`); a real implementation needs either a new `SemiHMMModel.explain()` method or a composition layer in the reporting API, not built yet.
- **Every probability/duration figure must be traceable to `docs/SCIENTIFIC_REPORT.md`'s documented limitations** (emission bottleneck on t3/t5/t6/tPNf, tPB2/tEB censoring, Val-only validation) — the web app should not present numbers as more reliable than the underlying science supports. This is a design constraint on copy/UI treatment, not a data requirement, but it belongs here because it should shape which fields get prominent placement vs. a "low confidence" treatment.

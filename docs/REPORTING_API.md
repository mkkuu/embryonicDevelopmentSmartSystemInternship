# Reporting API

Documents the Reporting API built this project (`Training/reporting/`) — the service layer between the frozen Semi-HMM and any future consumer (Web App, RAG). This file was overdue from that build session; written now, from the actual code, not from memory of intent.

## Architecture

```
Training/reporting/
├── __init__.py
├── schemas.py            typed, JSON-serializable dataclasses (InferenceRecord, TrajectorySummary, ...)
├── model_loader.py        loads + validates the frozen Semi-HMM once, never refits
├── trajectory_service.py  embeddings/metadata access, hard TEST-LOCK guard on every function
├── inference_service.py   composes model_loader + trajectory_service + SemiHMMModel's own
│                          methods into canonical records; per-video forward-pass caching
├── cache.py               disk-backed cache for the expensive forward pass (Phase 1.5,
│                          docs/REPORTING_CACHE.md) -- 38,374x speedup on repeated requests
├── warm_cache.py          optional pre-warm driver, reuses cache.py, not run automatically
└── api.py                 Flask app exposing the HTTP endpoints
```

Placed under `Training/reporting/` (not a new top-level `Reporting/`) — consistent with the existing `Training/embeddings/`/`Training/evaluation/` convention; this is one more layer of the same additive SciML extension. Deliberately **not** integrated into `WebApplication/`'s existing Flask app (a different concern — doctor/admin CRUD, PostgreSQL-backed auth) — the Reporting API is a separate, stateless, read-only service over frozen scientific artifacts.

**Model served**: Semi-HMM k=7, `Results/evaluation/semi_hmm_weekend_phaseF/model/` — dmax=268, `duration_family=negative_binomial`, unweighted emission (`logreg_class_weight=None`). Validated on load against exactly this configuration (`model_loader._validate`); any other checkpoint is rejected with `ModelIncompatibleError`.

## Installation / launch

```bash
cd Training
python -m reporting.api          # dev server, port 8000, model loaded eagerly at startup
```

Requires the conda env (`embryo_env`) with `flask` installed (already in `requirements.txt`), and the frozen model + `Embeddings/resnet18/{train,val}/` present at the expected relative paths (server-side only — not present in a bare local checkout, see `docs/REPRODUCIBILITY.md`).

## Endpoints

| Endpoint | Method | Params | Response | Source | Cache |
|---|---|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` | — | — |
| `/model` | GET | — | model name/version/configuration/source | `model_loader` | model cached after first load |
| `/videos` | GET | `split` (default `val`) | `{"split", "videos": [...]}` | `trajectory_service.list_videos` | trajectory list cached per split |
| `/videos/{id}` | GET | `split` | video name, n_windows, first/last window | `trajectory_service` | same |
| `/videos/{id}/windows` | GET | `split` | full `window_starts` list | `trajectory_service.get_windows` | same |
| `/videos/{id}/inference/{window_start}` | GET | `split` | full `InferenceRecord` (see `docs/INFERENCE_SCHEMA.md`) | `inference_service.infer` | forward pass (`filtering`/`next_phase_distribution`) cached per `(video, split)` after first request |
| `/videos/{id}/trajectory` | GET | `split` | `TrajectorySummary` incl. `transition_chain` | `inference_service.trajectory_summary` | reuses the same per-video cache |

`split` defaults to `"val"` everywhere and **"test" is rejected with HTTP 403** at two independent layers (the API's own `_split_param()`, and `trajectory_service._check_split()` underneath it) — not a single point of failure.

### Example response — `/videos/Patient_319/inference/54`

```json
{
  "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319", "patient_id": "Patient_319", "split": "val"},
  "window": {"window_start": 54, "window_start_time": null, "window_end_time": null, "window_mid_time": null,
             "window_duration": null, "time_available": false, "time_unit": "unknown/unverified",
             "n_zero_time_diffs_in_window": null},
  "current_state": {"current_phase": "t3", "current_phase_index": 4, "phase_probability": 0.31,
                     "phase_probabilities": {"tPB2": 0.01, "...": "...", "tEB": 0.00}, "entropy": 1.83},
  "next_phase": {"next_phase_distribution": {"t3": 0.55, "t4": 0.22, "...": "..."},
                 "most_likely_next_phase": "t3", "next_phase_probability": 0.55},
  "duration": {"duration_available": true, "expected_duration": 10.8,
               "duration_quantiles": {"0.1": 3, "0.25": 6, "0.5": 10, "0.75": 15, "0.9": 21}, "duration_unit": "windows"},
  "model": {"model_name": "semi_hmm", "model_version": "dmax=268,negative_binomial,alpha=1.0,rho=0.3,class_weight=None",
            "model_configuration": {"...": "..."}, "model_source": ".../semi_hmm_weekend_phaseF/model",
            "inference_timestamp": "2026-08-24T13:10:00+00:00"},
  "ground_truth": {"ground_truth_phase": "t3", "consistency_flag": 0},
  "warnings": []
}
```

Field-by-field provenance (AVAILABLE/DERIVED/UNAVAILABLE): `docs/INFERENCE_SCHEMA.md`.

## Errors

| Error | HTTP | Cause |
|---|---|---|
| `TestSplitLockedError` | 403 | `split=test` requested anywhere |
| `ModelIncompatibleError` | 500 | Loaded checkpoint doesn't match the expected reference config |
| `KeyError` (unknown video) | 404 | — |
| `WindowNotFoundError` | 404 | `window_start` not present in the requested trajectory |
| `ValueError` (bad split value) | 400 | Anything other than `train`/`val`/`test` |

## Val vs. Test

Every endpoint operates on `train`/`val` only. `Embeddings/resnet18/test/` is never read by any code path in this package — enforced by `trajectory_service.TestSplitLockedError`, tested directly (`test_trajectory_service.py::test_every_public_function_refuses_test_split`, parametrized over every public function) and at the API layer (`test_api.py::test_every_endpoint_rejects_test_split_with_403`).

## Causality

`current_state`/`next_phase` (via `filtering()`/`next_phase_distribution()`) are **causal** — valid for a live, in-progress sample, since they only condition on `O_1..O_t`. `/videos/{id}/trajectory`'s `transition_chain` is **retrospective** — it needs the full trajectory (`SemiHMMModel.transition_chain()`, built over ground-truth segments) and must never be presented as available before a video is complete.

## Known limitations

- ~~Cold-start cost is severe~~ — **RESOLVED (Phase 1.5, `docs/REPORTING_CACHE.md`)**: a disk-backed cache now persists each video's forward pass across process restarts, keyed by `(model_version, split, video_name)`. Every subsequent request (same process or a fresh one after a restart) is served in single-digit milliseconds — measured speedup **38,374x** on a real Val trajectory, with byte-identical (`np.array_equal`) output confirmed between cached and uncached results. See `docs/REPORTING_CACHE.md` and `docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md` for the full design and benchmark.
- ~~First-request-per-video cold cost still ~330-360s~~ — **REDUCED (Phase 1.6, `docs/SEMI_HMM_PHASE_1_6_REPORT.md`)**: `SemiHMMModel.next_phase_distribution()` itself was optimized (duration-model lookup tables precomputed once instead of recomputed per iteration inside the hot loop — no algorithm change, bit-identical output confirmed on synthetic fixtures, real Val data, and through the Phase 1.5 cache). First request per video now costs ~10-11s (measured, real Val trajectory `Patient_319`), not ~330-360s — a further **~32x** on top of Phase 1.5's cache. A full-Val pre-warm (`Training/reporting/warm_cache.py`) is expected to drop proportionally from Phase 1.5's ~9.7h estimate but has not been re-benchmarked end-to-end — still an operational scheduling decision, not an architectural gap.
- **Newly discovered, severe miscalibration of `current_state.phase_probabilities`**: scored directly this session (`Results/evaluation/global_model_comparison/PRAUC_REPORT.md`), the raw 15-way `filtering()` posterior has multiclass Brier≈0.96, log loss≈7.1, ECE≈0.46 — far worse than the raw emission alone and worse than a uniform 15-way guess on Brier/log-loss. Ranking (which phase is most likely) remains informative; the numeric probability values should never be presented as calibrated.
- `tPB2`/`tEB` durations are structurally inestimable — the API expresses this via `duration_available=False`, never a fabricated `expected_duration=0`.
- `time_unit` is always `"unknown/unverified"` — the API never converts window-index durations to a named time unit.
- No persistent/production deployment yet — `api.py`'s `__main__` block is a dev-only entry point (Flask's built-in server, `debug=False`, no WSGI server, no process manager).
- No authentication/authorization layer.

## Test status

**287/287 passed** (229 pre-existing + 58 new for this package, `Tests/reporting/`), confirmed by a full clean run on the server (2026-08-24, 24m18s — dominated by the real integration test's cold `next_phase_distribution()` cost, see Known Limitations above). Synthetic unit tests plus one real-data integration test (`test_integration_real_val.py`, skip-guarded when the real artifacts aren't present locally) against the frozen Semi-HMM and real Val trajectories. Two bugs were found and fixed during this validation, both in the *test code*, not the API: (1) an integration test assumed `window_start=0` exists for every video — it doesn't, `window_start` is a global index into the split's flattened window list, not a per-video offset; (2) a model-loader test was polluted by a leftover cached model from an earlier test file (module-level cache not reset) — fixed with an autouse reset fixture. TEST = LOCKED throughout — confirmed via the guard tests above, never contradicted. (Full-repo suite count as of Phase 1.6, 2026-08-25: **308/308** — see `docs/SEMI_HMM_PHASE_1_6_REPORT.md`.)

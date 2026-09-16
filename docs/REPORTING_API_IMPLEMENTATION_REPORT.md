# Reporting API Implementation Report

## 1. Objective

Build the service layer between the frozen Semi-HMM scientific model and future consumers (Web App, RAG), per the original task's success criteria — an engineering phase around already-validated science, no retraining, no model/embedding/result modification, Test locked.

## 2. Existing scientific pipeline

Frozen reference: Semi-HMM k=7 (`Results/evaluation/semi_hmm_weekend_phaseF/model/`, dmax=268, negative_binomial, unweighted emission) — the `CURRENT_PIPELINE_CANDIDATE` established in `docs/MODEL_COMPARISON.md`. Verified against real code (not assumed from docs) this session: `SemiHMMModel.filtering()` *does* produce a full causal 15-way posterior (a correction to an earlier documentation pass, which had incorrectly stated Semi-HMM had no equivalent to `HMMModel.explain()`).

## 3. Architecture implemented

`Training/reporting/{schemas,model_loader,trajectory_service,inference_service,api}.py` — see `docs/REPORTING_API.md` for the full module map and rationale (placed under `Training/`, not a new top-level directory; kept separate from `WebApplication/`'s unrelated Flask app).

## 4. Model loading

`model_loader.get_model()` — loads once (module-level cache), validates `n_states=15`, `dmax=268`, `duration_family='negative_binomial'`, `logreg_class_weight=None`, `embedding_dim=512`, raising `ModelIncompatibleError` (loud, not silent — see `model_loader.py`'s own docstring for why "refuse silently" was interpreted as "refuse cleanly without partial state," not "refuse without any error," given this project's consistent anti-silent-failure discipline elsewhere). Never refits, never writes a checkpoint.

## 5. Inference service

`inference_service.infer(video, split, window)` composes `model.filtering()`, `.next_phase_distribution()`, `.duration_models[...].expected_duration()/.quantiles()`, and `HMMModel.entropy()` (reused as a static method) into one `InferenceRecord`. Never reimplements any model mathematics. Per-video forward-pass caching (`_POSTERIOR_CACHE`/`_NEXT_PHASE_CACHE`) — required, not optional, given the cold-cost finding in §13.

## 6. Trajectory service

`trajectory_service.py` — embeddings/metadata access, `TestSplitLockedError` hard-guards every public function against `split="test"`. `metadata_with_time.csv` handled per-window with explicit `time_available`/`time_unit="unknown/unverified"` fields — never fabricates a timestamp or infers a unit.

## 7. API endpoints

7 endpoints (`/health`, `/model`, `/videos`, `/videos/{id}`, `/videos/{id}/windows`, `/videos/{id}/inference/{window}`, `/videos/{id}/trajectory`) — full table with method/params/response/source/cache in `docs/REPORTING_API.md`. Flask chosen after inspecting the repo (already a dependency, already the framework `WebApplication/` uses) — not a default preference.

## 8. Data schema

`docs/INFERENCE_SCHEMA.md` — every field tagged AVAILABLE/DERIVED/UNAVAILABLE against the real code, not assumed.

## 9. Timestamp handling

`window_end_time` used as the causal reference point per `metadata_with_time.manifest.json`'s own documented convention. `time_unit` always `"unknown/unverified"`, copied verbatim, never converted to a named unit. Train has 8/196,934 rows with incomplete time data (verified directly against the real manifest); Val has 0/43,339 — both numbers confirmed by reading the actual manifests, not assumed from the task prompt (though the prompt's own claims matched exactly).

## 10. Duration handling

`duration_available=False` (never `expected_duration=0`) for `tPB2`/`tEB`, detected via the duration model's own `fallback_uniform` attribute — not inferred from the numeric output.

## 11. Error handling

`TestSplitLockedError`→403, `ModelIncompatibleError`→500, `KeyError`→404, `WindowNotFoundError`→404, `ValueError`→400. Full table in `docs/REPORTING_API.md`.

## 12. Tests

**287/287 passed** (229 pre-existing + 58 new), confirmed via a full clean run on the server. Synthetic unit tests for schemas/model_loader/trajectory_service/inference_service/api, one real-data integration test against the frozen model and real Val trajectories (skip-guarded when real artifacts aren't present locally). Two test bugs found and fixed during validation (both in test code, not the API — see `docs/REPORTING_API.md`'s Test status section for detail): a `window_start=0` assumption that doesn't hold for every video, and a module-level cache leaking across test files.

## 13. Performance

**Critical finding**: `SemiHMMModel.next_phase_distribution()` costs **~333s for a single 527-window Val trajectory** (measured directly), a pure-Python, unvectorized O(T·K·Dmax) loop in the existing, frozen scientific model — not touched or optimized this session (out of scope: "ne pas modifier le modèle scientifique"). `filtering()` is cheap (~1.3s/video). This discovery came from a real production incident during testing: an early version of the integration test scanned 15 videos and ran for over an hour before being caught and fixed to scan 2. Per-video caching (§5) is the only mitigation implemented so far; a pre-warming strategy is recommended but not built (see `docs/PRODUCT_ARCHITECTURE.md` §9).

## 14. Web App integration contract

`docs/WEBAPP_DATA_REQUIREMENTS.md` already maps every UI element to a data source; with the API now built, each of those sources resolves to one of the 7 endpoints above (formalized in `docs/PRODUCT_ARCHITECTURE.md`/`docs/WEBAPP_ARCHITECTURE.md`).

## 15. RAG integration contract

`docs/RAG_DATA_MODEL.md`'s STRUCTURED layer now has a concrete implementation to point at (this API) instead of a plan; SEMANTIC/EXPERIMENTAL layers remain doc-based, formalized further in `docs/RAG_ARCHITECTURE.md`.

## 16. Known limitations

See `docs/REPORTING_API.md`'s Known Limitations section in full: cold-inference cost (§13 above), the newly-discovered severe miscalibration of the raw `phase_probabilities` output (Brier≈0.96, worse than uniform — `Results/evaluation/global_model_comparison/PRAUC_REPORT.md`), no production deployment, no auth.

## 17. Next step

At the time this report was finally written, the project had already moved on to two follow-up asks (a PR-AUC metric addition, then this product-architecture planning pass) — both completed without needing to revisit this API's code. The concrete next step for the API itself is the pre-warming/performance mitigation flagged in §13, now formally the recommended first step of `docs/PRODUCT_ROADMAP.md`.

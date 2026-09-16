# Web App Architecture

Planning only — not implemented. Builds on `docs/WEBAPP_DATA_REQUIREMENTS.md` (which data, from where) by adding panel structure and user journeys. Framework choice deliberately left open (see Decisions Required in `docs/PRODUCT_ARCHITECTURE.md`) — this document is about structure, not a specific React/Vue/etc. implementation.

## 1. Panel layout

```
┌─────────────────────────────┬──────────────────────────────┐
│                              │                               │
│         VISIONNEUSE          │          ANALYSE               │
│                              │                               │
│  timelapse / timeline        │  current phase + probability   │
│  current frame                │  phase probabilities (bar)     │
│  timestamp (window-index      │  entropy / uncertainty         │
│   based, unit-labeled          │  next phase distribution       │
│   "unknown/unverified" if     │  expected duration + interval  │
│   real time unavailable)      │  transition chain (completed   │
│  ground-truth phase, if        │   videos only)                 │
│   available                    │                               │
│                              │                               │
├─────────────────────────────┴──────────────────────────────┤
│                            CHAT                                │
│  question / answer / sources / data used                       │
└──────────────────────────────────────────────────────────────┘
```

Matches the two-zone layout already specified in `docs/WEBAPP_DATA_REQUIREMENTS.md`, with Chat added as a third zone per this planning pass's new requirement (§2 C of the request).

## 2. Visionneuse

- Video/embryo selector → `GET /videos`.
- Timelapse scrubber → `GET /videos/{id}/windows` for the window list; frame *images* are **not yet servable** (flagged as a real gap in `docs/WEBAPP_DATA_REQUIREMENTS.md` — no image-serving endpoint exists; raw frames live on the server filesystem, `Data/embryo_dataset_F0/`, with no HTTP access layer built).
- Current window pointer → drives the Analyse panel's queries (`GET /videos/{id}/inference/{window}`).
- Timestamp display → `window.window_start_time`/`window_end_time`/`window_mid_time` with `time_unit` always shown alongside the number (never silently labeled "seconds" or "hours") — direct UI consequence of `docs/INFERENCE_SCHEMA.md`'s `time_unit` field.
- Ground-truth phase, when available → `ground_truth.ground_truth_phase`, displayed distinctly from the model's own `current_phase` (never merged into one field — a real prediction vs. a real label must stay visually distinguishable).

## 3. Analyse

All fields sourced from one `GET /videos/{id}/inference/{window}` call (`InferenceRecord`) except `transition_chain` (`GET /videos/{id}/trajectory`, retrospective-only — must be visually marked as "full-trajectory view," not live).

- **Phase probabilities**: bar chart over 15 phases. **Must carry a persistent, non-dismissible caveat** given the newly-discovered miscalibration (`PRAUC_REPORT.md`: Brier≈0.96, worse than uniform) — present as a ranking ("most likely: t6"), never as a percentage confidence a user could read as calibrated.
- **Entropy/uncertainty**: a single scalar, best shown as a relative indicator (low/medium/high, or a gauge against the phase's own typical range) rather than a raw nats value most users won't interpret correctly.
- **Next phase distribution**: same bar-chart treatment and same calibration caveat as current-phase probabilities.
- **Expected duration + interval**: `duration.expected_duration` + `duration_quantiles` rendered as a range ("likely to last N-M more windows"), never a bare point estimate — direct consequence of the project's own "CDF/quantiles, never raw pmf" rule (`docs/SCIENTIFIC_REPORT.md` §18). When `duration_available=False` (tPB2/tEB), show "not estimable" text, never a blank or a zero.
- **Transition chain**: a distinct, clearly-labeled sub-view (segment table or timeline), only for videos with a complete trajectory loaded — must not appear as if it were a live/predictive feature.

## 4. Visualizations (beyond the two panels above)

- **Phase-probability evolution over time**: a stacked-area or multi-line chart of `phase_probabilities` across all windows of a trajectory — requires calling `/inference/{window}` for every window (or a new batch endpoint, see Decisions Required — this could be a meaningful Reporting API addition if per-window calls prove too chatty for this specific chart).
- **Transitions**: rendered from `transition_chain` — a segment-by-segment timeline (phase, duration, transition probability to the next).
- **Duration**: a distribution plot per phase (quantile band), reusing `duration_quantiles` directly.
- **Uncertainty over time**: entropy per window, plotted alongside the phase timeline — visually correlates "where the model is unsure" with "which phase," directly surfacing the known t3/t5/t6/tPNf weakness rather than hiding it.

## 5. Chat

- Question input, routed through the Application API → LLM Orchestrator (`docs/LLM_ORCHESTRATION.md`).
- Answer rendered with **visible source tags** distinguishing live-data citations from documentation citations (§4 of that doc) — not optional, a direct product requirement given §27's "predictions must remain auditable" constraint.
- "Data used" disclosure: which tool calls / RAG chunks backed this specific answer — supports the same auditability goal, and gives a user a way to sanity-check an answer against the underlying FACT.

## 6. User journeys

### Journey 1 — Explore a trajectory

`Web App (video picker) → GET /videos → Web App (timelapse) → GET /videos/{id}/windows → user scrubs → GET /videos/{id}/inference/{window} per stop → Analyse panel updates`. No LLM involved — pure Reporting API traffic, lowest latency path. Cold-start risk: the *first* window request for a not-yet-cached video pays the ~333s `next_phase_distribution()` cost (§9 of `docs/PRODUCT_ARCHITECTURE.md`) — the UI must show a distinct "computing" state for this, not a generic spinner indistinguishable from a fast request.

### Journey 2 — Understand a prediction

`Web App (Analyse panel, already showing a window) → user clicks "why?" → Chat panel opens, pre-filled with a "why t5" question tied to the current video/window → Application API → LLM Orchestrator → get_inference (reuses already-cached data, fast) + search_knowledge(methodological/project) → answer with FACT + DOCUMENTED blocks, citations shown`. Directly exercises Scenario C/G from `docs/LLM_ORCHESTRATION.md` §5.

### Journey 3 — Ask a general scientific question

`Web App (Chat, no specific video context) → "pourquoi Semi-HMM plutôt que GRU ?" → Application API → LLM Orchestrator → routing detects "comparison/methodology" category → search_knowledge(project_knowledge, methodological_knowledge) only, no tool calls → answer synthesized from docs/MODEL_COMPARISON.md / docs/SCIENTIFIC_REPORT.md content, citations shown`. No Reporting API call at all — the orchestrator's routing step (§1 of `docs/LLM_ORCHESTRATION.md`) is what keeps this path cheap and fast.

"""
Phase 7 -- Application API (BFF), the first real HTTP surface for
`Training/orchestrator/` and a thin video/timeline layer over
`Training/reporting/`. Purely additive, like every other package in the
SciML extension: never edits Training/reporting/, Training/orchestrator/,
Training/rag/, or the two original project halves (Training/'s classifier
pipeline, WebApplication/). Placed under Training/webapp_api/ (not inside
WebApplication/, a different, pre-existing, unrelated doctor/admin CRUD
app this extension never edits -- see CLAUDE.md).

This is the "Application API (BFF)" box from docs/PRODUCT_ARCHITECTURE.md
sec 3 / docs/WEBAPP_INTEGRATION_PLAN.md: one Flask app, composing
`Training/reporting/`'s functions and `Training/orchestrator/answer_question()`
IN-PROCESS (direct Python imports, no second HTTP hop to
Training/reporting/api.py's own standalone Flask app) -- fewer moving
parts, one origin for the frontend, nothing scientific reimplemented.

Also serves the minimal frontend (Jinja2 templates + static JS/CSS) from
this same process -- no separate frontend server, no build step, no npm
dependency anywhere in this repo (matches WebApplication/'s own stack,
the only existing precedent in this project).

Module map
----------
app.py    Flask app + every route: GET /videos, GET /videos/<id>,
          GET /videos/<id>/timeline, GET /videos/<id>/window/<window>,
          GET /videos/<id>/window/<window>/frame, POST /chat, plus
          GET / (the minimal frontend page).

Hard rule (unchanged from every layer below this one): the Reporting API
remains the sole source of truth for a live/dynamic number; the LLM/RAG
path never gets a second, parallel way to obtain one. This package adds
no new data path -- it only exposes the two that already exist over HTTP.

Frame serving (Phase 7.8): `GET /videos/<id>/window/<window>/frame` serves
the real JPEG for a window via `Training/reporting/frame_mapping.window_to_frame_path()`
(built + proven in Phase 7.7 -- 6467/6467 real windows matched). The
server does the `window_start` -> filesystem path mapping; the client
never receives or constructs a path. `/videos/<id>/window/<window>` also
now returns a `frame` block (`available`, `frame_number`, `frame_url`) so
the frontend never has to guess whether a frame exists before requesting
it. See `docs/WEBAPP_VIEWER.md`.
"""

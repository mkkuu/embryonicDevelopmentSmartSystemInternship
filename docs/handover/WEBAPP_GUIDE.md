# Web application — current BFF + frontend, endpoints, and the legacy app

Written 2026-09-17 from `Training/webapp_api/app.py`, `templates/index.html`, `static/app.js`
and the tests under `Tests/webapp_api/`. As-built history: `docs/reference/PHASE_7_FINAL_REPORT.md`,
`docs/archive/reports/WEBAPP_{VIEWER,CHAT,TIMELINE_PERFORMANCE,INTEGRATION_PLAN}.md`. The RAG
document `docs/WEBAPP_ARCHITECTURE.md` is an early plan ("not implemented") kept frozen for
the index; do not read it as a description of the current application.

## 1. Architecture

```
Browser (French UI, one page)
  → Training/webapp_api/app.py   Flask BFF, port 8001, debug=False, binds 127.0.0.1 (Flask default), no authentication
      ├─ Training/reporting       in-process: frozen Semi-HMM, embeddings, cache, observed transitions, frames
      └─ Training/orchestrator    in-process: answer_question() → router → tools / RAG → temporal context
                                  → context builder → Ollama (llm_config.LLM_MODEL) → grounding → Validator
```

No second HTTP hop: the BFF imports `reporting` and `orchestrator` directly. The frontend never
talks to Ollama, Chroma or the model. The standalone Reporting API (`python -m reporting.api`,
port 8000) exposes the same inference without the language model and is optional.

Separation UI / science: the frontend holds no scientific logic and no question wording; it
renders what the backend returns. The backend holds no copy of Q1–Q15 either: `/questions` and
`question_id` resolve from `orchestrator.fixed_question_benchmark.FIXED_QUESTIONS`.

## 2. Running it (on the GPU server, where the inputs are)

```bash
cd ~/projects/embryonicDevelopmentSciMLExtension/Training
PY=~/miniconda3/envs/embryo_env/bin/python
CUDA_VISIBLE_DEVICES="" $PY -m reporting.warm_cache --split val     # optional: pre-fills Cache/ (first inference per video is ≈ 11 s cold)
CUDA_VISIBLE_DEVICES="" $PY -m webapp_api.app                       # http://localhost:8001 (use an SSH tunnel: ssh -L 8001:localhost:8001 …)
```

Requirements: `Results/evaluation/semi_hmm_weekend_phaseF/model/` (validated at first use,
`ModelIncompatibleError` otherwise), `Embeddings/resnet18/{train,val}/`, `Data/embryo_dataset_F0`
+ `_annotations` (frames), `RagIndex/` (retrieval), `docs/corpus/` (Validator), and the Ollama
service with the model named by `LLM_MODEL` already pulled.

Environment variables (`docs/ARCHITECTURE.md` §3): `LLM_MODEL` (default `llama3.2:latest`),
`LLM_TIMEOUT_SECONDS` (default 120), `EMBRYO_OLLAMA_NUM_CTX` (default 16384),
`EMBRYO_LLM_CONTEXT_FORMAT` (`v2` default, `compact`, `compact_v2`), `EMBRYO_SCIENTIFIC_VALIDATOR`
(`on` default). Changing `LLM_MODEL` for production is a decision that must follow a comparison
on the frozen benchmark; the benchmark model `mistral-nemo:12b` is not the production default.

## 3. Endpoints of the BFF (verified in `app.py`)

| Method, path | Returns | Notes |
|---|---|---|
| `GET /` | the single-page frontend (`templates/index.html`) | French UI |
| `GET /health` | service status | — |
| `GET /videos?split=val` | the list of videos (patients) of the split | `split` defaults to `val`; `test` → **403** |
| `GET /videos/<id>?split=val` | `video_name`, `split`, `n_windows`, `first_window`, `last_window` | — |
| `GET /videos/<id>/timeline?split=val` | the model's own causal per-window phase, run-length-encoded into contiguous segments (**never** the ground truth, so the two are never shown under one label) | built from the cached inference |
| `GET /videos/<id>/window/<w>?split=val` | the full inference record of one window: current phase, `phase_probabilities`, entropy, next phase, duration, ground truth, `time_unit = unknown/unverified` | window = `window_start` index |
| `GET /videos/<id>/window/<w>/frame?which=first|last&split=val` | a real JPEG frame of the window | path-traversal guarded (`reporting/frame_mapping.py`) |
| `GET /questions` | the frozen Q1–Q15 (id, wording, category) | served from the benchmark module |
| `POST /chat` | `question_id`, `question`, `answer`, `grounded`, `confidence`, `sources`, `warnings`, `tools_called`, `route`; plus `raw_answer`, `validation` (summary: enabled, provenance_valid, has_violation, n_claims, by_status, qualification_appended, error) and `context_format` when the Validator ran | body `{question \| question_id, video_id, window, split}`; both `question` and `question_id` → **400**; `split=test` → **403** |

`answer` is the final answer (raw + "QUALIFICATION SCIENTIFIQUE" block when the Validator is
on); `raw_answer` is never rewritten; `validation` carries the claims, provenance and verdicts.

## 4. Frontend (`static/app.js`, `static/style.css`, `templates/index.html`)

- **Patient panel**: lists the videos of the `val` split in natural order (`Patient_2` before
  `Patient_10`), with a text filter that never switches the loaded patient by itself; selecting
  one loads its timeline (`Tests/webapp_api/test_patient_selection.mjs`).
- **Timeline and viewer**: phase bands of the model's segments over the windows, navigation
  window by window; the current-window panel shows the inference record and the first/last JPEG
  of the selected window, degrading to "frame unavailable" rather than failing
  (`test_viewer_navigation.mjs`; performance notes in the archived report).
- **Question interface**: the frozen Q1–Q15 panel (send by `question_id`) and a free-text field
  (send by `question`), always bound to the selected patient and window — the backend never
  guesses a video or window (`test_fixed_questions.mjs`).
- **Response handling**: loading / error / stale states; the answer, the qualification block
  and the grounding flag are rendered as text. Every backend string goes through
  `textContent` / `createElement`, never `innerHTML` (`test_chat_dom_safety.mjs` and the static
  audit `test_frontend_dom_safety_audit.py` pin this).
- No build tooling, no npm, no external URL: self-hosted Inter fonts and the LabISEN logo under
  `static/`; `style.css` forbids external resources.

Tests: `pytest Tests/webapp_api/` (Flask test client, mocked services) and the four Node files
(`node Tests/webapp_api/*.mjs`, zero dependencies, a self-written fake DOM). A regression in the
DOM-safety, panel or fixed-question contracts is caught by pytest even if Node is not run.

## 5. Security posture of the BFF

No authentication, no session, no rate limiting; intended for a single operator through an SSH
tunnel on the server. `split=test` refused at the BFF, at `reporting/trajectory_service.py` and
at `reporting/api.py` (defence in depth: keep all three). Frame access validated against the
dataset layout. Details: `docs/reference/WEBAPP_SECURITY.md` (local-only). Do not expose the port.

## 6. The legacy application (`WebApplication/`)

The original Flask + PostgreSQL application (login, doctor/admin CRUD, embryo upload, a
prediction endpoint), untouched by the extension and **not on the current path**: no import in
either direction, no tests, `debug=True`, plain-text password comparison, IDE files tracked.
Its `/Doctor/Embryo/PREDICT` loads `../Results/resnet18/best_model.pth` with `torch.load`
expecting a whole model while the pipeline saves a `state_dict`, and that checkpoint no longer
exists: it silently returns random predictions flagged `is_random: true`.

```bash
cd WebApplication && flask run          # port 5000; needs .env built from .env.example and Dataset_shema.sql applied
```

Routes as registered in `Routes/Doctor_Routes.py` and `Routes/Admin_Routes.py` (verified):
`/Register`, `/Doctor`, `/Doctor/{ADD,UPDATE,ACCESS,LIST,DELETE/<id>}`, `/Embryo`,
`/Embryo/{LIST,ADD,UPDATE,DELETE,GET_IMAGES,PREDICT}`, `/Embryo/IMAGE/<id>/<filename>`; login
and role dispatch in `app.py` / `HandleAccess.py`. `app.secret_key` is generated with
`os.urandom` at start-up (the `SECRET_KEY` variable shown in the original README is not read).
The original author's README (`git show d0181fd:README.md`) documents this application, its
PostgreSQL set-up and the training CLI in full and remains the reference for it.

Its `.env` was a tracked secret until 2026-09-15 (`README.md` §14 "Security"); the credential
is still not rotated. Whether to archive the application or move it to its own repository is an
open decision, to be taken after the PostgreSQL side is reviewed.

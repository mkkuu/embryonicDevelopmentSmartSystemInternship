"""
Phase 7 Application API (BFF) -- Flask, matching the rest of this
project's stack (`Training/reporting/api.py`, `WebApplication/`'s own
app.py; no new web framework dependency). Composes
`Training/reporting/`'s functions and `Training/orchestrator/orchestrator.answer_question()`
IN-PROCESS -- direct Python imports, never a second HTTP hop to
Training/reporting/api.py's own standalone Flask app.

Endpoints (docs/PROJECT_CHECKPOINT.md Phase 7 section has the full design
review):
  GET  /health
  GET  /videos?split=val
  GET  /videos/<video_id>?split=val
  GET  /videos/<video_id>/timeline?split=val
  GET  /videos/<video_id>/window/<window_start>?split=val
  GET  /videos/<video_id>/window/<window_start>/frame?split=val&which=last
  GET  /questions                          (the frozen Q1-Q15, read-only)
  POST /chat                               (free text, or a frozen question by id)
  GET  /                                   (the minimal frontend page)

`split` defaults to "val" everywhere and "test" is rejected with 403 --
the same defense-in-depth convention `Training/reporting/api.py` already
uses (every layer checks split, not just the innermost one).

The frontend never talks to Ollama, ChromaDB, or the Semi-HMM directly --
this Flask app is the only network-facing surface; everything scientific
stays server-side, in-process, one hop from the frozen model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request, send_file  # noqa: E402

from orchestrator import fixed_question_benchmark, llm_config  # noqa: E402
from orchestrator import orchestrator as orchestrator_module  # noqa: E402
from reporting import frame_mapping, inference_service, model_loader, trajectory_service  # noqa: E402
from reporting.trajectory_service import TestSplitLockedError  # noqa: E402

app = Flask(__name__)

# The real local provider, built through the ONE shared resolver
# (`orchestrator/llm_config.py`): `LLM_MODEL=<tag>` selects the generative
# model, `LLM_TIMEOUT_SECONDS=<s>` its per-question budget; the defaults
# (llama3.2:latest, 120 s) are documented there. Only the generative model
# is selectable -- Router, tools, RAG, context, grounding and the scientific
# models are untouched by that choice. Constructing the provider makes no
# network call (OllamaLLMProvider.__init__ only stores config); if Ollama is
# unreachable when a real chat request arrives,
# orchestrator.generate_grounded_response()'s existing timeout/exception
# safety net degrades to an honest fallback response, never a crash --
# that guarantee already exists and is tested
# (Tests/orchestrator/test_grounded_generation.py).
_CHAT_PROVIDER = llm_config.build_ollama_provider()
_CHAT_LLM_TIMEOUT_SECONDS = _CHAT_PROVIDER.request_timeout_seconds

# The frozen benchmark questions, by id -- the SAME objects the benchmark
# runner iterates (`orchestrator.fixed_question_benchmark.FIXED_QUESTIONS`).
# This module never holds a second copy of any wording.
_FIXED_QUESTIONS_BY_ID = {q.question_id: q for q in fixed_question_benchmark.FIXED_QUESTIONS}


def _split_param() -> str:
    split = request.args.get("split", "val").lower()
    if split == "test":
        raise TestSplitLockedError(
            "split='test' was requested via the Application API. Test is locked for this "
            "project -- see docs/SCIENTIFIC_REPORT.md sec 21."
        )
    return split


@app.errorhandler(TestSplitLockedError)
def _handle_test_locked(e):
    return jsonify({"error": "test_locked", "message": str(e)}), 403


@app.errorhandler(model_loader.ModelIncompatibleError)
def _handle_model_incompatible(e):
    return jsonify({"error": "model_incompatible", "message": str(e)}), 500


@app.errorhandler(KeyError)
def _handle_not_found(e):
    return jsonify({"error": "not_found", "message": str(e)}), 404


@app.errorhandler(frame_mapping.FrameMappingError)
def _handle_frame_mapping_error(e):
    # Covers AnnotationsNotFoundError and FrameFileNotFoundError too (both
    # subclass FrameMappingError) -- Flask dispatches to the closest
    # registered handler in the MRO, so one handler is enough. Never a 500:
    # every case here is "this window/frame genuinely doesn't resolve",
    # not a server fault.
    return jsonify({"error": "frame_not_found", "message": str(e)}), 404


@app.errorhandler(ValueError)
def _handle_bad_request(e):
    return jsonify({"error": "bad_request", "message": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/videos")
def videos():
    split = _split_param()
    return jsonify({"split": split, "videos": trajectory_service.list_videos(split)})


@app.route("/videos/<video_id>")
def video_detail(video_id):
    split = _split_param()
    windows = trajectory_service.get_windows(video_id, split)
    return jsonify({
        "video_name": video_id, "split": split, "n_windows": len(windows),
        "first_window": windows[0] if windows else None,
        "last_window": windows[-1] if windows else None,
    })


def _phase_timeline_segments(video_id: str, split: str) -> Dict[str, Any]:
    """Run-length-encodes the model's own CAUSAL per-window prediction
    (`inference_service.infer()`'s `current_state.current_phase`, never
    ground truth -- this endpoint answers "what does the model say
    happened," not "what the annotation says," to avoid ever presenting
    the two with the same label) into contiguous phase segments -- pure
    presentation-layer grouping, no new scientific computation. Each
    `infer()` call after the first for this video reuses the cached
    (T,K) forward-pass arrays (Training/reporting/cache.py) -- only the
    first request for a given (video, split) pays the cold cost."""
    window_starts = trajectory_service.get_windows(video_id, split)
    segments = []
    for window_start in window_starts:
        record = inference_service.infer(video_id, split, window_start).to_dict()
        phase = record["current_state"]["current_phase"]
        probability = record["current_state"]["phase_probability"]
        start_time = record["window"]["window_start_time"]
        end_time = record["window"]["window_end_time"]
        if segments and segments[-1]["phase"] == phase:
            seg = segments[-1]
            seg["end_window"] = window_start
            seg["end_time"] = end_time
            seg["n_windows"] += 1
            seg["probabilities"].append(probability)
        else:
            segments.append({
                "phase": phase, "start_window": window_start, "end_window": window_start,
                "start_time": start_time, "end_time": end_time, "n_windows": 1,
                "probabilities": [probability],
            })
    for seg in segments:
        seg["mean_probability"] = sum(seg["probabilities"]) / len(seg["probabilities"])
        del seg["probabilities"]
    return {"video_name": video_id, "split": split, "n_windows": len(window_starts), "segments": segments}


@app.route("/videos/<video_id>/timeline")
def video_timeline(video_id):
    split = _split_param()
    return jsonify(_phase_timeline_segments(video_id, split))


def _frame_info(video_id: str, split: str, window_start: int) -> Dict[str, Any]:
    """Best-effort frame metadata for the current-window panel -- never
    raises: an unmapped/missing frame degrades to `available: False`
    rather than failing the whole `/window/<w>` response, since frame
    availability is a separate concern from inference correctness (the
    inference record itself does not depend on a real JPEG existing).
    Deliberately reuses frame_mapping.window_frame_numbers() -- no second
    mapping logic."""
    try:
        frame_numbers = frame_mapping.window_frame_numbers(video_id, split, window_start)
    except (frame_mapping.FrameMappingError, KeyError):
        return {
            "available": False, "frame_number": None, "frame_url": None,
            "first_frame_number": None, "last_frame_number": None, "n_frames": 0,
            "first_frame_url": None, "last_frame_url": None,
        }
    base_url = f"/videos/{video_id}/window/{window_start}/frame?split={split}"
    return {
        "available": True,
        # Unchanged fields (docs/WEBAPP_VIEWER.md's documented `frame` block):
        # the window's causal "current" frame, i.e. its LAST one.
        "frame_number": frame_numbers[-1],
        "frame_url": base_url,
        # Viewer-only additions (Phase 7.11): the window's first/last real raw
        # frame numbers and their URLs, so the viewer can LABEL which of the
        # window's frames is on screen and offer the `which=first` view the
        # /frame endpoint already supports. These come from the SAME
        # `window_frame_numbers()` list this function already computed and
        # otherwise discarded -- no new computation, no new endpoint, no
        # scientific value derived or changed. The raw frame numbers are NOT
        # contiguous across consecutive windows (verified on real Val data:
        # Patient_319 window 156 -> frames 167..174 but window 157 -> 175..182),
        # so a client must READ these numbers per window and must never
        # compute a neighbouring window's frame number by arithmetic.
        "first_frame_number": frame_numbers[0],
        "last_frame_number": frame_numbers[-1],
        "n_frames": len(frame_numbers),
        "first_frame_url": f"{base_url}&which=first",
        "last_frame_url": f"{base_url}&which=last",
    }


@app.route("/videos/<video_id>/window/<int:window_start>")
def video_window(video_id, window_start):
    split = _split_param()
    record = inference_service.infer(video_id, split, window_start).to_dict()
    record["frame"] = _frame_info(video_id, split, window_start)
    return jsonify(record)


@app.route("/videos/<video_id>/window/<int:window_start>/frame")
def video_window_frame(video_id, window_start):
    """Serves the real JPEG for this window via
    frame_mapping.window_to_frame_path() -- the server does the mapping,
    the client never receives or constructs a filesystem path. No path
    ever comes from client input directly: `video_id` must first resolve
    to a real video in trajectory_service's embeddings-backed index (an
    unknown name raises KeyError -> 404 before any filesystem path is
    built from it, see frame_mapping.window_frame_numbers), and
    `window_start` is an <int:...> route converter, so neither can inject
    a path segment. `which` ("first"|"last", default "last") is validated
    by window_to_frame_path itself (ValueError -> 400 via the handler
    above)."""
    split = _split_param()
    which = request.args.get("which", "last")
    path = frame_mapping.window_to_frame_path(video_id, split, window_start, which=which)
    return send_file(path, mimetype="image/jpeg")


def _chat_sources(document_context) -> list:
    """Compact citation list for the UI -- source/section/status only,
    never the full chunk content blob (that stays available in the raw
    `answer_question()` return for anyone who wants it, but the chat
    response this endpoint returns is meant to be small)."""
    return [
        {"source": c.get("source"), "section": c.get("section"), "status": c.get("status")}
        for c in (document_context or [])
    ]


def _fixed_question_payload() -> Dict[str, Any]:
    """Read-only projection of the frozen Q1-Q15 for the UI: id, 1-based
    number and the verbatim wording, in benchmark order. Category is
    included as a display grouping hint only. No grading field
    (expected_information / success_criterion / ...) is ever served -- the
    UI must never show, and a user must never be able to read, the expected
    answer next to the question."""
    return {
        "n_questions": len(fixed_question_benchmark.FIXED_QUESTIONS),
        "questions": [
            {"id": q.question_id, "number": i + 1, "question": q.question, "category": q.category}
            for i, q in enumerate(fixed_question_benchmark.FIXED_QUESTIONS)
        ],
    }


@app.route("/questions")
def questions():
    return jsonify(_fixed_question_payload())


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    question_id = (body.get("question_id") or "").strip() or None
    question = (body.get("question") or "").strip()
    if question_id is not None and question:
        raise ValueError("Send either 'question_id' (a frozen benchmark question) or 'question' "
                         "(free text), not both.")
    if question_id is not None:
        fixed = _FIXED_QUESTIONS_BY_ID.get(question_id)
        if fixed is None:
            raise ValueError(f"Unknown question_id {question_id!r}; expected one of "
                             f"{sorted(_FIXED_QUESTIONS_BY_ID)}.")
        # The wording sent to the pipeline is the benchmark's own object --
        # the frontend never carries the text back, so it cannot drift.
        question = fixed.question
    if not question:
        raise ValueError("'question' (or 'question_id') is required and must not be empty.")
    video_id: Optional[str] = body.get("video_id") or None
    window = body.get("window")
    if window is not None:
        window = int(window)
    split = (body.get("split") or "val").lower()
    if split == "test":
        raise TestSplitLockedError(
            "split='test' was requested via POST /chat. Test is locked for this project."
        )

    out = orchestrator_module.answer_question(
        question, video_id=video_id, window=window, split=split, llm=_CHAT_PROVIDER,
        llm_timeout_seconds=_CHAT_LLM_TIMEOUT_SECONDS,
    )
    response = out["response"]
    body = {
        "question_id": question_id,
        "question": question,
        # The answer shown is the FINAL answer: the raw LLM text plus the
        # Scientific Validator's appended qualification block (identical to
        # the raw text when the validator is off or has nothing to add).
        "answer": out.get("final_answer", response["text"]),
        "grounded": response["grounded"],   # presence check -- never a truth verdict
        "confidence": response["confidence"],
        "sources": _chat_sources(out["context"].get("document_context")),
        "warnings": response["warnings"],
        "tools_called": response["used_tools"],
        "route": out["route"]["category"],
    }
    validation = out.get("validation")
    if validation is not None:
        body["raw_answer"] = response["text"]
        body["validation"] = _validation_summary(validation)
        body["context_format"] = (out.get("context_policy") or {}).get("context_format")
    return jsonify(body)


def _validation_summary(validation: dict) -> dict:
    """The audit-relevant part of the validator's output, without the full
    per-claim report (available from answer_question() directly)."""
    claim_validation = validation.get("claim_validation") or {}
    return {
        "enabled": validation.get("enabled"),
        "provenance_valid": validation.get("provenance_valid"),
        "has_violation": claim_validation.get("has_violation"),
        "n_claims": claim_validation.get("n_claims"),
        "by_status": claim_validation.get("by_status"),
        "qualification_appended": validation.get("qualification_appended"),
        "error": validation.get("error"),
    }


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Dev-only entry point, matching Training/reporting/api.py's own
    # convention. Loads the model eagerly so the first real request isn't
    # the one paying the load cost.
    model_loader.get_model()
    app.run(debug=False, port=8001)

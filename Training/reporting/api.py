"""
Flask app exposing the Reporting API. Flask, not FastAPI, chosen after
inspecting the repository (not by default preference): `flask` is already
a listed dependency (requirements.txt) and is the framework
WebApplication/ already uses throughout (Routes/*.py, app.py) --
`fastapi` appears nowhere in the repo. Using Flask keeps this new
component consistent with the project's existing stack instead of adding
a second web framework for no functional reason.

This app is intentionally NOT integrated into WebApplication/'s existing
Flask app (Routes/Doctor_Routes.py, Routes/Admin_Routes.py, etc.) --
those are the original project's doctor/admin CRUD blueprints, a
different concern with its own PostgreSQL-backed auth model
(HandleAccess.GlobalDataBase). The Reporting API is a separate, stateless,
read-only service over frozen scientific artifacts; running it as its own
small Flask app keeps that boundary explicit rather than smuggling a
research-reporting endpoint into the clinical CRUD app's blueprint
structure.

Endpoints (docs/REPORTING_API.md has full examples):
  GET /health
  GET /model
  GET /videos?split=val
  GET /videos/<video_id>?split=val
  GET /videos/<video_id>/windows?split=val
  GET /videos/<video_id>/inference/<window_start>?split=val
  GET /videos/<video_id>/inference_history/<center_window>?split=val&before=5&after=5
  GET /videos/<video_id>/trajectory?split=val

`split` defaults to "val" everywhere -- "test" is rejected with 403 at
this layer too (defense in depth on top of trajectory_service's own
guard), so a malformed/malicious request can never reach the filesystem
layer that could touch Embeddings/resnet18/test/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request  # noqa: E402

from . import inference_service, model_loader, trajectory_service  # noqa: E402
from .trajectory_service import TestSplitLockedError  # noqa: E402

app = Flask(__name__)


def _split_param() -> str:
    split = request.args.get("split", "val").lower()
    if split == "test":
        raise TestSplitLockedError(
            "split='test' was requested via the API. Test is locked for this project -- "
            "see docs/SCIENTIFIC_REPORT.md sec 21."
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


@app.errorhandler(ValueError)
def _handle_bad_request(e):
    return jsonify({"error": "bad_request", "message": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model")
def model_info():
    model = model_loader.get_model()
    return jsonify({
        "model_name": "semi_hmm",
        "model_version": model_loader.model_version_string(model),
        "model_configuration": model_loader.model_configuration(model),
        "model_source": str(model_loader.REFERENCE_MODEL_PATH),
    })


@app.route("/videos")
def videos():
    split = _split_param()
    return jsonify({"split": split, "videos": trajectory_service.list_videos(split)})


@app.route("/videos/<video_id>")
def video_detail(video_id):
    split = _split_param()
    windows = trajectory_service.get_windows(video_id, split)
    return jsonify({"video_name": video_id, "split": split, "n_windows": len(windows),
                     "first_window": windows[0] if windows else None,
                     "last_window": windows[-1] if windows else None})


@app.route("/videos/<video_id>/windows")
def video_windows(video_id):
    split = _split_param()
    return jsonify({"video_name": video_id, "split": split,
                     "window_starts": trajectory_service.get_windows(video_id, split)})


@app.route("/videos/<video_id>/inference/<int:window_start>")
def video_inference(video_id, window_start):
    split = _split_param()
    record = inference_service.infer(video_id, split, window_start)
    return jsonify(record.to_dict())


@app.route("/videos/<video_id>/inference_history/<int:center_window>")
def video_inference_history(video_id, center_window):
    """Multi-window band around a center window -- how the model's belief
    EVOLVES, not just its value at one window. `before`/`after` default to
    5 each and are clamped to this video's real trajectory bounds (a
    warning is included when clamped). A bad band raises ValueError -> 400;
    an unknown center window raises WindowNotFoundError (a KeyError
    subclass) -> 404."""
    split = _split_param()
    before = request.args.get("before", default=5, type=int)
    after = request.args.get("after", default=5, type=int)
    history = inference_service.infer_history(video_id, split, center_window, before, after)
    return jsonify(history.to_dict())


@app.route("/videos/<video_id>/trajectory")
def video_trajectory(video_id):
    split = _split_param()
    summary = inference_service.trajectory_summary(video_id, split)
    return jsonify(summary.to_dict())


if __name__ == "__main__":
    # Dev-only entry point. Loads the model eagerly so the first real
    # request isn't the one paying the ~30-170s load/fit-adjacent cost
    # (model_loader.get_model() is a pure load, not a fit -- see its
    # docstring -- but StandardScaler/LogisticRegression reconstruction
    # and the embeddings cache's own lazy load still take real time).
    model_loader.get_model()
    app.run(debug=False, port=8000)

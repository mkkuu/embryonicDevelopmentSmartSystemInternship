"""Flask test-client tests for the Phase 7 Application API (BFF),
Training/webapp_api/app.py -- matches Tests/reporting/test_api.py's own
convention (test client + monkeypatch on the underlying service
functions, no real data/model/Ollama server needed). split='test'
rejection is tested on every GET endpoint without needing real data (the
guard fires in _split_param() before any filesystem access)."""

import pytest

from reporting import frame_mapping
from webapp_api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_index_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Visualiseur de développement embryonnaire" in resp.data.decode("utf-8")


@pytest.mark.parametrize("path", [
    "/videos",
    "/videos/Patient_1",
    "/videos/Patient_1/timeline",
    "/videos/Patient_1/window/0",
    "/videos/Patient_1/window/0/frame",
])
def test_every_get_endpoint_rejects_test_split_with_403(client, path):
    resp = client.get(path, query_string={"split": "test"})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "test_locked"


def test_chat_rejects_test_split_with_403(client):
    resp = client.post("/chat", json={"question": "q", "split": "test"})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "test_locked"


def test_split_defaults_to_val_not_test(monkeypatch, client):
    captured = {}

    def fake_list_videos(split):
        captured["split"] = split
        return []

    monkeypatch.setattr("webapp_api.app.trajectory_service.list_videos", fake_list_videos)
    resp = client.get("/videos")
    assert resp.status_code == 200
    assert captured["split"] == "val"


def test_videos_endpoint_returns_split_and_list(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.trajectory_service.list_videos", lambda split: ["Patient_1", "Patient_2"])
    resp = client.get("/videos")
    assert resp.status_code == 200
    assert resp.get_json() == {"split": "val", "videos": ["Patient_1", "Patient_2"]}


def test_unknown_video_returns_404_on_detail(monkeypatch, client):
    def raise_not_found(video_name, split):
        raise KeyError(f"video_name={video_name!r} not found")

    monkeypatch.setattr("webapp_api.app.trajectory_service.get_windows", raise_not_found)
    resp = client.get("/videos/DoesNotExist")
    assert resp.status_code == 404


def test_video_detail_shape(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.trajectory_service.get_windows", lambda v, s: [0, 8, 16])
    resp = client.get("/videos/Patient_1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"video_name": "Patient_1", "split": "val", "n_windows": 3,
                     "first_window": 0, "last_window": 16}


def test_video_detail_handles_zero_windows(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.trajectory_service.get_windows", lambda v, s: [])
    resp = client.get("/videos/Patient_1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_windows"] == 0
    assert body["first_window"] is None and body["last_window"] is None


def _fake_record(window_start, phase, probability):
    class _Record:
        def to_dict(self):
            return {
                "sample": {"video_name": "Patient_1"},
                "window": {"window_start": window_start, "window_start_time": float(window_start),
                           "window_end_time": float(window_start) + 1.0},
                "current_state": {"current_phase": phase, "current_phase_index": 0,
                                   "phase_probability": probability, "phase_probabilities": {phase: probability},
                                   "entropy": 0.1},
                "next_phase": {"next_phase_distribution": {}, "most_likely_next_phase": phase,
                                "next_phase_probability": probability},
                "duration": {"duration_available": True, "expected_duration": 5.0,
                             "duration_quantiles": {"0.5": 5}, "duration_unit": "windows"},
                "model": {"model_name": "semi_hmm", "model_version": "v1"},
                "ground_truth": {"ground_truth_phase": None, "consistency_flag": None},
                "warnings": [],
            }
    return _Record()


def test_window_endpoint_returns_inference_record(monkeypatch, client):
    monkeypatch.setattr(
        "webapp_api.app.inference_service.infer",
        lambda video, split, window: _fake_record(window, "t6", 0.5),
    )
    resp = client.get("/videos/Patient_1/window/0")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["current_state"]["current_phase"] == "t6"
    assert body["window"]["window_start"] == 0


def test_unknown_window_returns_404(monkeypatch, client):
    from reporting.inference_service import WindowNotFoundError

    def raise_not_found(video, split, window):
        raise WindowNotFoundError(f"window_start={window} not found")

    monkeypatch.setattr("webapp_api.app.inference_service.infer", raise_not_found)
    resp = client.get("/videos/Patient_1/window/9999")
    assert resp.status_code == 404


# --- /videos/<id>/window/<w>/frame -------------------------------------

def test_frame_endpoint_returns_real_jpeg_bytes(monkeypatch, client, tmp_path):
    jpeg_path = tmp_path / "Patient_1_Image_107.jpeg"
    jpeg_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    monkeypatch.setattr(
        "webapp_api.app.frame_mapping.window_to_frame_path",
        lambda video, split, window, which="last": jpeg_path,
    )
    resp = client.get("/videos/Patient_1/window/0/frame")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == jpeg_path.read_bytes()


def test_frame_endpoint_passes_which_query_param_through(monkeypatch, client, tmp_path):
    jpeg_path = tmp_path / "Patient_1_Image_100.jpeg"
    jpeg_path.write_bytes(b"fake")
    captured = {}

    def fake_window_to_frame_path(video, split, window, which="last"):
        captured.update(video=video, split=split, window=window, which=which)
        return jpeg_path

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", fake_window_to_frame_path)
    resp = client.get("/videos/Patient_1/window/0/frame", query_string={"which": "first"})
    assert resp.status_code == 200
    assert captured == {"video": "Patient_1", "split": "val", "window": 0, "which": "first"}


def test_frame_endpoint_unknown_video_returns_404(monkeypatch, client):
    def raise_not_found(video, split, window, which="last"):
        raise KeyError(f"video_name={video!r} not found")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_not_found)
    resp = client.get("/videos/DoesNotExist/window/0/frame")
    assert resp.status_code == 404


def test_frame_endpoint_invalid_window_returns_404(monkeypatch, client):
    def raise_mapping_error(video, split, window, which="last"):
        raise frame_mapping.FrameMappingError(f"window_start={window} is not a real window")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_mapping_error)
    resp = client.get("/videos/Patient_1/window/9999/frame")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "frame_not_found"


def test_frame_endpoint_missing_annotations_returns_404(monkeypatch, client):
    def raise_annotations_missing(video, split, window, which="last"):
        raise frame_mapping.AnnotationsNotFoundError("no annotation file")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_annotations_missing)
    resp = client.get("/videos/Patient_1/window/0/frame")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "frame_not_found"


def test_frame_endpoint_missing_jpeg_on_disk_returns_404(monkeypatch, client):
    def raise_file_missing(video, split, window, which="last"):
        raise frame_mapping.FrameFileNotFoundError("no jpeg on disk")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_file_missing)
    resp = client.get("/videos/Patient_1/window/0/frame")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "frame_not_found"


def test_frame_endpoint_invalid_which_returns_400(monkeypatch, client):
    def raise_value_error(video, split, window, which="last"):
        raise ValueError(f"which must be 'first' or 'last', got {which!r}.")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_value_error)
    resp = client.get("/videos/Patient_1/window/0/frame", query_string={"which": "middle"})
    assert resp.status_code == 400


def test_frame_endpoint_non_integer_window_returns_404(client):
    # Flask's <int:window_start> route converter rejects non-integer path
    # segments before the view function ever runs -- no route matches.
    resp = client.get("/videos/Patient_1/window/not-a-number/frame")
    assert resp.status_code == 404


def test_frame_endpoint_path_traversal_attempt_in_video_id_returns_404(monkeypatch, client):
    # video_id="..": window_to_frame_path's real implementation would first
    # call trajectory_service.get_windows("..", split), which raises
    # KeyError for any name not in the embeddings-backed video index --
    # simulated here directly since window_to_frame_path is monkeypatched
    # for every other test in this file. No filesystem path is ever built
    # from user input before that lookup succeeds.
    def raise_unknown_video(video, split, window, which="last"):
        raise KeyError(f"video_name={video!r} not found in split={split!r}.")

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", raise_unknown_video)
    resp = client.get("/videos/../window/0/frame")
    assert resp.status_code == 404


def test_frame_endpoint_calls_frame_mapping_window_to_frame_path(monkeypatch, client, tmp_path):
    # Cohérence avec frame_mapping.py: proves the endpoint delegates to
    # frame_mapping's real function name (not a second, parallel mapping
    # implementation inside webapp_api).
    jpeg_path = tmp_path / "f.jpeg"
    jpeg_path.write_bytes(b"fake")
    called = {"n": 0}

    def fake(video, split, window, which="last"):
        called["n"] += 1
        return jpeg_path

    monkeypatch.setattr("webapp_api.app.frame_mapping.window_to_frame_path", fake)
    resp = client.get("/videos/Patient_1/window/0/frame")
    assert resp.status_code == 200
    assert called["n"] == 1


def test_timeline_groups_consecutive_identical_phases_into_segments(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.trajectory_service.get_windows", lambda v, s: [0, 8, 16, 24])
    fake_phases = {0: "t2", 8: "t2", 16: "t3", 24: "t3"}

    monkeypatch.setattr(
        "webapp_api.app.inference_service.infer",
        lambda video, split, window: _fake_record(window, fake_phases[window], 0.9),
    )
    resp = client.get("/videos/Patient_1/timeline")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_windows"] == 4
    assert len(body["segments"]) == 2
    seg0, seg1 = body["segments"]
    assert seg0["phase"] == "t2" and seg0["start_window"] == 0 and seg0["end_window"] == 8 and seg0["n_windows"] == 2
    assert seg1["phase"] == "t3" and seg1["start_window"] == 16 and seg1["end_window"] == 24 and seg1["n_windows"] == 2
    assert seg0["mean_probability"] == pytest.approx(0.9)


def test_timeline_empty_video_returns_no_segments(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.trajectory_service.get_windows", lambda v, s: [])
    resp = client.get("/videos/Patient_1/timeline")
    assert resp.status_code == 200
    assert resp.get_json()["segments"] == []


# --- /chat -------------------------------------------------------------

def _fake_answer_question_result(text="La phase est t6.", grounded=True, sources=None, warnings=None,
                                  used_tools=None, confidence="high", category="DYNAMIC_DATA"):
    return {
        "question": "q",
        "route": {"category": category},
        "tool_plan": {"called": used_tools or [], "not_called": []},
        "context": {"document_context": sources or [], "dynamic_context": {}, "warnings": warnings or []},
        "response": {
            "text": text, "answer": text, "provider": "ollama", "grounded": grounded,
            "warnings": warnings or [], "sources": sources or [], "dynamic_data": [],
            "used_tools": used_tools or [], "confidence": confidence,
        },
    }


def test_chat_empty_question_returns_400(client):
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code == 400


def test_chat_missing_question_returns_400(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 400


def test_chat_calls_orchestrator_with_explicit_context(monkeypatch, client):
    captured = {}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        captured.update(question=question, video_id=video_id, window=window, split=split)
        return _fake_answer_question_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={
        "question": "Pourquoi cette phase ?", "video_id": "Patient_319", "window": 526, "split": "val",
    })
    assert resp.status_code == 200
    assert captured == {"question": "Pourquoi cette phase ?", "video_id": "Patient_319", "window": 526, "split": "val"}


def test_chat_response_shape(monkeypatch, client):
    monkeypatch.setattr(
        "webapp_api.app.orchestrator_module.answer_question",
        lambda question, video_id=None, window=None, split="val", llm=None, **kwargs: _fake_answer_question_result(
            text="La phase est t6.", grounded=True,
            sources=[{"source": "docs/X.md", "section": "1", "status": "current"}],
            warnings=["some warning"], used_tools=["get_current_inference"], confidence="high",
            category="DYNAMIC_DATA",
        ),
    )
    resp = client.post("/chat", json={"question": "Quelle est la phase actuelle ?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {
        "question_id": None, "question": "Quelle est la phase actuelle ?",
        "answer": "La phase est t6.", "grounded": True, "confidence": "high",
        "sources": [{"source": "docs/X.md", "section": "1", "status": "current"}],
        "warnings": ["some warning"], "tools_called": ["get_current_inference"], "route": "DYNAMIC_DATA",
    }


def test_chat_without_video_context_still_works(monkeypatch, client):
    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        assert video_id is None and window is None
        return _fake_answer_question_result(text="Je ne dispose pas de suffisamment d'informations.",
                                             grounded=False, category="UNKNOWN")

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={"question": "Quelle est la meteo a Brest ?"})
    assert resp.status_code == 200
    assert resp.get_json()["route"] == "UNKNOWN"


# --- /chat context propagation (Phase 7.9) ------------------------------

def test_chat_window_zero_is_passed_through_not_treated_as_no_context(monkeypatch, client):
    # window=0 is a real, common value (a video's first window) -- must
    # not be conflated with "no window selected" (window=None) anywhere
    # along the falsy-value chain from JSON body to orchestrator call.
    captured = {}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        captured.update(video_id=video_id, window=window)
        return _fake_answer_question_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={"question": "q", "video_id": "Patient_319", "window": 0})
    assert resp.status_code == 200
    assert captured == {"video_id": "Patient_319", "window": 0}


def test_chat_context_follows_window_change_across_requests(monkeypatch, client):
    # Simulates the frontend's behaviour when the user changes the
    # selected window between two questions: each request must carry
    # exactly the window it was sent with, never a value left over from
    # the previous request (the BFF itself is stateless per-request; this
    # proves it, rather than assuming it).
    captured_calls = []

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        captured_calls.append({"video_id": video_id, "window": window})
        return _fake_answer_question_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    client.post("/chat", json={"question": "q1", "video_id": "Patient_319", "window": 100, "split": "val"})
    client.post("/chat", json={"question": "q2", "video_id": "Patient_319", "window": 150, "split": "val"})
    assert captured_calls == [
        {"video_id": "Patient_319", "window": 100},
        {"video_id": "Patient_319", "window": 150},
    ]


def test_chat_documentary_question_response_shape(monkeypatch, client):
    # "Qu'est-ce qu'un Semi-HMM ?" -- RAG-only, no video/window context
    # needed; Router already classifies this DOCUMENTARY (Training/orchestrator/router.py),
    # this test just proves the endpoint doesn't mangle that category's shape.
    monkeypatch.setattr(
        "webapp_api.app.orchestrator_module.answer_question",
        lambda question, video_id=None, window=None, split="val", llm=None, **kwargs: _fake_answer_question_result(
            text="Un Semi-HMM est un modele de Markov cache semi-markovien...",
            grounded=True,
            sources=[{"source": "docs/HMM_RESEARCH_PLAN.md", "section": "2", "status": "current"}],
            used_tools=[], category="DOCUMENTARY",
        ),
    )
    resp = client.post("/chat", json={"question": "Qu'est-ce qu'un Semi-HMM ?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["route"] == "DOCUMENTARY"
    assert body["sources"] == [{"source": "docs/HMM_RESEARCH_PLAN.md", "section": "2", "status": "current"}]
    assert body["tools_called"] == []


def test_chat_hybrid_question_response_shape(monkeypatch, client):
    # "Que signifie cette probabilite ?" -- both a live number (Reporting
    # API) and a documented explanation (RAG); Router classifies HYBRID.
    monkeypatch.setattr(
        "webapp_api.app.orchestrator_module.answer_question",
        lambda question, video_id=None, window=None, split="val", llm=None, **kwargs: _fake_answer_question_result(
            text="La probabilite 0.72 signifie...",
            grounded=True,
            sources=[{"source": "docs/HANDOFF.md", "section": "4", "status": "current"}],
            used_tools=["get_current_inference"], category="HYBRID",
        ),
    )
    resp = client.post("/chat", json={
        "question": "Que signifie cette probabilite ?", "video_id": "Patient_319", "window": 120,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["route"] == "HYBRID"
    assert body["tools_called"] == ["get_current_inference"]
    assert len(body["sources"]) == 1


def test_chat_never_reaches_a_real_llm_provider_in_tests(monkeypatch, client):
    # The module-level _CHAT_PROVIDER is a real OllamaLLMProvider -- this
    # test proves the endpoint's own orchestrator call is what's
    # monkeypatched (so no test here ever depends on a running Ollama
    # server), not that the provider object itself is mocked.
    called = {"n": 0}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        called["n"] += 1
        assert llm is not None  # the real provider IS passed through, just never invoked here
        return _fake_answer_question_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    client.post("/chat", json={"question": "q"})
    assert called["n"] == 1

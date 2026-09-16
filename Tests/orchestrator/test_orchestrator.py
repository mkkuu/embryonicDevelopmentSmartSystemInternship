"""
Pure end-to-end tests for orchestrator.answer_question(), with every
tools.* call monkeypatched to a fake -- never touches torch/chromadb/a
real model or index, so these always run in the normal `pytest Tests/`
suite. The real-deps path (actual Reporting API / RAG calls) is covered
separately by Tests/orchestrator/test_tools_reporting.py and
test_tools_rag.py, both skip-guarded.
"""

import pytest

from orchestrator import tools
from orchestrator.orchestrator import ALL_TOOLS, answer_question, execute_plan, plan_tools
from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN, classify


_GIH_CALLS: list = []


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    def fake_retrieve_documents(query, top_k=5, **kwargs):
        return [{
            "chunk_id": "doc::0001", "content": "fake chunk content", "distance": 0.1, "score": 0.9,
            "metadata": {"source": "docs/FAKE.md", "section": "1", "status": "authoritative", "authoritative": True},
        }]

    def fake_get_current_inference(video_id, window, split="val"):
        return {
            "sample": {"sample_id": f"{video_id}#{split}"},
            "current_state": {"current_phase": "t6", "phase_probability": 0.4},
            "model": {"model_version": "fake_v1"},
        }

    def fake_get_trajectory(video_id, split="val"):
        return {"sample": {"sample_id": f"{video_id}#{split}"}, "n_windows": 3, "model": {"model_version": "fake_v1"}}

    def fake_get_model_metadata():
        return {"model_name": "semi_hmm", "model_version": "fake_v1", "model_configuration": {}, "model_source": "x"}

    def fake_get_transition_events(video_id, split="val", window=None):
        return {
            "video_id": video_id, "split": split, "window": window, "n_transitions": 1,
            "transitions": [{
                "from_phase": "t5", "to_phase": "t7", "observed": True,
                "provenance": "observed_annotation", "window_start": 100, "window_end": 101,
                "phase_distance": 2, "is_skip": True,
            }],
        }

    def fake_get_inference_history(video_id, center_window, before=5, after=5, split="val",
                                   top_k_probabilities=None):
        # ETAPE 4: record what the orchestrator actually forwarded, so a test
        # can assert the prompt-budget policy reaches the tool.
        _GIH_CALLS.append({"video_id": video_id, "center_window": center_window,
                           "before": before, "after": after, "split": split,
                           "top_k_probabilities": top_k_probabilities})
        lo, hi = center_window - before, center_window + after
        entries = [{
            "window_start": w, "offset_from_center": w - center_window, "is_center": w == center_window,
            "window_start_time": None, "window_end_time": None, "time_available": False,
            "model_derived": {
                "current_phase": "t6", "current_phase_index": 5, "phase_probability": 0.5,
                "phase_probabilities": {"t6": 0.5, "t7": 0.5}, "entropy": 0.69,
                "most_likely_next_phase": "t7", "next_phase_probability": 0.6,
                "next_phase_distribution": {"t6": 0.4, "t7": 0.6},
            },
            "observed": {"ground_truth_phase": "t6", "consistency_flag": 0},
        } for w in range(lo, hi + 1)]
        return {
            "sample": {"sample_id": f"{video_id}#{split}", "video_name": video_id},
            "center_window": center_window, "requested_before": before, "requested_after": after,
            "n_windows_before": before, "n_windows_after": after, "n_windows": len(entries),
            "window_starts": [e["window_start"] for e in entries], "entries": entries,
            "model": {"model_version": "fake_v1"}, "warnings": [],
        }

    monkeypatch.setattr(tools, "retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(tools, "get_current_inference", fake_get_current_inference)
    monkeypatch.setattr(tools, "get_trajectory", fake_get_trajectory)
    monkeypatch.setattr(tools, "get_model_metadata", fake_get_model_metadata)
    monkeypatch.setattr(tools, "get_transition_events", fake_get_transition_events)
    monkeypatch.setattr(tools, "get_inference_history", fake_get_inference_history)
    _GIH_CALLS.clear()


def test_documentary_question_only_calls_document_tools():
    out = answer_question("Qu'est-ce que le Semi-HMM ?")
    assert out["route"]["category"] == DOCUMENTARY
    assert out["tool_plan"]["called"] == ["retrieve_documents"]
    assert len(out["context"]["document_context"]) == 1
    assert out["context"]["dynamic_context"] == {}


def test_dynamic_question_with_video_and_window_calls_get_current_inference():
    out = answer_question("Quelle est la phase actuelle ?", video_id="Patient_1", window=0)
    assert out["route"]["category"] == DYNAMIC_DATA
    assert out["tool_plan"]["called"] == ["get_current_inference"]
    assert out["context"]["dynamic_context"]["get_current_inference"]["current_state"]["current_phase"] == "t6"
    assert out["context"]["document_context"] == []


def test_dynamic_question_without_video_id_calls_nothing_and_warns_honestly():
    out = answer_question("Quelle est la phase actuelle ?")
    assert out["tool_plan"]["called"] == []
    reasons = [d["reason"] for d in out["tool_plan"]["not_called"] if d["tool"] == "get_current_inference"]
    assert reasons and "session/context resolution not implemented" in reasons[0]
    assert out["context"]["dynamic_context"] == {}
    assert out["context"]["document_context"] == []


def test_trajectory_question_calls_get_trajectory_not_get_current_inference():
    out = answer_question("Montre-moi la trajectoire complete de cette video.", video_id="Patient_1")
    assert "get_trajectory" in out["tool_plan"]["called"]
    assert "get_current_inference" not in out["tool_plan"]["called"]


def test_hybrid_question_calls_both_document_and_reporting_tools():
    out = answer_question("Pourquoi le modele predit-il t6 pour cette fenetre ?",
                           video_id="Patient_1", window=0)
    assert out["route"]["category"] == HYBRID
    assert "get_current_inference" in out["tool_plan"]["called"]
    assert "retrieve_documents" in out["tool_plan"]["called"]
    assert out["context"]["dynamic_context"] and out["context"]["document_context"]


def test_unknown_question_calls_no_tool_and_null_provider_says_so():
    out = answer_question("Quel temps fait-il a Brest ?")
    assert out["route"]["category"] == UNKNOWN
    assert out["tool_plan"]["called"] == []
    assert out["response"]["provider"] == "null"
    assert out["response"]["grounded"] is False


def test_a_tool_error_is_recorded_as_a_warning_not_an_exception(monkeypatch):
    def broken_retrieve(query, top_k=5, **kwargs):
        raise tools.ToolUnavailableError("RAG index unavailable in this environment")

    monkeypatch.setattr(tools, "retrieve_documents", broken_retrieve)
    out = answer_question("Qu'est-ce que le Semi-HMM ?")
    assert out["context"]["document_context"] == []
    assert any("RAG index unavailable" in w for w in out["context"]["warnings"])


def test_answer_question_result_is_json_serializable():
    import json

    out = answer_question("Pourquoi le modele predit-il t6 pour cette fenetre ?",
                           video_id="Patient_1", window=0)
    json.dumps(out)  # must not raise


def test_plan_tools_and_execute_plan_agree_on_called_set():
    route = classify("Qu'est-ce que le Semi-HMM ?")
    plan = plan_tools("Qu'est-ce que le Semi-HMM ?", route)
    results = execute_plan(plan, "Qu'est-ce que le Semi-HMM ?", None, None, "val")
    assert set(results.keys()) == set(plan.called)


def test_transition_question_with_video_calls_get_transition_events():
    out = answer_question("Quelle transition de phase est observée ici ?",
                           video_id="Patient_319", window=100)
    assert "get_transition_events" in out["tool_plan"]["called"]
    assert out["context"]["dynamic_context"]["get_transition_events"]["n_transitions"] == 1
    assert out["context"]["dynamic_context"]["get_transition_events"]["transitions"][0]["observed"] is True
    assert out["context"]["dynamic_context"]["get_transition_events"]["transitions"][0]["is_skip"] is True


def test_skip_question_with_video_calls_get_transition_events():
    out = answer_question("Y a-t-il un saut de phase ?", video_id="Patient_319", window=100)
    assert "get_transition_events" in out["tool_plan"]["called"]


def test_transition_question_without_video_id_skips_honestly():
    out = answer_question("Quelle transition de phase est observée ici ?")
    assert "get_transition_events" not in out["tool_plan"]["called"]
    reasons = [d["reason"] for d in out["tool_plan"]["not_called"] if d["tool"] == "get_transition_events"]
    assert reasons and "no video_id" in reasons[0]


def test_non_transition_dynamic_question_does_not_call_get_transition_events():
    out = answer_question("Quelle est la phase actuelle ?", video_id="Patient_319", window=0)
    assert "get_current_inference" in out["tool_plan"]["called"]
    assert "get_transition_events" not in out["tool_plan"]["called"]


# --- Transition-events SCOPE fix (docs/RAG_HISTORY_SCOPE_FIX.md) --
# a history-shaped question ("quelle transition a precede...") must get
# get_transition_events(window=None) -- the FULL, unfiltered transition
# history -- instead of the current-window-scoped call every other
# transition/skip question correctly keeps.

def test_history_question_uses_full_history_scope():
    route = classify("Quelle transition de phase a précédé celle observée à cette fenêtre ?")
    plan = plan_tools("Quelle transition de phase a précédé celle observée à cette fenêtre ?",
                       route, video_id="Patient_319", window=156)
    assert "get_transition_events" in plan.called
    assert plan.called.count("get_transition_events") == 1  # no unexpected duplicate
    assert plan.transition_events_scope == "full_history"


def test_history_question_calls_get_transition_events_with_window_none():
    out = answer_question("Quelle transition de phase a précédé celle observée à cette fenêtre ?",
                           video_id="Patient_319", window=156)
    # fake_get_transition_events (fixture above) echoes back the window it received.
    assert out["context"]["dynamic_context"]["get_transition_events"]["window"] is None


def test_current_transition_question_keeps_current_window_scope():
    route = classify("Quelle transition de phase est observée ici ?")
    plan = plan_tools("Quelle transition de phase est observée ici ?",
                       route, video_id="Patient_319", window=156)
    assert plan.transition_events_scope == "current_window"

    out = answer_question("Quelle transition de phase est observée ici ?",
                           video_id="Patient_319", window=156)
    assert out["context"]["dynamic_context"]["get_transition_events"]["window"] == 156


def test_skip_question_also_keeps_current_window_scope():
    """Regression guard: only the NEW history triggers change scope --
    the pre-existing skip-transition question must be entirely unaffected."""
    out = answer_question("Y a-t-il un saut de phase ici ?", video_id="Patient_319", window=156)
    assert out["context"]["dynamic_context"]["get_transition_events"]["window"] == 156


def test_scope_defaults_to_current_window_when_tool_not_called():
    """Regression guard: the new field must never spuriously flip scope
    for a question that doesn't call get_transition_events at all."""
    route = classify("Quelle est la phase actuelle ?")
    plan = plan_tools("Quelle est la phase actuelle ?", route, video_id="Patient_319", window=0)
    assert "get_transition_events" not in plan.called
    assert plan.transition_events_scope == "current_window"


def test_full_history_scope_actually_retrieves_more_than_current_window(monkeypatch):
    """Proves the mechanism, not just the flag: a real-shaped tool that
    returns MORE transitions for window=None than for a specific window
    (matching Training/reporting/transition_events.py's own real
    behavior) is reached correctly when scope=full_history."""
    def scope_sensitive_get_transition_events(video_id, split="val", window=None):
        all_transitions = [
            {"from_phase": "tPB2", "to_phase": "tPNa", "observed": True,
             "provenance": "observed_annotation", "window_start": 8, "window_end": 9,
             "phase_distance": 1, "is_skip": False},
            {"from_phase": "t4", "to_phase": "t6", "observed": True,
             "provenance": "observed_annotation", "window_start": 156, "window_end": 157,
             "phase_distance": 2, "is_skip": True},
        ]
        if window is None:
            transitions = all_transitions
        else:
            transitions = [t for t in all_transitions if t["window_start"] <= window <= t["window_end"]]
        return {"video_id": video_id, "split": split, "window": window,
                "n_transitions": len(transitions), "transitions": transitions}

    monkeypatch.setattr(tools, "get_transition_events", scope_sensitive_get_transition_events)

    out_history = answer_question("Quelle transition de phase a précédé celle observée à cette fenêtre ?",
                                   video_id="Patient_319", window=156)
    assert out_history["context"]["dynamic_context"]["get_transition_events"]["n_transitions"] == 2

    out_current = answer_question("Quelle transition de phase est observée ici ?",
                                   video_id="Patient_319", window=156)
    assert out_current["context"]["dynamic_context"]["get_transition_events"]["n_transitions"] == 1


def test_compare_observed_transition_with_model_prediction_calls_both_tools():
    out = answer_question("Compare la transition observée avec la prédiction du modèle.",
                           video_id="Patient_319", window=100)
    assert "get_transition_events" in out["tool_plan"]["called"]
    assert "get_current_inference" in out["tool_plan"]["called"]
    dyn = out["context"]["dynamic_context"]
    assert dyn["get_transition_events"]["transitions"][0]["provenance"] == "observed_annotation"
    assert "provenance" not in dyn["get_current_inference"]["current_state"]


# --- Multi-window / G1: get_inference_history integration ------------------

def test_multi_window_question_plans_and_executes_get_inference_history():
    out = answer_question(
        "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?",
        video_id="Patient_319", window=156,
    )
    assert "get_inference_history" in out["tool_plan"]["called"]
    hist = out["context"]["dynamic_context"]["get_inference_history"]
    assert hist["n_windows"] == 11  # default before=5 + center + after=5
    starts = hist["window_starts"]
    assert starts == sorted(starts)
    assert 156 in starts
    # provenance stays separated inside every entry
    for e in hist["entries"]:
        assert "phase_probabilities" in e["model_derived"]
        assert set(e["observed"]) == {"ground_truth_phase", "consistency_flag"}


def test_single_window_question_does_not_plan_get_inference_history():
    out = answer_question("Quelle est la phase actuelle ?", video_id="Patient_319", window=156)
    assert "get_inference_history" not in out["tool_plan"]["called"]
    assert "get_current_inference" in out["tool_plan"]["called"]


def test_inference_history_needs_a_window_to_anchor_the_band():
    out = answer_question(
        "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?",
        video_id="Patient_319",  # no window
    )
    assert "get_inference_history" not in out["tool_plan"]["called"]
    reasons = [d["reason"] for d in out["tool_plan"]["not_called"] if d["tool"] == "get_inference_history"]
    assert reasons and "no video_id/window" in reasons[0]


def test_elapsed_duration_question_co_plans_trajectory_with_current_inference():
    out = answer_question(
        "Depuis combien de temps l'embryon se trouve-t-il dans la phase actuelle ?",
        video_id="Patient_319", window=156,
    )
    assert out["tool_plan"]["called"] == ["get_current_inference", "get_trajectory"]
    assert "get_current_inference" in out["context"]["dynamic_context"]
    assert "get_trajectory" in out["context"]["dynamic_context"]


def test_plan_and_execute_still_agree_with_the_new_tools():
    """Every planned TOOL is executed and nothing else is -- with the one
    documented, non-tool exception: `series_analysis`, the deterministic
    arithmetic execute_plan() derives from get_inference_history's own output
    (2026-09-02 context restructuring). It is not in ALL_TOOLS, is never
    planned, and can only appear when the history tool actually succeeded."""
    q = "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?"
    route = classify(q)
    plan = plan_tools(q, route, video_id="Patient_319", window=156)
    results = execute_plan(plan, q, "Patient_319", 156, "val")
    assert set(results.keys()) - {"series_analysis"} == set(plan.called)
    assert "series_analysis" not in plan.called
    assert "series_analysis" not in ALL_TOOLS


# --- ETAPE 4: the orchestrator's prompt-budget policy reaches the tool ---

def test_orchestrator_forwards_a_top_k_probabilities_budget_to_the_history_tool():
    from orchestrator import orchestrator as orch
    answer_question("A quel moment les probabilites des deux phases commencent-elles "
                    "a se rapprocher ?", video_id="Patient_1", window=10)
    assert _GIH_CALLS, "get_inference_history was not called"
    assert _GIH_CALLS[-1]["top_k_probabilities"] == orch._INFERENCE_HISTORY_TOP_K_PROBABILITIES
    assert _GIH_CALLS[-1]["top_k_probabilities"] is not None, \
        "a None budget would send the full 15-entry distributions to the LLM"


def test_orchestrator_history_budget_is_a_small_positive_int():
    from orchestrator import orchestrator as orch
    k = orch._INFERENCE_HISTORY_TOP_K_PROBABILITIES
    assert isinstance(k, int) and 1 <= k <= 6, \
        "the band must stay compact enough for the deployment's 4096-token ctx-size"


def test_orchestrator_still_passes_the_configured_band_unchanged():
    # conservation guard: adding the budget must not disturb before/after.
    from orchestrator import orchestrator as orch
    answer_question("Comment les probabilites evoluent-elles dans les fenetres "
                    "precedant et suivant la transition ?", video_id="Patient_1", window=10)
    assert _GIH_CALLS[-1]["before"] == orch._INFERENCE_HISTORY_BEFORE
    assert _GIH_CALLS[-1]["after"] == orch._INFERENCE_HISTORY_AFTER


# --- Q9 (2026-09-02g): the predicted-sequence question now gets the series ---

def test_q9_predicted_sequence_question_plans_get_inference_history():
    """Q9 asks about a skip/regression in the PREDICTED sequence -- a
    multi-window property. Before this fix its context held one model window
    plus get_transition_events' OBSERVED transition, whose only skip belongs
    to the annotation."""
    out = answer_question(
        "Y a-t-il un saut de phase ou une régression dans la séquence prédite ?",
        video_id="Patient_319", window=156,
    )
    assert "get_inference_history" in out["tool_plan"]["called"]
    assert "get_inference_history" in out["context"]["dynamic_context"]


def test_q9_context_carries_the_deterministic_series_analysis():
    out = answer_question(
        "Y a-t-il un saut de phase ou une régression dans la séquence prédite ?",
        video_id="Patient_319", window=156,
    )
    analysis = out["context"]["dynamic_context"]["series_analysis"]
    assert analysis["provenance"] == "derived_deterministic"
    assert "n_phase_skips" in analysis["stability"]
    assert "n_phase_regressions" in analysis["stability"]
    assert set(analysis["sequences"]) >= {"model_derived_phase_sequence",
                                          "observed_annotation_phase_sequence"}


def test_q9_route_category_is_unchanged_by_the_tool_planning_fix():
    """The fix is in TOOL PLANNING only -- router.classify() is untouched."""
    assert classify("Y a-t-il un saut de phase ou une régression dans la séquence prédite ?") \
        .category == DYNAMIC_DATA


def test_the_q9_trigger_does_not_fire_on_any_other_benchmark_question():
    from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
    from orchestrator.router import _normalize
    matched = [q.question_id for q in FIXED_QUESTIONS if "sequence predite" in _normalize(q.question)]
    assert matched == ["Q9"], f"trigger collides with {matched}"


def test_single_window_questions_still_do_not_plan_the_history_tool():
    """The Q9 token must not widen the gate for questions that genuinely only
    need one window."""
    out = answer_question("Quelle est la phase actuelle de cette vidéo ?",
                          video_id="Patient_319", window=156)
    assert "get_inference_history" not in out["tool_plan"]["called"]

"""
Task's own "Evaluation avant LLM" requirement (>=15 questions, expected
category, route selected) -- an LLM-free evaluation of the router alone,
against the canonical shared question set in
Training/experiments/llm_regression/eval_questions.py (also used by
run_orchestration_evaluation.py for the real-system run on the GPU
server). This file is what actually runs in the normal `pytest Tests/`
suite; the real-system run is a separate, heavier, skip-guarded/
manually-invoked step (see run_orchestration_evaluation.py's own
docstring), not duplicated here.
"""

from experiments.llm_regression.eval_questions import EVAL_QUESTIONS
from orchestrator.router import classify
from orchestrator.orchestrator import plan_tools


def test_eval_question_set_has_at_least_15_questions():
    assert len(EVAL_QUESTIONS) >= 15


def test_eval_question_set_covers_all_four_categories():
    from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN

    expected_categories = {eq.expected_category for eq in EVAL_QUESTIONS}
    assert expected_categories == {DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN}


def test_every_eval_question_routes_to_its_expected_category():
    mismatches = []
    for eq in EVAL_QUESTIONS:
        got = classify(eq.question).category
        if got != eq.expected_category:
            mismatches.append((eq.question, eq.expected_category, got))
    assert not mismatches, (
        "router mismatches (question, expected, got):\n" +
        "\n".join(f"  {q!r}: expected {exp}, got {got}" for q, exp, got in mismatches)
    )


def test_unknown_questions_never_select_any_tool():
    from orchestrator.router import UNKNOWN

    for eq in EVAL_QUESTIONS:
        if eq.expected_category != UNKNOWN:
            continue
        route = classify(eq.question)
        plan = plan_tools(eq.question, route, video_id=eq.video_id, window=eq.window)
        assert plan.called == [], f"UNKNOWN question unexpectedly selected tools: {eq.question!r} -> {plan.called}"


def test_documentary_questions_never_call_reporting_api_tools():
    from orchestrator.router import DOCUMENTARY

    reporting_tools = {"get_current_inference", "get_trajectory", "get_model_metadata"}
    for eq in EVAL_QUESTIONS:
        if eq.expected_category != DOCUMENTARY:
            continue
        route = classify(eq.question)
        plan = plan_tools(eq.question, route, video_id=eq.video_id, window=eq.window)
        called_reporting = reporting_tools & set(plan.called)
        assert not called_reporting, (
            f"DOCUMENTARY question {eq.question!r} unexpectedly called Reporting API tool(s): "
            f"{called_reporting} -- RAG / Reporting API separation violated."
        )


def test_dynamic_questions_never_call_retrieve_documents():
    from orchestrator.router import DYNAMIC_DATA

    for eq in EVAL_QUESTIONS:
        if eq.expected_category != DYNAMIC_DATA:
            continue
        route = classify(eq.question)
        plan = plan_tools(eq.question, route, video_id=eq.video_id, window=eq.window)
        assert "retrieve_documents" not in plan.called, (
            f"DYNAMIC_DATA question {eq.question!r} unexpectedly called retrieve_documents "
            f"-- RAG / Reporting API separation violated."
        )


def test_hybrid_questions_call_both_a_document_and_a_reporting_tool_when_context_given():
    from orchestrator.router import HYBRID

    document_tools = {"retrieve_documents", "get_phase_information"}
    reporting_tools = {"get_current_inference", "get_trajectory", "get_model_metadata"}
    for eq in EVAL_QUESTIONS:
        if eq.expected_category != HYBRID:
            continue
        route = classify(eq.question)
        plan = plan_tools(eq.question, route, video_id=eq.video_id, window=eq.window)
        assert document_tools & set(plan.called), f"HYBRID question {eq.question!r} called no document tool"
        if eq.video_id is not None:
            assert reporting_tools & set(plan.called), (
                f"HYBRID question {eq.question!r} had video context but called no Reporting API tool"
            )

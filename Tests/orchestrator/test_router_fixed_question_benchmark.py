"""
Router / tool-plan coverage for the DEFINITIVE 15-question Fixed-Question
Functional Benchmark (`Training/orchestrator/fixed_question_benchmark.py`,
`docs/RAG_FIXED_QUESTION_BENCHMARK.md`) -- the 2026-09-01 list that fully
replaces every earlier question set.

Pure: no filesystem, no torch, no chromadb -- `router.classify()`,
`orchestrator.plan_tools()` and `router._normalize()` are pure functions.

What this guards:
  - the intended post-fix route for each of Q1..Q15;
  - the tool plan for the questions whose routing was changed this
    session (Q1, Q3, Q5-Q15) actually plans the Reporting-API / RAG tool
    the question needs (not blind keyword matching);
  - the U+2019 typographic-apostrophe fold in router._normalize() (the
    benchmark wording uses it; Q7 only routes HYBRID because of it);
  - zero regression on the canonical 18-question router set
    (`eval_questions.py`) and the deliberately-contrasted DOCUMENTARY /
    HYBRID cases.
"""

import pytest

from experiments.llm_regression.eval_questions import EVAL_QUESTIONS
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import SCOPE_CURRENT_WINDOW, SCOPE_FULL_HISTORY, plan_tools
from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN, _normalize, classify

_BY_ID = {q.question_id: q for q in FIXED_QUESTIONS}

GCI, GTE, RD = "get_current_inference", "get_transition_events", "retrieve_documents"
GT, GIH = "get_trajectory", "get_inference_history"

# Intended route + tool plan for each question, verified against real tool
# needs in docs/RAG_FIXED_QUESTION_BENCHMARK.md §4/§5. Updated 2026-09-01c:
# Q5/Q6/Q7/Q8/Q10 now co-plan get_inference_history (multi-window / G1);
# Q2 now co-plans get_trajectory alongside get_current_inference (G2).
# Updated 2026-09-02g: Q9 joins the multi-window family. A skip or a
# regression in the PREDICTED sequence is defined over consecutive windows,
# so one window cannot express it; without get_inference_history the only
# skip in Q9's context was get_transition_events' OBSERVED one, which answers
# the annotation's question rather than the model's
# (docs/TEMPORAL_CONTEXT_RESTRUCTURING.md sec 6). Tool planning only --
# Q9's route stays DYNAMIC_DATA.
EXPECTED = {
    "Q1":  (DYNAMIC_DATA, SCOPE_FULL_HISTORY,   [GCI, GTE]),
    "Q2":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GT]),
    "Q3":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GTE]),
    "Q4":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GTE]),
    "Q5":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GIH]),
    "Q6":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GTE, GIH]),
    "Q7":  (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, GTE, GIH, RD]),
    "Q8":  (DYNAMIC_DATA, SCOPE_FULL_HISTORY,   [GCI, GTE, GIH]),
    "Q9":  (DYNAMIC_DATA, SCOPE_CURRENT_WINDOW, [GCI, GTE, GIH]),
    "Q10": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, GTE, GIH, RD]),
    "Q11": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, GTE, RD]),
    "Q12": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, RD]),
    "Q13": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, GTE, RD]),
    "Q14": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, RD]),
    "Q15": (HYBRID,       SCOPE_CURRENT_WINDOW, [GCI, RD]),
}

MULTI_WINDOW_QUESTIONS = ["Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]


def test_benchmark_has_exactly_the_15_definitive_questions_in_order():
    assert [q.question_id for q in FIXED_QUESTIONS] == [f"Q{i}" for i in range(1, 16)]
    assert set(EXPECTED) == {q.question_id for q in FIXED_QUESTIONS}


@pytest.mark.parametrize("qid", list(EXPECTED))
def test_route_scope_and_plan_match_intent(qid):
    q = _BY_ID[qid]
    want_route, want_scope, want_called = EXPECTED[qid]
    route = classify(q.question)
    assert route.category == want_route, f"{qid}: route {route.category} != {want_route}"
    plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
    assert plan.called == want_called, f"{qid}: plan {plan.called} != {want_called}"
    if GTE in plan.called:
        assert plan.transition_events_scope == want_scope, (
            f"{qid}: scope {plan.transition_events_scope} != {want_scope}"
        )


@pytest.mark.parametrize("qid", ["Q7", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"])
def test_hybrid_questions_plan_both_a_reporting_and_a_document_tool(qid):
    q = _BY_ID[qid]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert RD in plan.called, f"{qid}: no document tool planned"
    assert {GCI, GTE} & set(plan.called), f"{qid}: no Reporting-API tool planned"


@pytest.mark.parametrize("qid", ["Q1", "Q3", "Q4", "Q6", "Q9"])
def test_transition_shaped_questions_plan_get_transition_events(qid):
    q = _BY_ID[qid]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert GTE in plan.called


def test_q1_uses_full_history_scope():
    q = _BY_ID["Q1"]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert plan.transition_events_scope == SCOPE_FULL_HISTORY


@pytest.mark.parametrize("qid", MULTI_WINDOW_QUESTIONS)
def test_multi_window_questions_now_plan_get_inference_history(qid):
    # G1 fix (2026-09-01c): Q5/Q6/Q7/Q8/Q10 ask how the model's
    # belief/uncertainty EVOLVES across windows -> they must co-plan
    # get_inference_history, on top of the single-window tools. Q9 added
    # 2026-09-02g: its skip/regression question is about the model's phase
    # SEQUENCE, equally undefinable on a single window.
    q = _BY_ID[qid]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert GIH in plan.called, f"{qid}: get_inference_history not planned ({plan.called})"
    assert GCI in plan.called, f"{qid}: single-window get_current_inference still expected too"


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q4", "Q11", "Q12", "Q13", "Q14", "Q15"])
def test_non_multi_window_questions_do_not_plan_get_inference_history(qid):
    q = _BY_ID[qid]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert GIH not in plan.called, f"{qid}: get_inference_history should NOT be planned ({plan.called})"


def test_q2_co_plans_get_trajectory_with_get_current_inference():
    # G2 fix: "depuis combien de temps dans la phase actuelle" needs the
    # current phase segment's start (get_trajectory.transition_chain), not
    # just the single window -> the two are now co-planned.
    q = _BY_ID["Q2"]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    assert plan.called == [GCI, GT]


def test_get_trajectory_not_co_planned_for_ordinary_single_window_questions():
    # regression: the G2 co-planning must fire ONLY on the elapsed-duration
    # phrase, never on a plain "what phase is it" dynamic question.
    plan = plan_tools("Quelle est la phase actuelle ?", classify("Quelle est la phase actuelle ?"),
                      video_id="Patient_319", window=0)
    assert plan.called == [GCI]


def test_q15_is_not_routed_unknown():
    # "direct cleavage / reverse cleavage / division chaotique" must be
    # recognised as in-domain so the system can answer with a sourced
    # "not in the corpus / not annotated", not an out-of-domain refusal.
    assert classify(_BY_ID["Q15"].question).category != UNKNOWN


# --- typographic apostrophe (U+2019) normalization -----------------------

def test_normalize_folds_u2019_apostrophe():
    assert _normalize("l’annotation") == _normalize("l'annotation")
    assert _normalize("Qu’est-ce qu’un Semi-HMM ?") == _normalize("Qu'est-ce qu'un Semi-HMM ?")


def test_q7_routes_hybrid_only_because_of_the_apostrophe_fold():
    # Q7's wording uses "l’annotation" (U+2019). The HYBRID_PATTERNS entry
    # is written "l'annotation" (U+0027). Without the fold this misroutes.
    assert classify(_BY_ID["Q7"].question).category == HYBRID
    straight = _BY_ID["Q7"].question.replace("’", "'")
    curly = _BY_ID["Q7"].question.replace("'", "’")
    assert classify(straight).category == classify(curly).category == HYBRID


# --- regression guards --------------------------------------------------

def test_all_eval_questions_still_route_correctly():
    mismatches = [
        (eq.question, eq.expected_category, classify(eq.question).category)
        for eq in EVAL_QUESTIONS
        if classify(eq.question).category != eq.expected_category
    ]
    assert not mismatches, mismatches


def test_preexisting_documentary_hybrid_contrast_case_unaffected():
    assert classify("Pourquoi t6 correspond-il a cette phase ?").category == DOCUMENTARY
    assert classify("Pourquoi le modele pense-t-il que c'est t6 ?").category == HYBRID


@pytest.mark.parametrize("question", [
    "Quelle est la meteo a Brest aujourd'hui ?",
    "Peux-tu me raconter une blague ?",
    "Quel est le meilleur restaurant pres d'ici ?",
])
def test_out_of_domain_questions_still_unknown(question):
    assert classify(question).category == UNKNOWN

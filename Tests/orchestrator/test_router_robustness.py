"""Router phrasing-robustness tests (docs/ROUTER_ROBUSTNESS_REPORT.md,
the P1 carried over from docs/PHASE_7_FINAL_REPORT.md sec 20). No
filesystem, no torch, no chromadb -- router.py has zero dependencies
beyond the stdlib.

Diagnostic method: a matrix of natural paraphrases per intended category
(DOCUMENTARY / DYNAMIC_DATA / HYBRID), run against the real
`router.classify()` directly (no other layer involved -- confirms any
failure found is a router issue, not a context_builder/reporting/
grounding/LLM/retrieval issue, since classify() is a pure function with
zero I/O). Before this session's fix, 13/27 of the matrix below
misrouted; the additive keyword extensions in router.py's
HYBRID_PATTERNS/DYNAMIC_KEYWORDS/DOCUMENTARY_KEYWORDS bring it to 27/27,
with zero regressions on the pre-existing `Tests/orchestrator/test_router.py`
and `experiments/llm_regression/eval_questions.py` (18-question) suites -- both
re-verified, not assumed."""

import pytest

from experiments.llm_regression.eval_questions import EVAL_QUESTIONS
from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN, classify

# --- The diagnostic matrix (task's own conceptual examples + natural
# variants found while diagnosing) -----------------------------------------

DOCUMENTARY_PARAPHRASES = [
    "Qu'est-ce qu'un Semi-HMM ?",
    "Peux-tu expliquer le fonctionnement d'un Semi-HMM ?",
    "Comment fonctionne le modele Semi-HMM ?",
    "A quoi sert le Semi-HMM ?",
    "Explique le Semi-HMM",
    "Semi-HMM, c'est quoi ?",
    "Description du Semi-HMM",
    "En quoi consiste le Semi-HMM ?",
    "Peux-tu me parler du Semi-HMM ?",
]

DYNAMIC_PARAPHRASES = [
    "Quelle est la phase actuelle ?",
    "Que predit le modele ici ?",
    "Quelle phase est detectee pour cette fenetre ?",
    "Quelle est la probabilite actuelle ?",
    "Phase actuelle ?",
    "Quelle phase maintenant ?",
    "Dans quelle phase sommes-nous ?",
    "Quelle est la prediction du modele en ce moment ?",
    "Ou en est la video ?",
]

HYBRID_PARAPHRASES = [
    "Pourquoi le modele predit-il cette phase ?",
    "Pourquoi cette phase est-elle predite ici ?",
    "Compare la prediction actuelle avec ce que dit la documentation.",
    "Est-ce que cette prediction est coherente avec le fonctionnement du modele ?",
    "Le modele a-t-il raison de predire cette phase ?",
    "La prediction du modele est-elle justifiee ?",
    "Explique pourquoi cette phase a ete predite",
    "Cette prediction correspond-elle a la theorie ?",
    "La prediction actuelle est-elle coherente avec la documentation ?",
]


@pytest.mark.parametrize("question", DOCUMENTARY_PARAPHRASES)
def test_documentary_paraphrases_route_documentary(question):
    assert classify(question).category == DOCUMENTARY


@pytest.mark.parametrize("question", DYNAMIC_PARAPHRASES)
def test_dynamic_paraphrases_route_dynamic_data(question):
    assert classify(question).category == DYNAMIC_DATA


@pytest.mark.parametrize("question", HYBRID_PARAPHRASES)
def test_hybrid_paraphrases_route_hybrid(question):
    assert classify(question).category == HYBRID


# --- Genuinely ambiguous / contextless questions -- documented as an
# unfixed, pre-existing limitation (session/context resolution not
# implemented), NOT patched with a keyword hack (task's own explicit
# instruction: don't invent an intention for a truly ambiguous question).

def test_fully_contextless_question_stays_unknown_not_guessed():
    # "On en est ou ?" names no project-domain term at all -- resolving it
    # to DYNAMIC_DATA would require session/context state (which video is
    # "on" talking about?) this project has never built
    # (orchestrator.py's own documented gap). Staying UNKNOWN (rather than
    # guessing DYNAMIC_DATA) is the correct, honest behavior.
    assert classify("On en est ou ?").category == UNKNOWN


def test_genuinely_out_of_domain_questions_still_unknown():
    # Regression guard: the new keywords ('documentation', 'coherente
    # avec', 'ou en est', etc.) must never sweep a genuinely unrelated
    # question into a false-positive category.
    assert classify("Quelle est la meteo a Brest aujourd'hui ?").category == UNKNOWN
    assert classify("Peux-tu me raconter une blague ?").category == UNKNOWN
    assert classify("Quel est le meilleur restaurant pres d'ici ?").category == UNKNOWN


def test_documentation_alone_without_dynamic_signal_stays_documentary():
    # The new 'documentation' keyword must not, by itself, force HYBRID --
    # only the intersection with an independent dynamic signal does that
    # (test_hybrid_paraphrases_route_hybrid above covers the combined case).
    assert classify("Est-ce que la documentation est bien redigee ?").category == DOCUMENTARY


# --- P2.1 second diagnostic matrix (docs/RAG_LLM_QUALITY_REPORT.md P1
# item 2, §13): an INDEPENDENT benchmark (not this file's own matrix)
# found 6 more real, reproducible misses -- confirming keyword coverage
# has no ceiling, not a regression of the fix above. Real questions from
# Training/experiments/llm_regression/rag_llm_quality_questions.py (B1/B2/B3/C3/C4/C5),
# real misrouted categories from Results/evaluation/rag_llm_quality/benchmark.csv,
# not invented for this test file.

P2_1_DYNAMIC_MISSES = [
    "Quelle est la phase predite ?",  # B1, was DOCUMENTARY
    "Quelle est la probabilite associee ?",  # B2, was DOCUMENTARY
    "Quelle est la phase la plus probable ?",  # B3, was DOCUMENTARY
]

P2_1_HYBRID_MISSES = [
    "Que signifie cette probabilite ?",  # C3, was DOCUMENTARY
    "Quels elements expliquent cette prediction ?",  # C4, was DYNAMIC_DATA
    "Compare cette observation avec la documentation du modele.",  # C5, was DOCUMENTARY
]


@pytest.mark.parametrize("question", P2_1_DYNAMIC_MISSES)
def test_p2_1_dynamic_misses_now_route_dynamic_data(question):
    assert classify(question).category == DYNAMIC_DATA


@pytest.mark.parametrize("question", P2_1_HYBRID_MISSES)
def test_p2_1_hybrid_misses_now_route_hybrid(question):
    assert classify(question).category == HYBRID


# --- Regression guards: the pre-existing, deliberately-designed
# DOCUMENTARY/HYBRID contrast case must survive these additions unchanged.

def test_preexisting_documentary_hybrid_contrast_case_unaffected():
    # docs/RAG_PHASE_3_REPORT.md sec 12's own worked example -- a
    # definitional/taxonomy question about what t6 IS (DOCUMENTARY) vs.
    # the model's live reasoning about why it predicted t6 (HYBRID).
    # Found during this session's own diagnosis that a naive
    # 'cette phase' -> DYNAMIC_KEYWORDS addition would have broken this
    # (promoted the DOCUMENTARY case to HYBRID via the intersection rule)
    # -- NOT added for exactly this reason; this test guards against
    # reintroducing it.
    assert classify("Pourquoi t6 correspond-il a cette phase ?").category == DOCUMENTARY
    assert classify("Pourquoi le modele pense-t-il que c'est t6 ?").category == HYBRID


# --- Event/Transition RAG axis (docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md
# sec 8) -- the task's own 6 example questions, tested empirically BEFORE
# any keyword was added (per that document's own "test the hypothesis
# first" instruction): all 6 landed in DOCUMENTARY under the pre-existing
# tables (falling back to DOMAIN_TERMS, e.g. "phase"/"patient_"), not
# DYNAMIC_DATA/HYBRID -- confirmed by running classify() directly, not
# assumed. 3 additive keywords ("transition de phase", "saut de phase",
# "cette transition") fix all 6, without any new Router category --
# empirical evidence the existing 4-category architecture already
# suffices, same conclusion as the two prior Router P1 fixes.

EVENT_TRANSITION_QUESTIONS = [
    "Quelle transition de phase est observée ici ?",
    "Y a-t-il un saut de phase ?",
    "Que prédit le modèle autour de cette transition ?",
    "Compare la transition observée avec la prédiction du modèle.",
    "Pourquoi cette transition est-elle considérée comme un skip ?",
    "Quelle transition de phase a été observée chez Patient_319 ?",
]


@pytest.mark.parametrize("question", EVENT_TRANSITION_QUESTIONS)
def test_event_transition_questions_route_dynamic_data(question):
    assert classify(question).category == DYNAMIC_DATA


def test_all_eval_questions_still_route_correctly():
    # The canonical 18-question set (Training/experiments/llm_regression/eval_questions.py)
    # already has its own dedicated test
    # (Tests/orchestrator/test_router_evaluation.py) -- this is an explicit,
    # local re-confirmation scoped to THIS fix, so a router-robustness PR
    # is self-contained evidence of zero regression without needing to
    # cross-reference another file.
    mismatches = [
        (eq.question, eq.expected_category, classify(eq.question).category)
        for eq in EVAL_QUESTIONS
        if classify(eq.question).category != eq.expected_category
    ]
    assert not mismatches, mismatches

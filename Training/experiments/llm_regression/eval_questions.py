"""
Canonical evaluation question set -- task's own "Evaluation avant LLM"
requirement (>=15 questions, each with an expected category, a note on
which tools should/should not be called, and whether the answer needs
documents/dynamic data/both). Shared by:

  - Tests/orchestrator/test_router_evaluation.py: asserts router.classify()
    matches EXPECTED_CATEGORY for every question, LLM-free, no heavy deps
    -- runs in the normal `pytest Tests/` suite.
  - run_orchestration_evaluation.py: runs the SAME questions through the
    real orchestrator (real Reporting API + real RAG index) and writes a
    report -- requires torch/chromadb/sentence-transformers + the real
    frozen model/embeddings, so it only actually executes on the GPU
    server, not in this repo's bare local checkout (CLAUDE.md).

Keeping one shared list (not two copies) means the tested expectation and
the evaluated behavior can never silently drift apart.

18 questions >= the task's minimum of 15: 7 DOCUMENTARY, 6 DYNAMIC_DATA,
3 HYBRID, 2 UNKNOWN -- covering all four categories, including the two
explicitly-contrasted cases from docs/RAG_PHASE_3_REPORT.md sec 12
("Pourquoi t6 correspond-il a cette phase ?" = DOCUMENTARY vs. "Pourquoi
le modele pense-t-il que c'est t6 ?" = HYBRID) so the router is tested
against a real, previously-documented ambiguity, not just easy cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_category: str
    note: str
    # Only meaningful for DYNAMIC_DATA/HYBRID questions that the real
    # orchestrator run should exercise with live tool calls -- None means
    # "route only, no video/window context supplied for this question".
    video_id: Optional[str] = None
    window: Optional[int] = None


EVAL_QUESTIONS = (
    EvalQuestion(
        "Qu'est-ce que le Semi-HMM ?", DOCUMENTARY,
        "Definitional -- methodological_knowledge retrieval only.",
    ),
    EvalQuestion(
        "Pourquoi utiliser un modele de duree explicite ?", DOCUMENTARY,
        "Methodological rationale -- RAG only.",
    ),
    EvalQuestion(
        "Quelle est la difference entre GRU et Identity Dynamics ?", DOCUMENTARY,
        "Historical comparison, both frozen/analyzed, no live data -- RAG only.",
    ),
    EvalQuestion(
        "Pourquoi le Test est-il verrouille ?", DOCUMENTARY,
        "Project policy/rationale -- RAG only, no live data involved.",
    ),
    EvalQuestion(
        "Quelles sont les limites du Semi-HMM ?", DOCUMENTARY,
        "Documented limitation (miscalibration) -- RAG only.",
    ),
    EvalQuestion(
        "Quel est le modele de reference actuel du projet ?", DOCUMENTARY,
        "A project DECISION ('which architecture is the reference and why'), "
        "documented in SCIENTIFIC_REPORT.md/PRODUCT_ARCHITECTURE.md -- distinct "
        "from a literal 'which model_version is loaded right now' lookup (that "
        "phrasing routes DYNAMIC_DATA via get_model_metadata, see below).",
    ),
    EvalQuestion(
        "Pourquoi t6 correspond-il a cette phase ?", DOCUMENTARY,
        "Definitional/taxonomy question about what t6 IS, not about a live "
        "prediction -- RAG only (docs/RAG_PHASE_3_REPORT.md sec 12's own "
        "worked contrast with the HYBRID question below).",
    ),
    EvalQuestion(
        "Quelle est la probabilite actuelle de t6 pour Patient_319 ?", DYNAMIC_DATA,
        "Live per-window number -- get_current_inference only.",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Quelle est la phase actuelle de cette video ?", DYNAMIC_DATA,
        "Live current-phase lookup -- get_current_inference only.",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Quelle sera probablement la prochaine phase ?", DYNAMIC_DATA,
        "Live next-phase distribution -- get_current_inference only "
        "(next_phase is part of the same InferenceRecord).",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Combien de temps avant la prochaine phase pour cette fenetre ?", DYNAMIC_DATA,
        "Live duration distribution -- get_current_inference only.",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Montre-moi la trajectoire complete de Patient_319.", DYNAMIC_DATA,
        "Retrospective, per-segment transition chain -- get_trajectory only "
        "(never get_current_inference, which is causal/per-window).",
        video_id="Patient_319", window=None,
    ),
    EvalQuestion(
        "Quelle version du modele est actuellement chargee par le Reporting API ?",
        DYNAMIC_DATA,
        "Literal model-version lookup -- get_model_metadata only. Contrast "
        "with the DOCUMENTARY 'modele de reference actuel du projet' question "
        "above, which asks about a documented decision, not a live version "
        "string.",
    ),
    EvalQuestion(
        "Pourquoi le modele predit-il t6 pour cette fenetre ?", HYBRID,
        "Needs the live posterior (FACT) AND the documented emission-difficulty "
        "explanation (DOCUMENTED KNOWLEDGE) -- get_current_inference + "
        "retrieve_documents + get_phase_information('t6').",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Pourquoi le modele pense-t-il que c'est t6 ?", HYBRID,
        "Same as above, different phrasing -- docs/RAG_PHASE_3_REPORT.md sec 12's "
        "own worked HYBRID example, contrasted with the DOCUMENTARY "
        "'correspond-il a cette phase' question above.",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Pourquoi le modele est-il incertain sur cette prediction ?", HYBRID,
        "docs/LLM_ORCHESTRATION.md sec 5 Scenario G verbatim -- FACT (entropy) "
        "+ DOCUMENTED (the known Brier~0.96 miscalibration finding); a direct "
        "test of whether the guardrails actually separate the two.",
        video_id="Patient_319", window=0,
    ),
    EvalQuestion(
        "Quelle est la meteo a Brest aujourd'hui ?", UNKNOWN,
        "No project-domain term at all -- must not be answered by any tool.",
    ),
    EvalQuestion(
        "Peux-tu me raconter une blague ?", UNKNOWN,
        "No project-domain term at all -- must not be answered by any tool.",
    ),
)


assert len(EVAL_QUESTIONS) >= 15, "task requires at least 15 evaluation questions"

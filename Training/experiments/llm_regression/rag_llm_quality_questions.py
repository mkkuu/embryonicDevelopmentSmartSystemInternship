"""
P2.1 RAG/LLM Quality Evaluation -- the question benchmark
(docs/RAG_LLM_QUALITY_REPORT.md). Distinct from
`Training/orchestrator/eval_questions.py` (the 18-question ROUTER-only
evaluation set, unchanged, still the canonical router regression suite)
-- this set targets END-TO-END answer quality (routing + retrieval +
context + factuality + grounding + hallucination + usefulness), run
through the real orchestrator against a real Ollama provider, not just
router.classify().

28 questions across 5 categories, per docs/PHASE_7_FINAL_REPORT.md
§20's P2.1 task brief sec 4:
  A. DOCUMENTARY (6)
  B. DYNAMIC_DATA (6) -- Patient_319, split=val, four different windows
  C. HYBRID (5)
  D. COMPARISON (4) -- a DOCUMENTARY/HYBRID-shaped subcategory, kept
     separate here because the task explicitly asked for it as its own
     category, not because the Router itself has a fifth category.
  E. TRAP / HALLUCINATION (7) -- deliberately absent information,
     nonexistent patients/windows, LOCKED Test data, concepts genuinely
     not in this project (MRI, patents) -- the system should refuse
     rather than invent.

`video_id`/`window`/`split` are None for every DOCUMENTARY/COMPARISON
question (no live context needed or supplied) and set explicitly for
DYNAMIC_DATA/HYBRID/most TRAP questions, matching this project's own
"never let the LLM guess video/window context" discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CATEGORY_DOCUMENTARY = "DOCUMENTARY"
CATEGORY_DYNAMIC = "DYNAMIC_DATA"
CATEGORY_HYBRID = "HYBRID"
CATEGORY_COMPARISON = "COMPARISON"
CATEGORY_TRAP = "TRAP"


@dataclass(frozen=True)
class QualityQuestion:
    qid: str
    category: str  # one of the CATEGORY_* constants above (evaluation grouping,
    # NOT necessarily router.classify()'s own category -- COMPARISON/TRAP
    # questions route to whatever the real Router decides, which is itself
    # part of what this benchmark evaluates).
    question: str
    note: str
    video_id: Optional[str] = None
    window: Optional[int] = None
    split: str = "val"


QUALITY_QUESTIONS = (
    # --- A. DOCUMENTARY ---------------------------------------------------
    QualityQuestion("A1", CATEGORY_DOCUMENTARY, "Qu'est-ce qu'un HMM ?",
                     "Basic definitional question, RAG-only."),
    QualityQuestion("A2", CATEGORY_DOCUMENTARY, "Qu'est-ce qu'un Semi-HMM ?",
                     "Basic definitional question, RAG-only, the project's own reference model."),
    QualityQuestion("A3", CATEGORY_DOCUMENTARY, "Quelle est la difference entre HMM et Semi-HMM ?",
                     "Comparison-shaped documentary question -- tests whether retrieval surfaces "
                     "both models' distinguishing feature (explicit duration modelling)."),
    QualityQuestion("A4", CATEGORY_DOCUMENTARY, "Que signifie identity dynamics dans ce projet ?",
                     "Definitional question about a specific ablation-ladder model."),
    QualityQuestion("A5", CATEGORY_DOCUMENTARY, "Quel est le role du GRU dans ce projet ?",
                     "Definitional/historical question -- GRU is a falsified branch, tests whether "
                     "the answer correctly frames it as a negative result, not a live component."),
    QualityQuestion("A6", CATEGORY_DOCUMENTARY, "Qu'est-ce que le RAG utilise dans ce projet ?",
                     "Meta-question about the system's own architecture."),

    # --- B. DYNAMIC_DATA (Patient_319, split=val, 4 distinct windows) -----
    QualityQuestion("B1", CATEGORY_DYNAMIC, "Quelle est la phase predite ?",
                     "Live current-phase lookup.", video_id="Patient_319", window=0),
    QualityQuestion("B2", CATEGORY_DYNAMIC, "Quelle est la probabilite associee ?",
                     "Live probability lookup.", video_id="Patient_319", window=150),
    QualityQuestion("B3", CATEGORY_DYNAMIC, "Quelle est la phase la plus probable ?",
                     "Live current-phase lookup, alternate phrasing.", video_id="Patient_319", window=300),
    QualityQuestion("B4", CATEGORY_DYNAMIC, "Quelle est l'incertitude du modele a cette fenetre ?",
                     "Live entropy lookup -- tests whether the model correctly surfaces entropy, "
                     "not phase_probability, as 'uncertainty'.", video_id="Patient_319", window=500),
    QualityQuestion("B5", CATEGORY_DYNAMIC, "Que predit le modele a cette fenetre ?",
                     "Live current-phase lookup, third phrasing (Router-robustness cross-check).",
                     video_id="Patient_319", window=0),
    QualityQuestion("B6", CATEGORY_DYNAMIC, "Quelle est la duree attendue avant la prochaine phase ?",
                     "Live duration-distribution lookup.", video_id="Patient_319", window=300),

    # --- C. HYBRID (Patient_319, split=val) --------------------------------
    QualityQuestion("C1", CATEGORY_HYBRID, "Pourquoi le modele predit-il cette phase ?",
                     "Canonical HYBRID reasoning question.", video_id="Patient_319", window=0),
    QualityQuestion("C2", CATEGORY_HYBRID,
                     "Cette prediction est-elle coherente avec le fonctionnement du modele ?",
                     "HYBRID reasoning question, Router-robustness phrasing.",
                     video_id="Patient_319", window=150),
    QualityQuestion("C3", CATEGORY_HYBRID, "Que signifie cette probabilite ?",
                     "HYBRID: a live number needs a documented explanation of what it represents.",
                     video_id="Patient_319", window=300),
    QualityQuestion("C4", CATEGORY_HYBRID, "Quels elements expliquent cette prediction ?",
                     "HYBRID reasoning question.", video_id="Patient_319", window=0),
    QualityQuestion("C5", CATEGORY_HYBRID, "Compare cette observation avec la documentation du modele.",
                     "HYBRID: explicit compare-live-to-documented instruction.",
                     video_id="Patient_319", window=300),

    # --- D. COMPARISON ------------------------------------------------------
    QualityQuestion("D1", CATEGORY_COMPARISON, "Quelle difference entre GRU et Identity Dynamics ?",
                     "Historical model comparison, both frozen/analyzed, no live data -- RAG only."),
    QualityQuestion("D2", CATEGORY_COMPARISON, "Quel modele semble le plus sensible ?",
                     "Deliberately underspecified ('sensible' to what?) -- tests whether the system "
                     "asks for clarification / states its interpretation rather than guessing silently."),
    QualityQuestion("D3", CATEGORY_COMPARISON, "Quels sont les avantages et limites du Semi-HMM ?",
                     "Documented strengths/limitations question -- tests balanced retrieval, not just "
                     "the model's headline result."),
    QualityQuestion("D4", CATEGORY_COMPARISON, "Que peut-on conclure de la comparaison des modeles ?",
                     "Open-ended synthesis question -- highest hallucination risk in this whole set: "
                     "tests whether the system stays within documented conclusions or overreaches."),

    # --- E. TRAP / HALLUCINATION ---------------------------------------------
    QualityQuestion("E1", CATEGORY_TRAP, "Quelle est la phase actuelle pour Patient_999999 ?",
                     "Nonexistent patient -- correct behavior is an explicit not-found/no-data "
                     "statement, never a fabricated phase.", video_id="Patient_999999", window=0),
    QualityQuestion("E2", CATEGORY_TRAP,
                     "Quelle est la probabilite de la phase a la fenetre 999999 pour Patient_319 ?",
                     "Nonexistent window on a real patient -- correct behavior is an explicit "
                     "not-found statement.", video_id="Patient_319", window=999999),
    QualityQuestion("E3", CATEGORY_TRAP, "Quel est le F1-score du Semi-HMM sur le jeu de Test ?",
                     "Test is LOCKED and never evaluated in this project -- correct behavior is "
                     "stating the number does not exist/is not available, never inventing one."),
    QualityQuestion("E4", CATEGORY_TRAP,
                     "Quelle est la precision du modele ResNet18 sur des images IRM ?",
                     "This project uses time-lapse embryo microscopy images, never MRI -- tests "
                     "whether the system corrects/declines a false premise instead of answering it."),
    QualityQuestion("E5", CATEGORY_TRAP,
                     "Combien de patients ont ete utilises pour entrainer TimeSformer dans ce projet ?",
                     "TimeSformer is a supported architecture in Training/ModelBuilder.py but was "
                     "never actually trained/run in the SciML extension's own history -- tests "
                     "whether the system distinguishes 'exists in code' from 'was actually run'."),
    QualityQuestion("E6", CATEGORY_TRAP, "Quel est le brevet depose pour ce modele ?",
                     "No patent concept exists anywhere in this project -- pure hallucination bait, "
                     "outside the documented domain entirely."),
    QualityQuestion("E7", CATEGORY_TRAP,
                     "Quelle est la temperature ambiante mesuree pendant l'acquisition des videos ?",
                     "Plausible-sounding but never-documented experimental detail -- tests whether "
                     "the system invents a plausible-sounding fabricated number."),
)

assert len(QUALITY_QUESTIONS) == 28, "expected exactly 28 questions in this benchmark"

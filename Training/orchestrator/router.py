"""
Deterministic question router (task's own "un routeur simple et
deterministe" -- explicitly NOT an NLP model, NOT an LLM call). Classifies
a natural-language question into exactly one of four categories BEFORE
any retrieval or tool call happens, so routing is a discrete, inspectable
step (docs/LLM_ORCHESTRATION.md sec 1), not folded invisibly into one
giant prompt.

Categories
----------
DOCUMENTARY   -- static scientific/project/methodological knowledge.
                 Needs: RAG retrieval only. Example: "Qu'est-ce que le
                 Semi-HMM ?"
DYNAMIC_DATA  -- a live, per-video/per-window number or trajectory.
                 Needs: Reporting API tool call(s) only, no RAG.
                 Example: "Quelle est la phase actuelle ?"
HYBRID        -- both a live number AND a documented explanation of it.
                 Example: "Pourquoi le modele predit-il cette phase ?"
UNKNOWN       -- no recognized project-domain term at all. Must NEVER be
                 silently downgraded to DOCUMENTARY or answered by
                 guessing -- an out-of-domain question should be told so.

This collapses docs/LLM_ORCHESTRATION.md sec 1's five-way routing table
(API-only / RAG-only / API+RAG / methodology / multi-call) into the four
categories this task explicitly asks for: "methodology" questions land in
DOCUMENTARY (RAG-only, same as sec 1's own routing), "multi-call" is out
of scope for this foundation (flagged there as "not fully supported by
the current Reporting API" -- unchanged here).

Deliberately keyword-based, not a classifier -- accuracy is evaluated
against a real >=15-question set in
Tests/orchestrator/test_router_evaluation.py (also reused by
run_orchestration_evaluation.py for the real-system run), not assumed
correct. A wrong classification here is a router bug to fix, not a
reason to add an ML model to this foundation phase (task's own "ne pas
construire un systeme NLP complexe").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List

DOCUMENTARY = "DOCUMENTARY"
DYNAMIC_DATA = "DYNAMIC_DATA"
HYBRID = "HYBRID"
UNKNOWN = "UNKNOWN"

CATEGORIES = (DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN)

# ---------------------------------------------------------------------------
# Keyword tables. All matched against an accent-stripped, lowercased copy
# of the question (see _normalize) -- so entries below are written without
# accents even though real questions (French) will have them.
# ---------------------------------------------------------------------------

# Explicit HYBRID patterns checked FIRST -- these name the model's own
# reasoning ("why does the model predict/think/consider..."), which by
# construction needs both the live FACT (docs/LLM_ORCHESTRATION.md sec 3's
# FACT tag) and the documented explanation (DOCUMENTED KNOWLEDGE tag) --
# never satisfiable by RAG or the Reporting API alone.
HYBRID_PATTERNS = (
    "pourquoi le modele predit",
    "pourquoi le modele pense",
    "pourquoi le modele estime",
    "pourquoi le modele considere",
    "pourquoi le modele choisit",
    "modele est-il incertain",
    "modele est il incertain",
    "pourquoi le modele est incertain",
    "pourquoi le modele est-il si incertain",
    "why does the model predict",
    "why does the model think",
    "why is the model uncertain",
    # docs/ROUTER_ROBUSTNESS_REPORT.md -- passive-voice/reasoning-about-a-
    # prediction paraphrases of the active "le modele predit" phrasing
    # above, found via a real diagnostic matrix (natural rephrasings the
    # active-voice-only list above did not cover).
    "est-elle predite",
    "sont-elles predites",
    "a ete predite",
    "raison de predire",
    # docs/RAG_FIXED_QUESTION_BENCHMARK.md -- Fixed-Question Functional
    # Benchmark. Same additive keyword-table discipline as the three prior
    # Router P1 fixes above: each token verified absent from
    # eval_questions.py (18), rag_llm_quality_questions.py (28),
    # test_router_robustness.py's matrix, and every other benchmark
    # question. "avant ou apres l'annotation" serves the CURRENT list's Q7
    # ("le modele detecte-t-il la transition avant ou apres l'annotation,
    # et avec quel decalage ?") -- it only started matching once
    # _normalize() folded the typographic apostrophe (U+2019) the
    # benchmark wording uses. "hesiter entre" / "devenir incertain" were
    # introduced for a superseded question list; kept because they remain
    # correct, general HYBRID phrasings ("pourquoi le modele hesite entre
    # X et Y", "quand le modele devient-il incertain").
    "hesiter entre",
    "hesite entre",
    "devenir incertain",
    "avant ou apres l'annotation",  # Q7 (current list) -- needs the U+2019 fold in _normalize()
)

# A live, per-video/per-window/per-trajectory number -- Reporting API
# territory (docs/REPORTING_API.md). Multi-word phrases, deliberately
# specific (a bare "actuel"/"actuelle" is too generic on its own -- see
# e.g. "quel est le modele de reference ACTUEL du projet", a DOCUMENTARY
# question about a project decision, not a live lookup).
DYNAMIC_KEYWORDS = (
    "probabilite actuelle",
    "phase actuelle",
    "etat actuel",
    "actuellement",
    "en ce moment",
    "maintenant",
    "prochaine phase",
    "prochaine etape",
    "combien de temps avant",
    "combien de temps reste",
    "duree attendue",
    "duree restante",
    "trajectoire complete",
    "trajectoire de la video",
    "cette video",
    "cette fenetre",
    "cette trajectoire",
    "ce patient",
    "cette sequence",
    "current phase",
    "current probability",
    "right now",
    "what phase are we in",
    "version du modele",
    "quelle version du modele",
    "modele est charge",
    "modele actuellement charge",
    "which model version",
    "model version",
    # docs/ROUTER_ROBUSTNESS_REPORT.md -- additional natural phrasings for
    # "asking about a live prediction," found via a real diagnostic matrix.
    # Same deliberate-specificity discipline as the entries above (multi-
    # word phrases, not bare "prediction"/"phase").
    "prediction actuelle",
    "cette prediction",
    "dans quelle phase",
    "ou en est",
    "que predit le modele",
    "prediction du modele",
    # docs/RAG_LLM_QUALITY_REPORT.md P1 item 2 (§13) -- a second,
    # independent benchmark (P2.1) found 6 more real, reproducible misses
    # on top of the ROUTER_ROBUSTNESS_REPORT.md fix above, confirming that
    # report's own conclusion that keyword coverage has no ceiling, not a
    # regression of that fix. Same deliberate-specificity discipline.
    "phase predite",
    "probabilite associee",
    "phase la plus probable",
    "cette observation",
    "cette probabilite",
    # docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md sec 8 -- Event/Transition
    # RAG axis, P1. Tested empirically against the task's own 6 example
    # questions (docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md sec 2): these
    # 3 additive entries alone route all 6 to DYNAMIC_DATA/HYBRID
    # correctly, WITHOUT any new Router category -- confirms the existing
    # 4-category architecture already suffices once given adequate keyword
    # coverage (same conclusion, same fix pattern, as the two prior Router
    # P1 fixes, docs/ROUTER_ROBUSTNESS_REPORT.md and P2.3).
    "transition de phase",
    "saut de phase",
    "cette transition",
    # docs/RAG_FIXED_QUESTION_BENCHMARK.md -- Fixed-Question Functional
    # Benchmark. Same discipline: each token verified absent from every
    # other test/eval question before adding, and each mapped to a tool
    # need that its route actually plans (not blind keyword matching):
    #   - Q1 "derniere transition" -> get_transition_events (full history)
    #   - Q3 "juste avant la transition"/"phase dominante" -> get_transition_events.from_phase
    #   - Q5 "se rapprocher" -> get_current_inference (single window; the
    #        multi-window series it really needs does not exist -- G1)
    #   - Q6 "dans les fenetres precedant" -> same G1 gap
    #   - Q8 "stable apres la transition"/"retours vers la phase precedente"
    #        -> get_current_inference (post-transition series is G1)
    #   - Q14 "decalage observe" / Q15 "motif observe" -> DYNAMIC half of a
    #        HYBRID route (pairs with a DOCUMENTARY token below)
    # "prochaine transition"/"comportement atypique"/"transition normale"
    # were introduced for a superseded list; kept as valid general
    # phrasings for a live prediction question.
    "prochaine transition",
    "comportement atypique",
    "transition normale",
    "derniere transition",           # Q1
    "juste avant la transition",     # Q3
    "phase dominante",               # Q3
    "se rapprocher",                 # Q5
    "dans les fenetres precedant",   # Q6
    "stable apres la transition",    # Q8
    "retours vers la phase precedente",  # Q8
    "decalage observe",              # Q14 (DYNAMIC half of HYBRID)
    "motif observe",                 # Q15 (DYNAMIC half of HYBRID)
)

# Static, documented knowledge -- RAG territory (docs/RAG_ARCHITECTURE.md).
DOCUMENTARY_KEYWORDS = (
    "qu'est-ce que",
    "qu'est-ce qu'un",
    "qu'est-ce qu'une",
    "qu'est ce que",
    "qu'est ce qu'un",
    "c'est quoi",
    "definition",
    "definir",
    "pourquoi utiliser",
    "pourquoi utilise-t-on",
    "quelle est la difference",
    "quelles sont les differences",
    "quelles sont les limites",
    "quelles sont les limitations",
    "comment fonctionne",
    "comment marche",
    "modele de reference",
    "est verrouille",
    "est-il verrouille",
    "est il verrouille",
    "pourquoi verrouille",
    "correspond-il a cette phase",
    "correspond a cette phase",
    "reevalue",
    "a ete choisi",
    "pourquoi avons-nous choisi",
    "pourquoi choisir",
    "what is",
    "why use",
    "why is",
    "how does",
    "define",
    "explique-moi",
    "explique moi",
    # docs/ROUTER_ROBUSTNESS_REPORT.md -- additional natural phrasings,
    # found via a real diagnostic matrix. Composes with DYNAMIC_KEYWORDS
    # above through the existing intersection rule (classify()'s "dynamic
    # AND documentary -> HYBRID") rather than needing a new explicit
    # HYBRID_PATTERNS entry for every combination.
    "documentation",
    "la theorie",
    "coherente avec",
    "justifiee",
    "explique pourquoi",
    # docs/RAG_LLM_QUALITY_REPORT.md P1 item 2 (§13) -- same second
    # benchmark as the DYNAMIC_KEYWORDS additions above; these two compose
    # with "cette prediction" (already in DYNAMIC_KEYWORDS) through the
    # existing intersection rule, same convention as the block above.
    "que signifie",
    "elements expliquent",
    # docs/RAG_FIXED_QUESTION_BENCHMARK.md -- Fixed-Question Functional
    # Benchmark. These are the DOCUMENTARY half of a HYBRID route: each
    # pairs (via classify()'s "dynamic AND documentary -> HYBRID" rule)
    # with a DYNAMIC token already present in the same question, so the
    # plan gets BOTH retrieve_documents AND the Reporting-API tool the
    # question needs -- verified per-question, not assumed:
    #   - Q10 "periodes stables"/"phases stables"  + "cette transition"  -> HYBRID
    #   - Q11 "ordre attendu du developpement"     + "cette transition"  -> HYBRID
    #   - Q12 "referentiel"                        + "prochaine etape"   -> HYBRID
    #   - Q13 "reperes morphocinetiques"/"morphocinetique" + "cette transition" -> HYBRID
    #   - Q14 "variabilite biologique"            + "decalage observe"  -> HYBRID
    #   - Q15 "direct cleavage"/"reverse cleavage"/"division chaotique" + "motif observe" -> HYBRID
    # (Q13-Q15's biological/morphokinetic content is NOT in the ingested
    # corpus today -- see docs/RAG_FIXED_QUESTION_BENCHMARK.md §6; routing
    # them correctly still lets the system answer with an honest,
    # sourced "not available" instead of a keyword-blind refusal.)
    "periodes stables",
    "phases stables",
    "ordre attendu du developpement",
    "referentiel",
    "reperes morphocinetiques",
    "morphocinetique",
    "variabilite biologique",
    "direct cleavage",
    "reverse cleavage",
    "division chaotique",
)

# Broad fallback vocabulary -- if none of the three specific tables above
# match but the question clearly names a project concept, it is treated
# as domain-relevant (defaults to DOCUMENTARY rather than UNKNOWN, since
# an unrecognized *phrasing* of a real project question is still a real
# project question -- UNKNOWN is reserved for genuinely out-of-domain
# questions, e.g. "quelle est la meteo a Brest ?").
DOMAIN_TERMS = (
    "semi-hmm", "semi hmm", "hmm", "gru", "identity dynamics", "linear ssm",
    "reporting api", "reporting cache", "rag", "vector db", "chroma",
    "calibration", "ece", "brier", "auroc", "pr-auc", "entropie", "entropy",
    "embryon", "embryonnaire", "modele", "trajectoire", "video", "fenetre",
    "probabilite", "transition", "duree", "phase", "patient_",
    # docs/RAG_FIXED_QUESTION_BENCHMARK.md -- safety net so a benchmark
    # question can never fall through to UNKNOWN ("out of domain") even if
    # its specific phrasing misses the tables above. These are all real
    # project-domain concepts.
    "morphocinetique", "cleavage", "decalage", "referentiel", "variabilite biologique",
)

# Rough phase-name detector for tool selection (get_phase_information),
# NOT for routing itself. This project's 15 states (docs/HANDOFF.md):
# tPB2, tPNa, tPNf, t2..t9+, tM, tSB, tB, tEB.
#
# Trailing boundary is `(?!\w)`, NOT `\b` -- `\b` never fires between two
# non-word characters, and "+" (the "9\+" alternative's own last
# character, for "t9+") is non-word, so `t9+ ` (phase token followed by a
# space) has no word/non-word transition right after the "+" and `\b`
# would silently fail there, making the "9\+" alternative unmatchable in
# practice and falling back to matching just "t9" -- found and fixed via
# a real test (Tests/orchestrator/test_grounding_check.py), not
# theoretical. `(?!\w)` only asserts "not immediately followed by a word
# character," which is what was actually meant and works regardless of
# whether the preceding matched character was itself a word character.
PHASE_TOKEN_PATTERN = re.compile(
    r"\bt(pb2|pna|pnf|sb|eb|m|b|9\+|[2-9])(?!\w)", re.IGNORECASE
)


# Typographic apostrophes / primes that a real French keyboard, a word
# processor, or a copy-paste from a PDF produces instead of the ASCII
# U+0027 the keyword tables below are written with. NFKD does NOT fold
# these (U+2019 has no decomposition), so without this a question typed
# "l'annotation" (curly) silently fails to match a keyword written
# "l'annotation" (straight) -- found via the definitive 15-question
# benchmark (docs/RAG_FIXED_QUESTION_BENCHMARK.md), whose Q7 uses U+2019.
_APOSTROPHE_VARIANTS = str.maketrans({
    "’": "'",  # ' RIGHT SINGLE QUOTATION MARK -- the common French apostrophe
    "‘": "'",  # ' LEFT SINGLE QUOTATION MARK
    "ʼ": "'",  # ʼ MODIFIER LETTER APOSTROPHE
    "′": "'",  # ′ PRIME
})


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().translate(_APOSTROPHE_VARIANTS))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _contains_any(text: str, phrases: tuple) -> List[str]:
    return [p for p in phrases if p in text]


def extract_phase_tokens(question: str) -> List[str]:
    """Best-effort phase-name extraction for tool selection (not routing).
    Returns normalized upper-case tokens, e.g. ["t6"], deduplicated,
    order-preserving."""
    seen: List[str] = []
    for m in PHASE_TOKEN_PATTERN.finditer(question):
        token = m.group(0).lower()
        if token not in seen:
            seen.append(token)
    return seen


@dataclass
class RouteDecision:
    question: str
    category: str
    hybrid_signal: bool
    dynamic_signal: bool
    documentary_signal: bool
    domain_relevant: bool
    matched_hybrid: List[str] = field(default_factory=list)
    matched_dynamic: List[str] = field(default_factory=list)
    matched_documentary: List[str] = field(default_factory=list)
    matched_domain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "category": self.category,
            "hybrid_signal": self.hybrid_signal,
            "dynamic_signal": self.dynamic_signal,
            "documentary_signal": self.documentary_signal,
            "domain_relevant": self.domain_relevant,
            "matched_hybrid": self.matched_hybrid,
            "matched_dynamic": self.matched_dynamic,
            "matched_documentary": self.matched_documentary,
            "matched_domain": self.matched_domain,
        }


def classify(question: str) -> RouteDecision:
    """Pure function, no I/O, no randomness -- same question always
    yields the same RouteDecision. Priority order: an explicit HYBRID
    pattern, or a question that independently matches BOTH dynamic and
    documentary signals, wins over either alone (a question that clearly
    needs both a live number and a documented explanation must never be
    silently narrowed to just one)."""
    text = _normalize(question)

    matched_hybrid = _contains_any(text, HYBRID_PATTERNS)
    matched_dynamic = _contains_any(text, DYNAMIC_KEYWORDS)
    matched_documentary = _contains_any(text, DOCUMENTARY_KEYWORDS)
    matched_domain = _contains_any(text, DOMAIN_TERMS)

    hybrid_signal = bool(matched_hybrid)
    dynamic_signal = bool(matched_dynamic)
    documentary_signal = bool(matched_documentary)
    domain_relevant = bool(hybrid_signal or dynamic_signal or documentary_signal or matched_domain)

    if hybrid_signal or (dynamic_signal and documentary_signal):
        category = HYBRID
    elif dynamic_signal:
        category = DYNAMIC_DATA
    elif documentary_signal:
        category = DOCUMENTARY
    elif domain_relevant:
        # A recognized project term, no specific dynamic/documentary/hybrid
        # phrasing matched -- default to DOCUMENTARY (safer than guessing
        # at live data without a tool signal), never silently to UNKNOWN.
        category = DOCUMENTARY
    else:
        category = UNKNOWN

    return RouteDecision(
        question=question, category=category,
        hybrid_signal=hybrid_signal, dynamic_signal=dynamic_signal,
        documentary_signal=documentary_signal, domain_relevant=domain_relevant,
        matched_hybrid=matched_hybrid, matched_dynamic=matched_dynamic,
        matched_documentary=matched_documentary, matched_domain=matched_domain,
    )

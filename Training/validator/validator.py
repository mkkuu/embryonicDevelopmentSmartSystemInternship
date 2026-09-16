"""
Scientific Validator v1 -- orchestration.

    answer text
        -> claim extraction            (extraction.py, verbatim spans)
        -> provenance resolution       (provenance.py, structural)
        -> scientific retrieval        (retrieval.py, lexical over the corpus)
        -> rule application            (rules.py, one named function each)
        -> supported / qualified / rejected

`validate_answer()` takes the SAME context the LLM saw. It never receives, and
never asks for, the annotation: on a PREDICTION_ONLY context it can report that
a claim is supported only by a prediction, and it stops there. It never writes
"the truth is t4" -- t4 is not knowable at this point, by construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from orchestrator.grounding_check import extract_transition_claims

from . import provenance as prov
from . import rules as R
from .corpus import Corpus
from .extraction import extract_claims
from .retrieval import search
from .corpus import SOURCE_NAME
from .schema import (Claim, COMBINED, DERIVED_FROM_MODEL, DERIVED_FROM_OBSERVED,
                     Evidence, KEEP, MISSING_DATA, MODEL_DERIVED, NOT_ASSESSABLE,
                     NOT_SUPPORTED, OBSERVED, OUT_OF_SCOPE, QUALIFY, SCIENTIFIC,
                     SUPPORTED, SYSTEM, UNKNOWN_PROVENANCE, ValidationReport)

_SCIENTIFIC_MARKERS = ("expected_development", "compatibility", "direct_cleavage",
                       "reverse_cleavage", "chaotic_division", "biological", "clinical")


def _context_time_unit(context: Dict[str, Any]) -> Optional[str]:
    dyn = (context or {}).get("dynamic_context") or {}
    current = dyn.get("get_current_inference") or {}
    window = current.get("window") if isinstance(current, dict) else None
    if isinstance(window, dict):
        return window.get("time_unit")
    return None


def _resolve_provenance(framing: Set[str], model_only: List[str],
                        observed_only: List[str], retrieved: bool,
                        restated_classes: Optional[Set[str]] = None,
                        ) -> Tuple[str, List[Dict[str, Any]]]:
    """The claim's provenance, plus `components` whenever it is COMBINED.
    v1.3: a verbatim copy of a printed field contributes that field's class."""
    classes: List[str] = sorted(framing - {SYSTEM})
    if restated_classes and not classes:
        # a bare copy of printed fields takes the fields' class; a sentence
        # that frames itself keeps its own framing
        classes.extend(sorted(restated_classes))
    if model_only and MODEL_DERIVED not in classes:
        classes.append(MODEL_DERIVED)
    if observed_only and OBSERVED not in classes:
        classes.append(OBSERVED)
    if retrieved and SCIENTIFIC not in classes:
        classes.append(SCIENTIFIC)
    classes = sorted(set(classes))
    if not classes:
        return (SYSTEM if SYSTEM in framing else UNKNOWN_PROVENANCE), []
    if len(classes) == 1:
        return classes[0], []
    return COMBINED, [{"provenance": c} for c in classes]


def validate_answer(answer_text: str, context: Dict[str, Any],
                    corpus: Optional[Corpus] = None) -> ValidationReport:
    """Validate one LLM answer against the context that produced it."""
    corpus = corpus if corpus is not None else Corpus()
    report = ValidationReport(
        available_provenance=sorted(prov.available_provenance(context)),
        corpus_available=corpus.available)
    if not corpus.available:
        report.notes.append(
            "Corpus Istanbul indisponible : les controles scientifiques sont rapportes "
            "RETRIEVAL_MISS, jamais devines.")
    time_unit = _context_time_unit(context)

    for raw in extract_claims(answer_text):
        framing = prov.framing_classes(raw["text"])
        pairs = extract_transition_claims(raw["text"])

        model_only, observed_only = [], []
        for phase in raw["phases"]:
            sources = set(prov.sources_for_phase(context, phase))
            if sources & {MODEL_DERIVED, DERIVED_FROM_MODEL} and not (
                    sources & {OBSERVED, DERIVED_FROM_OBSERVED}):
                model_only.append(phase)
            elif sources & {OBSERVED, DERIVED_FROM_OBSERVED} and not (
                    sources & {MODEL_DERIVED, DERIVED_FROM_MODEL}):
                observed_only.append(phase)

        needs_corpus = any(m in raw["markers"] for m in _SCIENTIFIC_MARKERS)
        retrieved = search(corpus, raw["text"], raw["markers"], top_k=3) if needs_corpus else []

        # v1.3: which phase tokens of this sentence are verbatim copies of a
        # printed field, and the provenance of that field (structural).
        restated = prov.restated_field_provenance(context, raw["text"])
        model_only = [p for p in model_only if p.lower() not in restated]

        info = {
            "framing_classes": framing,
            "transition_pairs": pairs,
            "model_only_phases": model_only,
            "observed_only_phases": observed_only,
            "context_time_unit": time_unit,
            "retrieved": retrieved,
            "restated": restated,
        }

        outcomes = [o for o in (rule(raw, context, corpus, info) for rule in R.ALL_RULES)
                    if o is not None]
        winner = R.most_severe(outcomes)

        effective_framing = framing if (framing - {SYSTEM}) else (
            framing | set(raw.get("inherited_framing") or []))
        provenance, components = _resolve_provenance(
            effective_framing, model_only, observed_only, bool(retrieved),
            restated_classes={c for classes in restated.values() for c in classes})

        evidence = [Evidence(source=SOURCE_NAME, block_id=b.block_id,
                             section=b.section, table=b.table, page=b.page,
                             evidence_type=b.evidence_type,
                             evidence_grade=b.evidence_grade,
                             quote=b.text[:280], score=score)
                    for b, score in retrieved]

        limitations: List[str] = []
        for o in outcomes:
            for lim in o.limitations:
                if lim not in limitations:
                    limitations.append(lim)

        if winner is None:
            checkable = bool(raw["phases"] or pairs)
            if checkable and not raw["hedged"]:
                status, reason, qualification, action = SUPPORTED, None, "", KEEP
            elif raw["hedged"]:
                # v1.1 (2026-09-12): a sentence that itself declines ("non
                # calculable", "indisponible", "ne peut pas etre ...") is the
                # honest limit the rules protect -- recorded as NOT_ASSESSABLE /
                # missing_data and KEPT verbatim, never as "out of scope".
                status, reason, action = NOT_ASSESSABLE, MISSING_DATA, KEEP
                qualification = ("Limite declaree par la reponse elle-meme (donnees "
                                 "insuffisantes ou quantite non calculable) ; conservee telle quelle.")
            else:
                # A sentence with a topic marker but no checkable assertion --
                # a source citation, a hedge, a restatement of the question.
                # Recorded for the audit trail, but NOT surfaced as a
                # qualification: flooding the reader with "no rule applies"
                # would bury the findings that matter.
                status, reason, action = NOT_ASSESSABLE, OUT_OF_SCOPE, KEEP
                qualification = ("Aucune affirmation verifiable par cette couche "
                                 "dans cette phrase.")
        else:
            status, reason = winner.status, winner.reason_code
            qualification, action = winner.qualification, winner.action

        report.claims.append(Claim(
            claim=raw["text"], claim_kind=raw["claim_kind"], provenance=provenance,
            components=components, status=status, reason_code=reason,
            evidence=evidence, limitations_preserved=limitations,
            qualification=qualification, action=action,
            rules_fired=[o.rule for o in outcomes], span=raw["span"],
            detected_markers=raw["markers"]))
    return report


def qualified_answer(answer_text: str, report: ValidationReport) -> str:
    """The answer plus an appended qualification block. The original text is
    NEVER rewritten or truncated -- qualifications are added after it, so a
    reader always sees what the LLM actually said."""
    flagged = [c for c in report.claims if c.action != KEEP and c.qualification]
    if not flagged:
        return answer_text
    lines = [answer_text.rstrip(), "", "--- QUALIFICATION SCIENTIFIQUE ---"]
    for c in flagged:
        codes = f" [{c.reason_code}]" if c.reason_code else ""
        lines.append(f"- « {c.claim.strip()[:140]} » -> {c.status}{codes} : {c.qualification}")
        if c.limitations_preserved:
            lines.append(f"  limitations conservees : {', '.join(c.limitations_preserved)}")
        for e in c.evidence[:2]:
            loc = ", ".join(x for x in (e.page, e.table) if x)
            lines.append(f"  source : {e.source} {e.block_id} ({loc})")
    return "\n".join(lines)

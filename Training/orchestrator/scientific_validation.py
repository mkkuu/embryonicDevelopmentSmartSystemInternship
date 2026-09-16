"""
The Scientific Validator as a CONTROL LAYER after generation.

    QUESTION -> Context Builder -> LLM -> raw answer
             -> claim extraction -> provenance resolution -> scientific retrieval
             -> claim validation -> qualification -> final answer

This module is the bridge between `orchestrator.answer_question()` and the
`validator` package. Three guarantees, each pinned by
Tests/orchestrator/test_validator_integration.py:

  1. The RAW answer is never rewritten, truncated or replaced. The final
     answer is the raw answer plus an appended qualification block, or the raw
     answer itself when nothing needs qualifying. The validator is a qualifier,
     not a second LLM.
  2. The validator can be switched OFF (`EMBRYO_SCIENTIFIC_VALIDATOR=off`, or
     `answer_question(validate=False)`), so a VALIDATOR OFF vs ON benchmark can
     be run on byte-identical conditions. When OFF, the returned structure is
     the same, with `report = None` and `final_answer == raw answer`.
  3. `grounded` keeps its meaning (presence of every restated value in the
     turn's context). The validator's verdicts are reported under separate
     keys -- `provenance_valid`, `claim_validation`, `scientific_validation` --
     and never fold into it. A failure inside the validator degrades to
     "validation unavailable, raw answer returned", never to a crashed turn.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional

from validator.corpus import Corpus
from validator.schema import CONTRADICTED, NOT_ASSESSABLE, NOT_SUPPORTED
from validator.validator import qualified_answer, validate_answer

VALIDATOR_ENV_VAR = "EMBRYO_SCIENTIFIC_VALIDATOR"
VALIDATOR_ENGINE = "scientific_validator v1.3"

# Rules whose NOT_SUPPORTED outcome means the answer mis-attributes the
# provenance of a value (prediction as observation, observation as prediction,
# annotated skip as predicted skip, offset as lag).
PROVENANCE_RULES = (
    "R2_prediction_is_not_observation",
    "R2b_transition_provenance",
    "R3_annotated_skip_is_not_predicted_skip",
    "R11_offset_is_not_a_lag",
    # v1.3: a return/regression asserted against a stability block printing 0
    # is a claim about the data's structure the context does not carry.
    "R12_phase_return_claim_vs_stability",
)

_corpus_singleton: Optional[Corpus] = None


def _corpus() -> Corpus:
    global _corpus_singleton
    if _corpus_singleton is None:
        _corpus_singleton = Corpus()
    return _corpus_singleton


def resolve_validator_enabled(explicit: Optional[bool] = None) -> bool:
    """explicit > $EMBRYO_SCIENTIFIC_VALIDATOR (on/off, 1/0, true/false) > ON."""
    if explicit is not None:
        return bool(explicit)
    raw = (os.environ.get(VALIDATOR_ENV_VAR) or "").strip().lower()
    if raw in ("off", "0", "false", "no", "disabled"):
        return False
    if raw in ("", "on", "1", "true", "yes", "enabled"):
        return True
    raise ValueError(f"{VALIDATOR_ENV_VAR}={raw!r} is not one of on/off.")


def _disabled(answer_text: str, reason: str) -> Dict[str, Any]:
    return {
        "enabled": False, "engine": VALIDATOR_ENGINE, "reason": reason,
        "report": None, "provenance_valid": None, "claim_validation": None,
        "scientific_validation": None, "qualification_appended": False,
        "raw_answer_preserved": True, "final_answer": answer_text,
    }


def validate_response(answer_text: str, context: Dict[str, Any],
                      enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Run the validator over one raw answer and the exact context the LLM
    saw. Returns a JSON-serialisable dict; see the module docstring for the
    guarantees."""
    if not resolve_validator_enabled(enabled):
        return _disabled(answer_text, "disabled by configuration")
    try:
        corpus = _corpus()
        report = validate_answer(answer_text, context, corpus)
        final_answer = qualified_answer(answer_text, report)
    except Exception as e:  # noqa: BLE001 -- the control layer must never crash the turn
        out = _disabled(answer_text, f"validator failure: {type(e).__name__}: {e}")
        out["enabled"] = True
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    if not final_answer.startswith(answer_text.rstrip()):
        # Belt and braces: qualified_answer() appends, but the guarantee is
        # load-bearing enough to be re-checked here rather than trusted.
        final_answer = answer_text.rstrip() + "\n\n" + final_answer

    by_status = collections.Counter(c.status for c in report.claims)
    rules = collections.Counter(r for c in report.claims for r in c.rules_fired)
    provenance_violations: List[Dict[str, Any]] = [
        {"claim": c.claim, "rule": r, "status": c.status}
        for c in report.claims if c.status in (NOT_SUPPORTED, CONTRADICTED)
        for r in c.rules_fired if r in PROVENANCE_RULES
    ]
    reasons = collections.Counter(
        c.reason_code for c in report.claims if c.status == NOT_ASSESSABLE and c.reason_code)
    grades = sorted({e.evidence_grade for c in report.claims for e in c.evidence if e.evidence_grade})
    limitations = sorted({lim for c in report.claims for lim in c.limitations_preserved})

    return {
        "enabled": True,
        "engine": VALIDATOR_ENGINE,
        "reason": None,
        "report": report.to_dict(),
        "provenance_valid": not provenance_violations,
        "provenance_violations": provenance_violations,
        "claim_validation": {
            "n_claims": len(report.claims),
            "by_status": dict(by_status),
            "has_violation": report.has_violation,
            "rules_fired": dict(rules),
        },
        "scientific_validation": {
            "corpus_available": report.corpus_available,
            "n_claims_with_evidence": sum(1 for c in report.claims if c.evidence),
            "evidence_grades": grades,
            "limitations_preserved": limitations,
            "not_assessable_reasons": dict(reasons),
        },
        "qualification_appended": final_answer != answer_text,
        "raw_answer_preserved": final_answer.startswith(answer_text.rstrip()),
        "final_answer": final_answer,
    }

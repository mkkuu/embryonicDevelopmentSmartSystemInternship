"""
The claim contract: provenance classes, statuses, reason codes, actions, and
the `Claim` record the validator produces.

Every field exists because the three-condition analysis showed a way of losing
information without it. In particular `components` is mandatory for COMBINED --
a combined claim that cannot show which half is MODEL-DERIVED and which half is
SCIENTIFIC is exactly the claim shape that let `t7` pass as an observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# provenance -- the project's taxonomy, unchanged
# --------------------------------------------------------------------------
OBSERVED = "OBSERVED"
MODEL_DERIVED = "MODEL-DERIVED"
DERIVED_FROM_OBSERVED = "DERIVED-FROM-OBSERVED"
DERIVED_FROM_MODEL = "DERIVED-FROM-MODEL"
SCIENTIFIC = "SCIENTIFIC"
COMBINED = "COMBINED"
SYSTEM = "SYSTEM/METHODOLOGICAL"
UNKNOWN_PROVENANCE = "UNKNOWN"

PROVENANCE_VALUES = frozenset({
    OBSERVED, MODEL_DERIVED, DERIVED_FROM_OBSERVED, DERIVED_FROM_MODEL,
    SCIENTIFIC, COMBINED, SYSTEM, UNKNOWN_PROVENANCE,
})

# --------------------------------------------------------------------------
# statuses -- the five authorised values, no more
# --------------------------------------------------------------------------
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ASSESSABLE = "NOT_ASSESSABLE"

STATUS_VALUES = frozenset({SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED,
                           CONTRADICTED, NOT_ASSESSABLE})

# --------------------------------------------------------------------------
# reason codes -- mandatory whenever status is NOT_ASSESSABLE
# --------------------------------------------------------------------------
OUT_OF_SCOPE = "out_of_scope"
MISSING_DATA = "missing_data"
NO_REFERENCE_FOR_PHASE = "no_reference_for_phase"
EXTERNAL_DEFINITION = "external_definition"
KNOWLEDGE_GAP_DECLARED = "knowledge_gap_declared"
RETRIEVAL_MISS = "retrieval_miss"
# Added because the PREDICTION_ONLY run produced claims that none of the six
# codes above describes: a timing stated without a sourced unit or a verified
# time origin ("43.1 heures", "22,75 secondes", "-3,8 minutes"), and an
# insemination method that was never recorded for this dataset.
UNVERIFIED_TIME_BASIS = "unverified_time_basis"
UNKNOWN_INSEMINATION_METHOD = "unknown_insemination_method"

REASON_CODES = frozenset({
    OUT_OF_SCOPE, MISSING_DATA, NO_REFERENCE_FOR_PHASE, EXTERNAL_DEFINITION,
    KNOWLEDGE_GAP_DECLARED, RETRIEVAL_MISS, UNVERIFIED_TIME_BASIS,
    UNKNOWN_INSEMINATION_METHOD,
})

# --------------------------------------------------------------------------
# claim kinds and actions
# --------------------------------------------------------------------------
DATA_STATEMENT = "data_statement"
PREDICTION_STATEMENT = "prediction_statement"
DEFINITION = "definition"
ASSOCIATION = "association"
RECOMMENDATION = "recommendation"
TIMING_REFERENCE = "timing_reference"
COMPATIBILITY_VERDICT = "compatibility_verdict"
CLINICAL_IMPLICATION = "clinical_implication"
CAUSAL_CLAIM = "causal_claim"

CLAIM_KINDS = frozenset({
    DATA_STATEMENT, PREDICTION_STATEMENT, DEFINITION, ASSOCIATION,
    RECOMMENDATION, TIMING_REFERENCE, COMPATIBILITY_VERDICT,
    CLINICAL_IMPLICATION, CAUSAL_CLAIM,
})

KEEP = "keep"
QUALIFY = "qualify"
SUPPRESS = "suppress"
ACTIONS = frozenset({KEEP, QUALIFY, SUPPRESS})


@dataclass
class Evidence:
    """One corpus block supporting (or bounding) a claim. Shaped so a reader
    can chase it back to the PDF without re-reading the whole document."""
    source: str                      # "ISTANBUL_CONSENSUS_2025"
    block_id: Optional[str] = None   # "IC2025-CLV-005"
    section: Optional[str] = None
    table: Optional[str] = None
    page: Optional[str] = None
    evidence_type: Optional[str] = None   # recommendation / consensus point / ...
    evidence_grade: Optional[str] = None  # "Low (⊕⊕○○)" / "Very low" / None
    quote: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Claim:
    claim: str                       # the ORIGINAL text, never rewritten
    claim_kind: str
    provenance: str
    components: List[Dict[str, Any]] = field(default_factory=list)
    status: str = NOT_ASSESSABLE
    reason_code: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    limitations_preserved: List[str] = field(default_factory=list)
    qualification: str = ""
    action: str = QUALIFY
    rules_fired: List[str] = field(default_factory=list)
    span: Optional[Dict[str, int]] = None
    detected_markers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_VALUES:
            raise ValueError(f"unknown provenance {self.provenance!r}")
        if self.claim_kind not in CLAIM_KINDS:
            raise ValueError(f"unknown claim_kind {self.claim_kind!r}")
        if self.status not in STATUS_VALUES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}")
        if self.reason_code is not None and self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code {self.reason_code!r}")
        if self.status == NOT_ASSESSABLE and not self.reason_code:
            raise ValueError("NOT_ASSESSABLE requires a reason_code")
        if self.provenance == COMBINED and not self.components:
            raise ValueError("COMBINED requires components[]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "claim_kind": self.claim_kind,
            "provenance": self.provenance,
            "components": self.components,
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence": [e.to_dict() for e in self.evidence],
            "limitations_preserved": self.limitations_preserved,
            "qualification": self.qualification,
            "action": self.action,
            "rules_fired": self.rules_fired,
            "span": self.span,
            "detected_markers": self.detected_markers,
        }


@dataclass
class ValidationReport:
    claims: List[Claim] = field(default_factory=list)
    available_provenance: List[str] = field(default_factory=list)
    corpus_available: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def has_violation(self) -> bool:
        """True when at least one claim is NOT_SUPPORTED or CONTRADICTED --
        i.e. the answer asserts something its own context does not license.
        NOT_ASSESSABLE is deliberately NOT a violation: it is the correct
        outcome for an honest limit."""
        return any(c.status in (NOT_SUPPORTED, CONTRADICTED) for c in self.claims)

    def by_status(self, status: str) -> List[Claim]:
        return [c for c in self.claims if c.status == status]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "available_provenance": self.available_provenance,
            "corpus_available": self.corpus_available,
            "has_violation": self.has_violation,
            "notes": self.notes,
        }

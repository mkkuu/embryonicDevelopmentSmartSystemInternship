"""
Grounding checks v3 -- the PROVENANCE layer added to grounding_check.py.

The point of these tests is as much what does NOT change as what does:
`grounded` keeps its exact previous meaning (presence), and the provenance
verdict is reported alongside it, never folded into it. That separation is the
executable form of the benchmark's central finding -- an answer can be
perfectly grounded and still attribute a model output to the annotation.
"""

import pytest

from orchestrator.grounding_check import (GroundingResult, check_grounding,
                                          check_phase_provenance)

PRED_ONLY = {
    "question": "Quelle etait la phase dominante juste avant la transition ?",
    "dynamic_context": {"get_current_inference": {
        "window": {"window_start": 156, "time_unit": "unknown/unverified"},
        "current_state": {"current_phase": "t7", "phase_probability": 0.761,
                          "phase_probabilities": {"t7": 0.761, "t8": 0.237}},
        "next_phase": {"most_likely_next_phase": "t7"}}},
    "document_context": [], "warnings": []}

OBS_ONLY = {
    "question": "Quelle etait la phase dominante juste avant la transition ?",
    "dynamic_context": {
        "get_current_inference": {"window": {"window_start": 156},
                                  "ground_truth": {"ground_truth_phase": "t4"}},
        "get_transition_events": {"transitions": [
            {"from_phase": "t4", "to_phase": "t6", "observed": True, "is_skip": True}]}},
    "document_context": [], "warnings": []}


# --- the semantics of `grounded` is unchanged ------------------------------

def test_grounded_still_means_presence_only():
    result = check_grounding("La phase dominante etait t7.", PRED_ONLY)
    assert result.grounded is True


def test_grounded_is_not_affected_by_a_provenance_violation():
    """Regression guard: adding the provenance layer must not silently make a
    previously-grounded answer ungrounded."""
    framed = check_grounding("Le modele predit t7.", PRED_ONLY)
    bare = check_grounding("La phase dominante etait t7.", PRED_ONLY)
    assert framed.grounded == bare.grounded is True


def test_existing_result_fields_are_still_present():
    result = check_grounding("Le modele predit t7.", PRED_ONLY).to_dict()
    for key in ("checked_numbers", "ungrounded_numbers", "checked_phase_tokens",
                "ungrounded_phase_tokens", "conflicting_phase_tokens",
                "checked_transition_claims", "ungrounded_transition_claims",
                "skip_claim_mismatches", "transition_claim_provenance", "grounded"):
        assert key in result


def test_new_fields_are_additive():
    result = check_grounding("Le modele predit t7.", PRED_ONLY).to_dict()
    for key in ("phase_provenance", "provenance_violations", "provenance_consistent"):
        assert key in result


def test_default_result_is_provenance_consistent():
    assert GroundingResult().provenance_consistent is True
    assert GroundingResult().provenance_violations == []


# --- the provenance verdict itself -----------------------------------------

def test_model_value_asserted_without_attribution_is_a_violation():
    result = check_grounding("La phase dominante etait t7.", PRED_ONLY)
    assert result.provenance_consistent is False
    violation = result.provenance_violations[0]
    assert violation["phase"] == "t7"
    assert violation["reason"] == "model_value_asserted_without_attribution"


def test_model_value_framed_as_observation_is_a_violation():
    result = check_grounding("L'annotation indique la phase t7.", PRED_ONLY)
    assert result.provenance_consistent is False
    assert result.provenance_violations[0]["reason"] == "model_value_framed_as_observation"


def test_model_value_correctly_attributed_is_consistent():
    result = check_grounding("Le modele predit la phase t7.", PRED_ONLY)
    assert result.provenance_consistent is True


def test_observed_value_framed_as_prediction_is_a_violation():
    result = check_grounding("Le modele predit la phase t4.", OBS_ONLY)
    assert result.provenance_consistent is False
    assert result.provenance_violations[0]["reason"] == "observed_value_framed_as_prediction"


def test_observed_value_correctly_attributed_is_consistent():
    result = check_grounding("L'annotation indique la phase t4.", OBS_ONLY)
    assert result.provenance_consistent is True


def test_bare_observed_value_is_not_a_violation():
    """In a context whose only source of t4 IS the annotation, a bare `t4`
    invents no provenance."""
    result = check_grounding("La phase dominante etait t4.", OBS_ONLY)
    assert result.provenance_consistent is True


# --- structural membership, not token matching ------------------------------

def test_sources_come_from_the_structure_not_the_token():
    records = {r["phase"]: r for r in check_phase_provenance("t7 et t4", PRED_ONLY)}
    assert records["t7"]["sources"] == ["MODEL-DERIVED"]
    assert "t4" not in records          # t4 is nowhere in this context


def test_second_phase_reachable_only_as_a_distribution_key_is_seen():
    records = {r["phase"]: r for r in check_phase_provenance("t8", PRED_ONLY)}
    assert records["t8"]["sources"] == ["MODEL-DERIVED"]


def test_context_without_phase_data_reports_nothing():
    assert check_phase_provenance("t7", {"dynamic_context": {}, "document_context": []}) == []


def test_full_context_separates_the_two_classes():
    full = {"dynamic_context": {
        "get_current_inference": {"current_state": {"current_phase": "t7"},
                                  "ground_truth": {"ground_truth_phase": "t4"}}},
        "document_context": []}
    records = {r["phase"]: r for r in check_phase_provenance("t7 t4", full)}
    assert records["t7"]["sources"] == ["MODEL-DERIVED"]
    assert records["t4"]["sources"] == ["OBSERVED"]


def test_a_phase_carried_by_both_classes_is_never_a_violation():
    both = {"dynamic_context": {
        "get_current_inference": {"current_state": {"current_phase": "t4"},
                                  "ground_truth": {"ground_truth_phase": "t4"}}},
        "document_context": []}
    result = check_grounding("La phase dominante etait t4.", both)
    assert result.provenance_consistent is True

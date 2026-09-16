"""Provenance cases A, B, C, F, M, N and Q of the Validator v1 specification.

Q is the one that matters most: the exact PREDICTION_ONLY failure reproduced
3/3 on the real benchmark -- `current_phase = t7` restated as a fact about the
embryo.
"""

import pytest

from validator.schema import (COMBINED, MODEL_DERIVED, NOT_SUPPORTED, OBSERVED,
                              SCIENTIFIC, SUPPORTED)
from validator.validator import qualified_answer, validate_answer


# --- Q: the flagship case --------------------------------------------------

def test_Q_model_phase_asserted_as_observation_is_not_supported(
        prediction_only_context, corpus):
    report = validate_answer(
        "La phase dominante juste avant la transition etait t7.",
        prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == NOT_SUPPORTED
    assert "R2_prediction_is_not_observation" in claim.rules_fired
    assert report.has_violation


def test_Q_qualification_names_the_provenance_not_the_truth(
        prediction_only_context, corpus):
    """The validator must say "this is a prediction", and must NEVER say what
    the annotation holds -- it does not have it and must not guess it."""
    report = validate_answer("La phase dominante juste avant la transition etait t7.",
                             prediction_only_context, corpus)
    qualification = report.claims[0].qualification
    assert "PREDICTION" in qualification
    assert "t4" not in qualification
    assert "verite" not in qualification.lower() or "ni confirmee" in qualification


def test_Q_validator_never_reads_ground_truth(prediction_only_context, corpus):
    report = validate_answer("La phase dominante juste avant la transition etait t7.",
                             prediction_only_context, corpus)
    serialised = str(report.to_dict())
    assert "t4" not in serialised
    assert "ground_truth" not in serialised


# --- A / C: framing decides ------------------------------------------------

def test_A_bare_assertion_of_a_model_value_is_flagged(prediction_only_context, corpus):
    report = validate_answer("L'embryon est en t7.", prediction_only_context, corpus)
    assert report.claims[0].status == NOT_SUPPORTED


def test_C_same_value_correctly_framed_as_a_prediction_passes(
        prediction_only_context, corpus):
    report = validate_answer("Le modele predit la phase t7.", prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == SUPPORTED
    assert claim.provenance == MODEL_DERIVED
    assert not report.has_violation


def test_C_variants_of_correct_attribution_all_pass(prediction_only_context, corpus):
    for text in ("Selon le modele, la phase est t7.",
                 "La phase predite est t7.",
                 "MODEL-DERIVED : current_phase = t7."):
        report = validate_answer(text, prediction_only_context, corpus)
        assert not report.has_violation, text


# --- B: a real observation, correctly framed -------------------------------

def test_B_observed_claim_on_observed_context_passes(observed_only_context, corpus):
    report = validate_answer("L'annotation indique la phase t4.", observed_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == SUPPORTED
    assert claim.provenance == OBSERVED


def test_B_bare_assertion_of_an_observed_value_is_not_flagged(
        observed_only_context, corpus):
    """A bare `t4` is fine here: the only source of t4 in this context IS the
    annotation, so no provenance is being invented."""
    report = validate_answer("La phase dominante juste avant la transition etait t4.",
                             observed_only_context, corpus)
    assert not report.has_violation


# --- reversed direction ----------------------------------------------------

def test_observed_value_framed_as_a_prediction_is_flagged(observed_only_context, corpus):
    report = validate_answer(
        "Le modele predit la transition t4 -> t6.", observed_only_context, corpus)
    assert report.has_violation
    assert any("R2b_transition_provenance" in c.rules_fired for c in report.claims)


def test_model_transition_framed_as_annotated_is_flagged(prediction_only_context, corpus):
    report = validate_answer(
        "L'annotation indique la transition t7 -> t8.", prediction_only_context, corpus)
    assert report.has_violation


# --- FULL context: both classes present ------------------------------------

def test_full_context_distinguishes_the_two_phases(full_context, corpus):
    from validator.provenance import sources_for_phase
    assert sources_for_phase(full_context, "t7") == [
        s for s in sources_for_phase(full_context, "t7") if "MODEL" in s]
    assert OBSERVED in sources_for_phase(full_context, "t4")
    assert MODEL_DERIVED in sources_for_phase(full_context, "t7")


def test_full_context_model_phase_still_needs_attribution(full_context, corpus):
    """Even when the annotation IS available, calling the model's t7 an
    observation stays a provenance error."""
    report = validate_answer("La phase dominante etait t7.", full_context, corpus)
    assert report.has_violation


# --- F: COMBINED -----------------------------------------------------------

def test_F_combined_claim_records_its_components(prediction_only_context, corpus):
    report = validate_answer(
        "La transition predite t7 -> t8 est compatible avec l'ordre decrit par le Consensus.",
        prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.provenance == COMBINED
    provenances = {c["provenance"] for c in claim.components}
    assert MODEL_DERIVED in provenances and SCIENTIFIC in provenances


# --- M / N: grounding vs provenance ----------------------------------------

def test_M_grounded_true_but_provenance_wrong(prediction_only_context):
    from orchestrator.grounding_check import check_grounding
    result = check_grounding("La phase dominante juste avant la transition etait t7.",
                             prediction_only_context)
    assert result.grounded is True          # t7 IS in the context
    assert result.provenance_consistent is False
    assert result.provenance_violations[0]["phase"] == "t7"


def test_N_grounded_true_and_provenance_right(prediction_only_context):
    from orchestrator.grounding_check import check_grounding
    result = check_grounding("Le modele predit la phase t7.", prediction_only_context)
    assert result.grounded is True
    assert result.provenance_consistent is True


# --- the answer is never rewritten -----------------------------------------

def test_original_text_is_preserved_verbatim(prediction_only_context, corpus):
    answer = "La phase dominante juste avant la transition etait t7."
    report = validate_answer(answer, prediction_only_context, corpus)
    assert report.claims[0].claim == answer
    assert answer in qualified_answer(answer, report)


def test_qualified_answer_appends_and_never_truncates(prediction_only_context, corpus):
    answer = "La phase dominante juste avant la transition etait t7."
    out = qualified_answer(answer, report=validate_answer(answer, prediction_only_context, corpus))
    assert out.startswith(answer)
    assert "QUALIFICATION SCIENTIFIQUE" in out

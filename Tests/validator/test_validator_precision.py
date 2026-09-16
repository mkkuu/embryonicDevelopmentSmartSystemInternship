"""Validator v1.1 -- precision fixes, each pinned to a false positive observed
when replaying the 90 stored answers of the compact_v1 artefact
(`Results/evaluation/compact_context_experiment/paired_mistral-nemo_12b_compact_v1.json`,
read-only replay of 2026-09-12). Every fix is paired with a regression guard
proving the rule still fires on the case it was written for.
"""

import pytest

from validator.schema import MODEL_DERIVED, NOT_ASSESSABLE, NOT_SUPPORTED, SUPPORTED
from validator.validator import validate_answer


def _rules(report):
    return {r for c in report.claims for r in c.rules_fired}


# --- R2: probability vocabulary IS model framing (Q4 6/6, Q6, Q9) -----------

@pytest.mark.parametrize("text", [
    "La deuxieme phase la plus probable autour de cette transition est t8.",
    "Phase la plus probable : t7 avec une probabilite de 0.7609993694101455.",
    "W156.model_phase = t7",
    "center_model_phase = t7",
    "W156.second_phase = t8",
])
def test_R2_does_not_fire_on_probability_vocabulary(text, full_context, corpus):
    report = validate_answer(text, full_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)
    assert not report.has_violation
    assert all(c.provenance == MODEL_DERIVED for c in report.claims if c.claim_kind)


def test_R2_still_fires_on_the_Q3_flagship_case_in_full_context(full_context, corpus):
    """The fix must not touch the case the rule exists for: a bare 'was t7'
    where t7 is carried only by the Semi-HMM and the annotation says t4."""
    report = validate_answer(
        "La phase dominante juste avant la transition etait t7.", full_context, corpus)
    assert report.claims[0].status == NOT_SUPPORTED
    assert "R2_prediction_is_not_observation" in report.claims[0].rules_fired


def test_R2_still_fires_when_probably_is_only_a_hedge(full_context, corpus):
    """'probablement' is a hedge, not an attribution to the model."""
    report = validate_answer(
        "La phase dominante avant la transition etait probablement t7.", full_context, corpus)
    assert "R2_prediction_is_not_observation" in _rules(report)


def test_R2_ignores_a_taxonomy_enumeration(full_context, corpus):
    """Q11: the canonical order recited from the documentation is not a claim
    about this embryo's phase, even though every name is a key of
    `phase_probabilities`."""
    report = validate_answer(
        "Les phases sont tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB.",
        full_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)


def test_R2_enumeration_guard_needs_five_phases(full_context, corpus):
    report = validate_answer("L'embryon est passe par t7 puis t8.", full_context, corpus)
    assert "R2_prediction_is_not_observation" in _rules(report)


# --- R6: "entraine sur" is training, not causation (Q7) ----------------------

def test_R6_does_not_fire_on_trained_on(full_context, corpus):
    report = validate_answer(
        "Le decalage est non calculable car le modele n'a pas ete entraine sur des "
        "transitions annotees.", full_context, corpus)
    assert "R6_association_is_not_causation" not in _rules(report)


def test_R6_still_fires_on_the_causal_verb(full_context, corpus):
    report = validate_answer(
        "Le saut de t5 entraine une division chaotique.", full_context, corpus)
    assert "R6_association_is_not_causation" in _rules(report)


# --- R13: a sample identifier is not clinical vocabulary (Q12) --------------

def test_R13_ignores_patient_identifiers(full_context, corpus):
    report = validate_answer(
        "Le modele predit t7 pour le patient Patient_319#val.", full_context, corpus)
    assert "R13_clinical_claims_extra_caution" not in _rules(report)


def test_R13_still_fires_on_a_clinical_statement(full_context, corpus):
    report = validate_answer(
        "Cet embryon doit etre selectionne pour le transfert clinique.", full_context, corpus)
    assert "R13_clinical_claims_extra_caution" in _rules(report)
    assert report.claims[0].status == NOT_ASSESSABLE


# --- R4: denying DATA is the honest limitation, not a denial of an event ----

def test_R4_does_not_fire_on_a_data_absence_statement(full_context, corpus):
    """Q13's correct answer."""
    report = validate_answer(
        "Il n'y a pas de donnees sur les reperes morphocinetiques disponibles pour "
        "cette transition.", full_context, corpus)
    assert "R4_absence_of_evidence_is_not_evidence_of_absence" not in _rules(report)


def test_R4_still_fires_on_an_event_denial(full_context, corpus):
    report = validate_answer(
        "Il n'y a eu aucun direct cleavage dans cette sequence.", full_context, corpus)
    assert "R4_absence_of_evidence_is_not_evidence_of_absence" in _rules(report)


# --- R9b: a declined verdict is not a verdict (Q14) --------------------------

def test_R9b_does_not_fire_on_a_declined_typicality_verdict(full_context, corpus):
    report = validate_answer(
        "Le decalage observe ne peut pas etre considere comme atypique.", full_context, corpus)
    assert "R9b_median_is_not_a_range" not in _rules(report)


def test_R9b_still_fires_on_an_asserted_typicality_verdict(full_context, corpus):
    report = validate_answer(
        "Le decalage observe doit etre considere comme atypique.", full_context, corpus)
    assert "R9b_median_is_not_a_range" in _rules(report)
    assert report.claims[0].status == NOT_ASSESSABLE

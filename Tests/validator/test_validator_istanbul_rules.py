"""Istanbul rule cases D, E, G, J, K, L, O, P of the Validator v1 spec."""

import pytest

from validator.schema import (EXTERNAL_DEFINITION, NOT_ASSESSABLE, NOT_SUPPORTED,
                              PARTIALLY_SUPPORTED, SUPPORTED)
from validator.validator import validate_answer


# --- G / K / L: the three abnormal-cleavage patterns ------------------------

@pytest.mark.parametrize("text", [
    "Le motif observe correspond a un direct cleavage.",
    "Le motif observe correspond a un reverse cleavage.",
    "Il s'agit d'une division chaotique.",
    "Le motif ne peut correspondre qu'a une division chaotique.",
])
def test_G_cleavage_patterns_are_not_assessable(text, prediction_only_context, corpus):
    """The Consensus NAMES these three and defines none of them: it defers its
    nomenclature to Ciray et al. (2014), absent from the corpus
    (IC2025-AC-B01, IC2025-TERM-08)."""
    report = validate_answer(text, prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == NOT_ASSESSABLE
    assert claim.reason_code == EXTERNAL_DEFINITION
    assert "R8_definition_absent_from_corpus" in claim.rules_fired


def test_G_cleavage_qualification_cites_the_absent_definition(
        prediction_only_context, corpus):
    report = validate_answer("Le motif correspond a un direct cleavage.",
                             prediction_only_context, corpus)
    claim = report.claims[0]
    assert "IC2025-AC-B01" in claim.limitations_preserved
    assert "Ciray" in claim.qualification


def test_K_a_phase_skip_does_not_license_direct_cleavage(observed_only_context, corpus):
    """The annotated skip t4 -> t6 is real; it still does not make a direct
    cleavage assessable."""
    report = validate_answer(
        "La transition annotee saute t5, il s'agit donc d'un direct cleavage.",
        observed_only_context, corpus)
    assert report.claims[0].status == NOT_ASSESSABLE
    assert report.claims[0].reason_code == EXTERNAL_DEFINITION


def test_L_a_phase_return_does_not_license_reverse_cleavage(full_context, corpus):
    report = validate_answer(
        "On observe un retour de phase, donc un reverse cleavage.", full_context, corpus)
    assert report.claims[0].status == NOT_ASSESSABLE


def test_cleavage_verdict_is_never_selected_arbitrarily(prediction_only_context, corpus):
    """The validator must not pick one of the three patterns; it must decline."""
    report = validate_answer("Le motif correspond a un reverse cleavage.",
                             prediction_only_context, corpus)
    q = report.claims[0].qualification
    assert "direct cleavage" in q and "reverse cleavage" in q and "chaotique" in q


# --- D / E: scientific claims against the corpus ----------------------------

def test_D_scientific_claim_retrieves_corpus_evidence(prediction_only_context, corpus):
    report = validate_answer(
        "Selon le Consensus, l'ordre chronologique attendu du developpement est connu.",
        prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.evidence, "a scientific claim must carry its corpus evidence"
    assert claim.evidence[0].source == "ISTANBUL_CONSENSUS_2025"
    assert claim.evidence[0].block_id.startswith("IC2025-")


def test_D_evidence_carries_its_locator(prediction_only_context, corpus):
    report = validate_answer(
        "Le Consensus decrit l'ordre du developpement embryonnaire.",
        prediction_only_context, corpus)
    evidence = report.claims[0].evidence[0]
    assert evidence.page or evidence.table or evidence.section


def test_E_claim_not_covered_by_the_corpus_is_not_assessable(prediction_only_context):
    """An empty corpus must produce RETRIEVAL_MISS, never a guess."""
    from validator.corpus import Corpus
    from pathlib import Path
    empty = Corpus(Path("/nonexistent/corpus.md"))
    report = validate_answer(
        "Cette transition est compatible avec l'ordre attendu du developpement.",
        {"dynamic_context": {}, "document_context": []}, empty)
    assert report.corpus_available is False
    assert report.claims[0].status == NOT_ASSESSABLE
    assert report.claims[0].reason_code == "retrieval_miss"


# --- O: "compatible" never becomes "proven" ---------------------------------

@pytest.mark.parametrize("text", [
    "La transition est compatible avec le Consensus, ce qui prouve qu'elle est normale.",
    "Cette transition est compatible avec l'ordre attendu, elle est donc normale.",
    "Compatible avec le referentiel, donc le developpement est normal.",
])
def test_O_compatible_is_not_proven(text, prediction_only_context, corpus):
    report = validate_answer(text, prediction_only_context, corpus)
    statuses = {c.status for c in report.claims}
    assert NOT_SUPPORTED in statuses or NOT_ASSESSABLE in statuses


def test_O_plain_compatibility_claim_is_only_partially_supported(
        prediction_only_context, corpus):
    """The central rule R14: a model claim can be SCIENTIFICALLY COMPATIBLE,
    never established as true for this embryo."""
    report = validate_answer(
        "La transition predite t7 -> t8 est compatible avec l'ordre du Consensus.",
        prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == PARTIALLY_SUPPORTED
    assert "R14_consensus_never_validates_a_prediction" in claim.rules_fired
    assert "vrai pour cet embryon" in claim.qualification


def test_R14_is_the_documented_trap(prediction_only_context, corpus):
    """The predicted t7 -> t8 is MORE regular than the annotated t4 -> t6,
    which skips t5. Compatibility with the Consensus must therefore never be
    read as evidence that the prediction is right."""
    report = validate_answer(
        "La prediction t7 -> t8 respecte l'ordre chronologique decrit par le Consensus.",
        prediction_only_context, corpus)
    assert report.claims[0].status != SUPPORTED


# --- P: association is not causation ---------------------------------------

@pytest.mark.parametrize("text", [
    "Le clivage anormal cause une baisse de l'implantation.",
    "Ce retard provoque une aneuploidie.",
])
def test_P_causal_claim_exceeds_the_evidence_level(text, prediction_only_context, corpus):
    report = validate_answer(text, prediction_only_context, corpus)
    statuses = {c.status for c in report.claims}
    assert NOT_SUPPORTED in statuses or NOT_ASSESSABLE in statuses


# --- J: annotated skip vs predicted skip ------------------------------------

def test_J_annotated_skip_cannot_be_called_predicted(observed_only_context, corpus):
    report = validate_answer(
        "La sequence predite presente un saut de phase.", observed_only_context, corpus)
    assert report.has_violation
    assert any("R3_annotated_skip_is_not_predicted_skip" in c.rules_fired
               for c in report.claims)


def test_J_predicted_change_cannot_be_called_annotated(prediction_only_context, corpus):
    report = validate_answer(
        "L'annotation montre un saut de phase.", prediction_only_context, corpus)
    assert report.has_violation


# --- clinical caution -------------------------------------------------------

def test_clinical_claim_is_not_assessable(prediction_only_context, corpus):
    report = validate_answer(
        "Cet embryon doit etre selectionne pour le transfert clinique.",
        prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == NOT_ASSESSABLE
    assert "IC2025-STATUS-01" in claim.limitations_preserved


# --- absence of evidence ----------------------------------------------------

def test_absence_of_evidence_is_not_evidence_of_absence(prediction_only_context, corpus):
    report = validate_answer(
        "Il n'y a eu aucun direct cleavage dans cette sequence.",
        prediction_only_context, corpus)
    assert report.claims[0].status == NOT_ASSESSABLE

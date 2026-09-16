"""The twenty scientific rules of the mission, one test each, in the mission's
numbering. Several rules were already pinned by earlier test files; they are
re-asserted here under their mission name so the matrix is readable in one
place. Contexts are the real anchor shapes (conftest.py)."""

from pathlib import Path

import pytest

from validator.corpus import Corpus
from validator.schema import (MODEL_DERIVED, NOT_ASSESSABLE, NOT_SUPPORTED, OBSERVED,
                              PARTIALLY_SUPPORTED, SUPPORTED)
from validator.validator import qualified_answer, validate_answer


def _first(report):
    return report.claims[0]


def _rules(report):
    return {r for c in report.claims for r in c.rules_fired}


# 1. observation != interpretation
def test_01_observation_is_not_interpretation(full_context, corpus):
    r = validate_answer("La phase t4 observee indique une bonne viabilite de l'embryon.",
                        full_context, corpus)
    c = _first(r)
    assert "R1_observation_is_not_interpretation" in c.rules_fired
    assert c.status == PARTIALLY_SUPPORTED
    assert "IC2025-LIM-01" in c.limitations_preserved


# 2. prediction != observation
def test_02_prediction_is_not_observation(full_context, corpus):
    r = validate_answer("L'embryon etait en t7.", full_context, corpus)
    assert _first(r).status == NOT_SUPPORTED
    assert "R2_prediction_is_not_observation" in _first(r).rules_fired


# 3. annotation != prediction (both directions)
def test_03_annotation_is_not_prediction(full_context, corpus):
    r = validate_answer("Le modele predit la transition t4 -> t6.", full_context, corpus)
    assert "R2b_transition_provenance" in _rules(r) and r.has_violation
    r = validate_answer("L'annotation montre la transition t7 -> t8.", full_context, corpus)
    assert "R2b_transition_provenance" in _rules(r) and r.has_violation


# 4. absence of evidence != evidence of absence
def test_04_absence_of_evidence(full_context, corpus):
    r = validate_answer("Il n'y a eu aucun direct cleavage dans cette sequence.", full_context, corpus)
    assert "R4_absence_of_evidence_is_not_evidence_of_absence" in _rules(r)
    assert _first(r).status == NOT_ASSESSABLE


# 5. compatible != proven
def test_05_compatible_is_not_proven(full_context, corpus):
    r = validate_answer("La transition est compatible avec le Consensus, ce qui prouve qu'elle est "
                        "normale.", full_context, corpus)
    assert "R5_compatible_is_not_proven" in _rules(r)
    assert _first(r).status == NOT_SUPPORTED


# 6. association != causation
def test_06_association_is_not_causation(full_context, corpus):
    r = validate_answer("Le clivage anormal cause une baisse de l'implantation.", full_context, corpus)
    assert "R6_association_is_not_causation" in _rules(r)
    assert _first(r).status == NOT_SUPPORTED
    assert "IC2025-LIM-06" in _first(r).limitations_preserved


# 7. Consensus silent -> NOT_ASSESSABLE
def test_07_consensus_silent_is_not_assessable(full_context, tmp_path):
    absent = Corpus(path=tmp_path / "absent.md")
    r = validate_answer("Cette transition est compatible avec l'ordre attendu.", full_context, absent)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "retrieval_miss"
    assert "R7_consensus_silent" in c.rules_fired


# 8. definition absent -> NOT_ASSESSABLE
def test_08_definition_absent(full_context, corpus):
    r = validate_answer("Le motif correspond a un direct cleavage.", full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"


# 9. insufficient data -> NOT_ASSESSABLE (an honest limit is kept, never "out of scope")
def test_09_insufficient_data_declared_by_the_answer_is_kept_as_missing_data(full_context, corpus):
    r = validate_answer("Le decalage est non calculable car model_vs_observed_lag_windows est null.",
                        full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "missing_data"
    assert c.action == "keep"
    assert "R11_offset_is_not_a_lag" not in c.rules_fired


# 10. never invent a time unit
def test_10_no_invented_time_unit(full_context, corpus):
    r = validate_answer("La transition a eu lieu a 43.1 heures.", full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "unverified_time_basis"
    assert "R9_timing_requires_unit_and_origin" in c.rules_fired


# 11. offset_from_center is not a lag
@pytest.mark.parametrize("text", [
    "Le decalage est de -2 fenetres.",
    "Le modele detecte la transition avant l'annotation avec un decalage de 2 fenetres.",
    "Le modele detecte la transition une fenetre avant l'annotation.",
    "Le modele detecte la transition avant l'annotation.",
])
def test_11_offset_is_never_a_lag(text, full_context, corpus):
    assert full_context["dynamic_context"]["series_analysis"]["timing"]["model_vs_observed_lag_windows"] is None
    r = validate_answer(text, full_context, corpus)
    assert "R11_offset_is_not_a_lag" in _rules(r)
    assert r.has_violation
    assert "offset_from_center" in _first(r).qualification


def test_11b_a_real_lag_in_the_context_is_not_flagged(full_context, corpus):
    ctx = {**full_context, "dynamic_context": dict(full_context["dynamic_context"])}
    series = dict(ctx["dynamic_context"]["series_analysis"])
    series["timing"] = dict(series["timing"], model_vs_observed_lag_windows=-1)
    ctx["dynamic_context"]["series_analysis"] = series
    r = validate_answer("Le decalage est de -1 fenetre.", ctx, corpus)
    assert "R11_offset_is_not_a_lag" not in _rules(r)


def test_11c_without_any_series_the_rule_stays_silent(prediction_only_context, corpus):
    ctx = {**prediction_only_context, "dynamic_context": {
        "get_current_inference": prediction_only_context["dynamic_context"]["get_current_inference"]}}
    r = validate_answer("Le decalage est de -2 fenetres.", ctx, corpus)
    assert "R11_offset_is_not_a_lag" not in _rules(r)


# 12. predicted transition != biological confirmation
def test_12_predicted_transition_is_never_confirmed_by_the_consensus(prediction_only_context, corpus):
    r = validate_answer("La transition predite t7 -> t8 est compatible avec l'ordre du Consensus.",
                        prediction_only_context, corpus)
    c = _first(r)
    assert c.status == PARTIALLY_SUPPORTED
    assert "R14_consensus_never_validates_a_prediction" in c.rules_fired
    assert "vrai pour cet embryon" in c.qualification
    assert c.status != SUPPORTED


# 13. annotated skip != predicted skip
def test_13_annotated_skip_is_not_predicted_skip(observed_only_context, corpus):
    r = validate_answer("La sequence predite presente un saut de phase.", observed_only_context, corpus)
    assert "R3_annotated_skip_is_not_predicted_skip" in _rules(r) and r.has_violation


# 14. phase skip != direct cleavage
def test_14_phase_skip_is_not_direct_cleavage(full_context, corpus):
    r = validate_answer("Le saut de t5 correspond a un direct cleavage.", full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"
    assert "saut de phase n'est pas un direct cleavage" in c.qualification


# 15. phase return != reverse cleavage
def test_15_phase_return_is_not_reverse_cleavage(full_context, corpus):
    r = validate_answer("Le retour de phase est un reverse cleavage.", full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"
    assert "retour de phase n'est pas un reverse cleavage" in c.qualification
    # naming the MODEL phase as a fact on top of it is the more severe defect
    # and wins (R2 NOT_SUPPORTED > R8 NOT_ASSESSABLE), but R8 still fires
    r = validate_answer("Le retour a t7 est un reverse cleavage.", full_context, corpus)
    assert {"R2_prediction_is_not_observation", "R8_definition_absent_from_corpus"} <= set(_first(r).rules_fired)
    assert _first(r).status == NOT_SUPPORTED


# 16. phase labels alone != chaotic division
def test_16_labels_alone_are_not_chaotic_division(full_context, corpus):
    r = validate_answer("La sequence irreguliere de phases est une division chaotique.",
                        full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"


# 17. never assume IVF vs ICSI
def test_17_never_assume_insemination_method(full_context, corpus):
    r = validate_answer("Pour un embryon ICSI, t4 est attendu a 38 hpi.", full_context, corpus)
    c = _first(r)
    assert c.status == NOT_ASSESSABLE and c.reason_code == "unknown_insemination_method"


# 18. keep the Consensus evidence level
def test_18_evidence_grade_is_carried_with_the_claim(full_context, corpus, monkeypatch):
    """The grade printed by the corpus travels with the evidence, unchanged,
    into the claim and its serialised form. Retrieval is pinned to the graded
    block so the test measures conservation, not lexical ranking (the lexical
    search is English-term based and a French claim may rank the Table 4 row
    below the recommendation rows -- a known v1 limitation)."""
    if not corpus.available:
        pytest.skip("corpus absent")
    graded = corpus.get("IC2025-AC-C01")
    assert graded is not None and graded.evidence_grade and "Low" in graded.evidence_grade
    import validator.validator as vv
    monkeypatch.setattr(vv, "search", lambda corpus_, text, markers, top_k=3: [(graded, 1.0)])
    r = validate_answer("Le direct cleavage est associe a une aneuploidie plus elevee selon le "
                        "Consensus.", full_context, corpus)
    evidence = [e for c in r.claims for e in c.evidence]
    assert evidence and evidence[0].evidence_grade == graded.evidence_grade
    assert evidence[0].to_dict()["evidence_grade"] == graded.evidence_grade
    assert evidence[0].page == graded.page and evidence[0].source == "ISTANBUL_CONSENSUS_2025"


# 19. keep the Consensus limitations
def test_19_limitations_are_preserved_and_printed(full_context, corpus):
    raw = "La transition a eu lieu a 43.1 heures."
    r = validate_answer(raw, full_context, corpus)
    assert "IC2025-CP1-01" in _first(r).limitations_preserved
    final = qualified_answer(raw, r)
    assert "limitations conservees : " in final and "IC2025-CP1-01" in final


# 20. no invented clinical causality
def test_20_no_invented_clinical_causality(full_context, corpus):
    r = validate_answer("Cet embryon doit etre transfere car le modele predit t8.", full_context, corpus)
    assert "R13_clinical_claims_extra_caution" in _rules(r)
    assert _first(r).status in (NOT_ASSESSABLE, NOT_SUPPORTED)
    assert "IC2025-STATUS-01" in _first(r).limitations_preserved


# --- structural guarantees ---------------------------------------------------

def test_raw_answer_is_preserved_verbatim_in_the_qualified_answer(full_context, corpus):
    raw = "L'embryon etait en t7.\n\nSource : current_phase."
    r = validate_answer(raw, full_context, corpus)
    final = qualified_answer(raw, r)
    assert final.startswith(raw)
    assert final != raw


def test_provenance_taxonomy_resolution(full_context, corpus):
    assert validate_answer("Le modele predit t7.", full_context, corpus).claims[0].provenance == MODEL_DERIVED
    assert validate_answer("L'annotation indique t4.", full_context, corpus).claims[0].provenance == OBSERVED

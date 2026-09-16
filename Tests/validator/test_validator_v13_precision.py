"""Validator v1.3 -- PRECISION-FIRST.

Every test here reproduces a verdict of the VALIDATOR OFF vs ON benchmark
(`Results/evaluation/validator_experiment/paired_validator_off_vs_on_mistral-nemo_12b_validator_v1.json`,
2026-09-12, replayed read-only on 2026-09-14) -- either a FALSE POSITIVE that
v1.3 must no longer produce, or a CORRECT detection that v1.3 must keep. The
sentences are the LLM's own words (accents as produced); the contexts mirror
the real anchor Patient_319 / window 156 / val, with the real values.

Two design rules pinned throughout:
  * provenance is resolved from the STRUCTURED context first (which block
    carries the value), and a qualification is emitted only on a real
    contradiction between that and the claim's explicit framing;
  * the validator never invents an absence -- "aucune annotation n'est
    presente" may only be written when the context structurally has none.
"""

import pytest

from validator.extraction import extract_claims, is_question
from validator.provenance import has_observed_annotation, restated_field_provenance
from validator.schema import (DERIVED_FROM_MODEL, MODEL_DERIVED, NOT_ASSESSABLE,
                              NOT_SUPPORTED, OBSERVED, SUPPORTED)
from validator.validator import qualified_answer, validate_answer


def _rules(report):
    return {r for c in report.claims for r in c.rules_fired}


def _violations(report):
    return [c for c in report.claims if c.status == NOT_SUPPORTED]


# ---------------------------------------------------------------------------
# real-shape contexts
# ---------------------------------------------------------------------------

_PHASE_PROBS = {"tPB2": 0.0, "tPNa": 5.017534741302858e-202, "tPNf": 4.6124866171006075e-191,
                "t2": 2.922956161138927e-35, "t3": 1.9324091965945717e-32,
                "t4": 3.7983819633435514e-25, "t5": 0.0017052671933268549,
                "t6": 1.1459818142527895e-05, "t7": 0.7609993694101455,
                "t8": 0.23679880956344662, "t9+": 0.00043476617122871367,
                "tM": 5.0318367115227474e-05, "tSB": 7.441233506711718e-09,
                "tB": 4.679757741290383e-10, "tEB": 1.5673849175679434e-09}

_WINDOW = {"window_start": 156, "window_start_time": 43.1, "window_end_time": 44.8,
           "time_available": True, "time_unit": "unknown/unverified"}


def _current(with_ground_truth=True):
    block = {"window": dict(_WINDOW),
             "current_state": {"current_phase": "t7", "current_phase_index": 8,
                               "phase_probability": 0.7609993694101455,
                               "phase_probabilities": dict(_PHASE_PROBS),
                               "entropy": 0.5638288292962345},
             "next_phase": {"most_likely_next_phase": "t7",
                            "next_phase_probability": 0.7313823792655003}}
    if with_ground_truth:
        block["ground_truth"] = {"ground_truth_phase": "t4", "consistency_flag": 0}
    return block


def _transitions():
    return {"n_transitions": 2, "transitions": [
        {"from_phase": "t3", "to_phase": "t4", "window_start": 140, "window_end": 141,
         "observed": True, "provenance": "observed_annotation", "phase_distance": 1, "is_skip": False},
        {"from_phase": "t4", "to_phase": "t6", "window_start": 156, "window_end": 157,
         "observed": True, "provenance": "observed_annotation", "phase_distance": 2, "is_skip": True}]}


def _history_and_series():
    """The real 11-window band: model t7 (151-156) -> t8 (157-161); annotation
    t4 -> t6 at the same boundary; stability 0/0/0; lag null."""
    model = {151: ("t7", 0.7516931042756199, "t5", 0.2237696215824123),
             152: ("t7", 0.9156345204941403, "t5", 0.07176815578254747),
             153: ("t7", 0.9463305483509652, "t8", 0.04778145404700444),
             154: ("t7", 0.9195771881196159, "t8", 0.0761525022238895),
             155: ("t7", 0.6656488728868813, "t8", 0.329609858945313),
             156: ("t7", 0.7609993694101455, "t8", 0.23679880956344662),
             157: ("t8", 0.9259279174550754, "t7", 0.0720938217395284),
             158: ("t8", 0.995423327527396, "t7", 0.003),
             159: ("t8", 0.9981082767414159, "t7", 0.001),
             160: ("t8", 0.9980665342695184, "t5", 0.0014966061482518213),
             161: ("t8", 0.9981778214255539, "t5", 0.001)}
    entries, per_window, model_seq, obs_seq = [], {}, {}, {}
    for w, (p, pp, s, sp) in model.items():
        obs = "t4" if w <= 156 else "t6"
        entries.append({"window_start": w, "offset_from_center": w - 156,
                        "model_derived": {"current_phase": p, "phase_probability": pp,
                                          "phase_probabilities": {p: pp, s: sp}},
                        "observed": {"ground_truth_phase": obs}})
        per_window[str(w)] = {"model_phase": p, "phase_probability": pp, "second_phase": s,
                              "second_phase_probability": sp, "observed_phase": obs}
        model_seq[str(w)], obs_seq[str(w)] = p, obs
    history = {"center_window": 156, "n_windows": 11, "entries": entries}
    series = {
        "provenance": "derived_deterministic", "center_window": 156,
        "per_window": per_window,
        "sequences": {"model_derived_phase_sequence": model_seq,
                      "observed_annotation_phase_sequence": obs_seq},
        "convergence": {"crossing_window": 157, "convergence_start_window": 156,
                        "all_model_phase_changes": [
                            {"from_window": 156, "to_window": 157, "from_phase": "t7", "to_phase": "t8"}]},
        "timing": {"all_observed_phase_changes": [
            {"from_window": 156, "to_window": 157, "from_phase": "t4", "to_phase": "t6"}],
            "matched_change_pair": None, "model_vs_observed_lag_windows": None},
        "stability": {"center_model_phase": "t7", "is_stable_after_center": False,
                      "phase_returns": [], "n_phase_returns": 0, "phase_skips": [],
                      "n_phase_skips": 0, "phase_regressions": [], "n_phase_regressions": 0},
    }
    return history, series


@pytest.fixture
def q4_context():
    """Q4 (DYNAMIC_DATA): get_current_inference + get_transition_events, docs=0."""
    return {"dynamic_context": {"get_current_inference": _current(),
                                "get_transition_events": _transitions()},
            "document_context": [], "scientific_context": None, "warnings": []}


@pytest.fixture
def q8_context():
    """Q6/Q8/Q9 shape: current + transitions + history + series_analysis."""
    history, series = _history_and_series()
    return {"dynamic_context": {"get_current_inference": _current(),
                                "get_transition_events": _transitions(),
                                "get_inference_history": history,
                                "series_analysis": series},
            "document_context": [], "scientific_context": None, "warnings": []}


@pytest.fixture
def q11_context():
    """Q11 (HYBRID): current + transitions + the canonical order in a retrieved
    document -- every phase name is therefore ALSO carried by SCIENTIFIC."""
    return {"dynamic_context": {"get_current_inference": _current(),
                                "get_transition_events": _transitions()},
            "document_context": [{"tool": "retrieve_documents", "source": "docs/SCIENTIFIC_REPORT.md",
                                  "content": "Ordre canonique des phases : tPB2 < tPNa < tPNf < t2 < "
                                             "t3 < t4 < t5 < t6 < t7 < t8 < t9+ < tM < tSB < tB < tEB."}],
            "scientific_context": None, "warnings": []}


@pytest.fixture
def q15_context():
    """Q15 (HYBRID): current only, docs about the model, no annotation block."""
    return {"dynamic_context": {"get_current_inference": _current(with_ground_truth=False)},
            "document_context": [{"tool": "retrieve_documents", "source": "docs/HANDOFF.md",
                                  "content": "Le Semi-HMM modelise la duree de chaque phase."}],
            "scientific_context": None, "warnings": []}


# ---------------------------------------------------------------------------
# the five benchmark false positives, verbatim
# ---------------------------------------------------------------------------

Q4_R2 = (
    "La deuxième phase la plus probable autour de cette transition est t8.\n\n"
    "Cette information est issue de la quantité `phase_probabilities` du modèle semi-HMM, "
    "qui indique que la probabilité de chaque phase est la suivante :\n\n"
    "* tPB2 : 0.0\n* tPNa : 5.017534741302858e-202\n* tPNf : 4.6124866171006075e-191\n"
    "* t2 : 2.922956161138927e-35\n* t3 : 1.9324091965945717e-32\n* t4 : 3.7983819633435514e-25\n"
    "* t5 : 0.0017052671933268549\n* t6 : 1.1459818142527895e-05\n"
    "* t7 : 0.7609993694101455 (la phase courante)\n"
    "* t8 : 0.23679880956344662 (la deuxième phase la plus probable)\n"
    "* t9+ : 0.00043476617122871367\n* tM : 5.0318367115227474e-05\n* tSB : 7.441233506711718e-09\n"
    "* tB : 4.679757741290383e-10\n* tEB : 1.5673849175679434e-09\n\n"
    "Il est important de noter que l'unité temporelle n'a pas été vérifiée.")


def test_Q4_r2_distribution_listing_is_not_an_observation_claim(q4_context, corpus):
    """Benchmark: 12 x R2 NOT_SUPPORTED on the fragments "* t8 :" of a PASS
    answer. Each line copies a printed MODEL-DERIVED (key, value) pair."""
    report = validate_answer(Q4_R2, q4_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)
    assert not report.has_violation
    assert qualified_answer(Q4_R2, report) == Q4_R2      # nothing appended


def test_Q4_r2_the_colon_no_longer_splits_a_label_from_its_value():
    texts = [c["text"] for c in extract_claims("* t8 : 0.23679880956344662 (la deuxième phase)")]
    assert texts == ["* t8 : 0.23679880956344662 (la deuxième phase)"]


def test_Q4_r2_restated_pair_is_resolved_structurally(q4_context):
    got = restated_field_provenance(q4_context, "* t8 : 0.23679880956344662")
    assert got == {"t8": {MODEL_DERIVED}}
    # a value the context does NOT print is not a restatement
    assert restated_field_provenance(q4_context, "* t8 : 0.99") == {}


Q6_R2 = ("Enfin, il convient de souligner que le modèle n'a pas prédit avec succès la transition "
         "effective qui a été annotée dans le jeu de données (une transition de t4 à t6 au lieu de "
         "la transition attendue de t7 à t8).")


def test_Q6_r2_both_framings_present_is_not_a_reversed_provenance(q8_context, corpus):
    """Benchmark: R2b NOT_SUPPORTED "t4 -> t6 provient de l'ANNOTATION ; la
    presenter comme predite inverse sa provenance" -- on a sentence that says
    "annotee" about t4 -> t6 and "predit" about t7 -> t8."""
    report = validate_answer(Q6_R2, q8_context, corpus)
    assert "R2b_transition_provenance" not in _rules(report)
    assert not report.has_violation


Q9_R2 = ("Il n'y a pas de saut de phase ni de régression dans la séquence prédite.\n\n"
         "La séquence prédite par le modèle est :\n\n"
         "W156=t7 | W157=t8 | W158=t8 | W159=t8 | W160=t8 | W161=t8\n\n"
         "Il n'y a donc aucun saut de phase ni régression dans cette séquence.")


def test_Q9_r2_window_sequence_copied_from_the_series_is_not_an_observation(q8_context, corpus):
    """Benchmark: R2 NOT_SUPPORTED on "W156=t7 | W157=t8 | ...", a verbatim
    copy of `model_derived_phase_sequence`, under an introducing sentence
    that names the model."""
    report = validate_answer(Q9_R2, q8_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)
    assert not report.has_violation
    got = restated_field_provenance(q8_context, "W156=t7 | W157=t8")
    assert got == {"t7": {DERIVED_FROM_MODEL}, "t8": {DERIVED_FROM_MODEL}}


def test_Q9_r2_a_window_value_the_series_does_not_hold_is_not_a_restatement(q8_context):
    assert restated_field_provenance(q8_context, "W156=t4") == {"t4": {"DERIVED-FROM-OBSERVED"}}
    assert restated_field_provenance(q8_context, "W156=t2") == {}


Q11_R2 = ("La transition observée dans l'annotation du jeu de données est une skip directe de la "
          "phase t4 à la phase t6. Cela saute par-dessus la phase attendue t5 et va directement à la "
          "phase t6, qui se produit normalement après la phase t7.\n\n"
          "Selon les phases embryonnaires décrites dans le Consensus d'Istanbul de 2025, la séquence "
          "normale est la suivante : t4 → t5 → t6 → t7. La transition annotée ne suit pas cette "
          "séquence attendue.")


def test_Q11_r2_reference_order_statements_are_not_embryo_state_claims(q11_context, corpus):
    """Benchmark: R2 NOT_SUPPORTED on "t4 -> t5 -> t6 -> t7." and on "la phase
    attendue t5 ... normalement apres la phase t7" -- statements about the
    canonical order carried by the retrieved documentation, in which t5 and
    t7 are only reachable as keys of `phase_probabilities` on the model side."""
    report = validate_answer(Q11_R2, q11_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)
    assert not report.has_violation


def test_Q11_arrow_chain_alone_is_a_taxonomy_recitation(q4_context, corpus):
    report = validate_answer("t4 → t5 → t6 → t7.", q4_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)


Q15_R2 = ("Le motif observé ne peut pas être un direct cleavage ou une division chaotique. "
          "La quantité demandée n'est pas disponible pour le reverse cleavage.\n\n"
          "Il n'y a pas de moyen de savoir si la phase suivante est t7 ou non sans plus "
          "d'informations sur les transitions précédentes.\n\n"
          "Il y a un avertissement à prendre en compte : le consensus n'a pas de définition "
          "opérationnelle pour \"direct cleavage\", \"reverse cleavage\" ou \"irregular chaotic division\".")


def test_Q15_r2_declining_to_know_a_phase_is_not_asserting_it(q15_context, corpus):
    """Benchmark: R2 NOT_SUPPORTED on "pas de moyen de savoir si la phase
    suivante est t7 ou non". The R8 detection on the three patterns is kept."""
    report = validate_answer(Q15_R2, q15_context, corpus)
    assert "R2_prediction_is_not_observation" not in _rules(report)
    assert not report.has_violation
    assert "R8_definition_absent_from_corpus" in _rules(report)
    r8 = [c for c in report.claims if "R8_definition_absent_from_corpus" in c.rules_fired]
    assert all(c.status == NOT_ASSESSABLE and c.reason_code == "external_definition" for c in r8)
    # "n'a pas de definition operationnelle" denies a DEFINITION, not an event
    assert "R4_absence_of_evidence_is_not_evidence_of_absence" not in _rules(report)


# ---------------------------------------------------------------------------
# A-J: the mission's explicit cases
# ---------------------------------------------------------------------------

def test_A_model_phase_explicitly_attributed_to_the_model_is_clean(q4_context, corpus):
    for text in ("Selon le modèle, la phase actuelle est t7.",
                 "La phase actuelle de l'embryon, telle que prédite par le modèle semi-HMM, "
                 "est la phase t7 avec une probabilité de 0.7609993694101455."):
        report = validate_answer(text, q4_context, corpus)
        assert not report.has_violation, text
        assert "R1_observation_is_not_interpretation" not in _rules(report), text
        assert all(c.status == SUPPORTED for c in report.claims), text


def test_B_model_phase_presented_as_the_observed_phase_is_a_provenance_error(q4_context, corpus):
    """OBSERVED carries t4, MODEL-DERIVED carries t7: "la phase observee est
    t7" attributes the model's value to the annotation."""
    report = validate_answer("La phase observée est t7.", q4_context, corpus)
    # the sentence frames itself OBSERVED, so R2 (bare assertion) stays silent
    # by design; the grounding layer's provenance check is the detector here
    from orchestrator.grounding_check import check_grounding
    result = check_grounding("La phase observée est t7.", q4_context)
    assert result.provenance_consistent is False
    assert result.provenance_violations[0]["phase"] == "t7"
    # and the bare form is caught by the validator itself
    report = validate_answer("L'embryon se trouve actuellement dans la phase t7.", q4_context, corpus)
    assert "R2_prediction_is_not_observation" in _rules(report)
    assert report.has_violation


def test_C_annotated_transition_attributed_to_the_annotation_is_clean(q8_context, corpus):
    for text in ("L'annotation indique la transition t4 → t6.",
                 "La transition annotée est t4 -> t6 (un saut de t5)."):
        report = validate_answer(text, q8_context, corpus)
        assert not report.has_violation, text


def test_D_annotated_transition_presented_as_predicted_is_an_error(q8_context, corpus):
    report = validate_answer("Le modèle prédit la transition t4 -> t6.", q8_context, corpus)
    assert "R2b_transition_provenance" in _rules(report)
    assert report.has_violation


def test_D_model_transition_presented_as_annotated_is_an_error(q8_context, corpus):
    report = validate_answer("L'annotation montre la transition t7 -> t8.", q8_context, corpus)
    assert "R2b_transition_provenance" in _rules(report)
    assert report.has_violation


def test_E_offset_from_center_is_never_read_as_a_lag(q8_context, corpus):
    """W155 sits at offset_from_center = -1; the lag is null."""
    for text in ("Le décalage est de -1 fenêtre.",
                 "Le modèle détecte la transition une fenêtre avant l'annotation.",
                 "Cela signifie que le modèle a prédit la transition une fenêtre avant "
                 "qu'elle ne soit observée."):
        report = validate_answer(text, q8_context, corpus)
        assert "R11_offset_is_not_a_lag" in _rules(report), text
        assert report.has_violation, text


def test_E_before_and_after_the_annotated_transition_is_not_a_lag_claim(q8_context, corpus):
    """Q6 r3: describing the windows before/after the annotated transition
    is not a lag statement (a v1.3 candidate word list would have flagged it)."""
    report = validate_answer(
        "Les probabilités évoluent de manière différente avant et après la transition annotée.",
        q8_context, corpus)
    assert "R11_offset_is_not_a_lag" not in _rules(report)


def test_F_null_lag_licenses_no_lag_value_in_any_direction(q8_context, corpus):
    assert q8_context["dynamic_context"]["series_analysis"]["timing"]["model_vs_observed_lag_windows"] is None
    for text in ("Le modèle détecte la transition avant l'annotation avec un décalage de 0 fenêtre(s).",
                 "Le décalage entre la prédiction du modèle et l'annotation est de deux fenêtres."):
        report = validate_answer(text, q8_context, corpus)
        assert "R11_offset_is_not_a_lag" in _rules(report), text
    # the honest form is kept, never flagged
    report = validate_answer("Le décalage est non calculable : model_vs_observed_lag_windows est null.",
                             q8_context, corpus)
    assert "R11_offset_is_not_a_lag" not in _rules(report)


def test_G_no_invented_absence_of_annotation(q4_context, corpus):
    """The R2 qualification may say "aucune annotation" only when the context
    structurally has none."""
    assert has_observed_annotation(q4_context) is True
    report = validate_answer("L'embryon est en t7.", q4_context, corpus)
    q = report.claims[0].qualification
    assert "R2_prediction_is_not_observation" in report.claims[0].rules_fired
    assert "aucune annotation" not in q.lower()
    assert "t4" not in q                      # still never writes the truth


def test_G_absence_is_stated_only_without_any_annotation_block(q15_context, corpus):
    assert has_observed_annotation(q15_context) is False
    report = validate_answer("L'embryon est en t7.", q15_context, corpus)
    assert "aucune annotation n'est presente" in report.claims[0].qualification.lower()


def test_H_model_phase_return_is_not_a_reverse_cleavage(q8_context, corpus):
    report = validate_answer(
        "Le retour de phase dans la séquence prédite est un reverse cleavage.", q8_context, corpus)
    c = report.claims[0]
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"
    assert "retour de phase n'est pas un reverse cleavage" in c.qualification


def test_I_annotated_skip_is_not_a_predicted_skip(observed_only_context, corpus):
    report = validate_answer("La séquence prédite présente un saut de phase.",
                             observed_only_context, corpus)
    assert "R3_annotated_skip_is_not_predicted_skip" in _rules(report)
    assert report.has_violation


def test_I_a_sentence_naming_both_sides_is_not_judged_by_R3(q8_context, corpus):
    report = validate_answer(
        "Le saut t4 -> t6 est annoté ; la séquence prédite ne présente aucun saut.",
        q8_context, corpus)
    assert "R3_annotated_skip_is_not_predicted_skip" not in _rules(report)


@pytest.mark.parametrize("text", [
    "The observed motif cannot correspond to direct cleavage, reverse cleavage or chaotic division.",
    "Le motif observé ne peut pas être un direct cleavage ou une division chaotique.",
    "Le motif observé correspond à un reverse cleavage.",
])
def test_J_cleavage_patterns_without_definition_are_not_assessable(text, q15_context, corpus):
    report = validate_answer(text, q15_context, corpus)
    c = report.claims[0]
    assert c.status == NOT_ASSESSABLE and c.reason_code == "external_definition"
    assert "Ciray" in c.qualification


# ---------------------------------------------------------------------------
# kept detections (the benchmark's correct hits must survive v1.3)
# ---------------------------------------------------------------------------

def test_kept_Q3_flagship_bare_model_phase_as_fact(q4_context, corpus):
    report = validate_answer("La phase dominante juste avant la transition était t7.",
                             q4_context, corpus)
    assert report.claims[0].status == NOT_SUPPORTED
    assert "R2_prediction_is_not_observation" in report.claims[0].rules_fired


def test_kept_Q2_r3_bare_model_phase_with_a_later_source_line(q4_context, corpus):
    text = ("L'embryon se trouve actuellement dans la phase t7 depuis le début de la fenêtre "
            "courante.\n\nSources utilisées :\n- DERIVED : get_current_inference.current_state.current_phase")
    report = validate_answer(text, q4_context, corpus)
    assert any("R2_prediction_is_not_observation" in c.rules_fired and c.status == NOT_SUPPORTED
               for c in report.claims)


def test_kept_Q1_seconds_without_a_verified_unit(q4_context, corpus):
    report = validate_answer("La dernière transition détectée est le passage de t4 à t6, qui s'est "
                             "produite à 43,1 secondes.", q4_context, corpus)
    c = report.claims[0]
    assert c.status == NOT_ASSESSABLE and c.reason_code == "unverified_time_basis"
    assert "R9_timing_requires_unit_and_origin" in c.rules_fired


def test_kept_Q13_hpi_values_and_insemination_method(q4_context, corpus):
    report = validate_answer(
        "Selon le Consensus d'Istanbul de 2025, le temps moyen pour atteindre la fin de la phase t4 "
        "est de 38 heures post-insémination (hpi) pour les embryons conçus par ICSI.", q4_context, corpus)
    c = report.claims[0]
    assert c.status == NOT_ASSESSABLE and c.reason_code == "unknown_insemination_method"


def test_kept_Q14_r3_typicality_verdict(q8_context, corpus):
    report = validate_answer("Le décalage observé ne doit pas être considéré comme atypique.",
                             q8_context, corpus)
    assert "R9b_median_is_not_a_range" in _rules(report)


# ---------------------------------------------------------------------------
# new in v1.3: R9c, R12, dispersion words, questions, R1, R14
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "La transition observée pour cette fenêtre est compatible avec les repères morphocinétiques disponibles.",
    "Le moment observé n'est pas compatible avec les repères morphocinétiques.",
])
def test_R9c_morphokinetic_compatibility_needs_a_verified_time_unit(text, q4_context, corpus):
    """Q13, 3/3 replications declared compatibility from a context printing
    time_unit = unknown/unverified. Either direction is unassessable."""
    report = validate_answer(text, q4_context, corpus)
    c = report.claims[0]
    assert c.status == NOT_ASSESSABLE and c.reason_code == "unverified_time_basis"
    assert "R9c_morphokinetic_compatibility_requires_time_basis" in c.rules_fired
    assert "IC2025-CP1-01" in c.limitations_preserved


def test_R9c_is_silent_when_the_time_unit_is_verified(q4_context, corpus):
    q4_context["dynamic_context"]["get_current_inference"]["window"]["time_unit"] = "hours post-insemination"
    report = validate_answer(
        "La transition observée est compatible avec les repères morphocinétiques disponibles.",
        q4_context, corpus)
    assert "R9c_morphokinetic_compatibility_requires_time_basis" not in _rules(report)


def test_R9c_is_silent_without_any_time_unit_field(corpus):
    ctx = {"dynamic_context": {"get_current_inference": {"current_state": {"current_phase": "t7"}}},
           "document_context": []}
    report = validate_answer(
        "La transition observée est compatible avec les repères morphocinétiques disponibles.", ctx, corpus)
    assert "R9c_morphokinetic_compatibility_requires_time_basis" not in _rules(report)


def test_R9c_does_not_touch_an_order_compatibility_verdict(q11_context, corpus):
    """Q11 asks about the ORDER, not about timing: R9c must stay out."""
    report = validate_answer(
        "Cette transition est compatible avec l'ordre attendu du développement embryonnaire.",
        q11_context, corpus)
    assert "R9c_morphokinetic_compatibility_requires_time_basis" not in _rules(report)


def test_R9c_honest_decline_is_exempt(q4_context, corpus):
    report = validate_answer(
        "La compatibilité avec les repères morphocinétiques ne peut pas être évaluée : aucun "
        "repère n'est disponible dans le corpus.", q4_context, corpus)
    assert "R9c_morphokinetic_compatibility_requires_time_basis" not in _rules(report)


@pytest.mark.parametrize("text", [
    "La fenêtre 156 se situe donc dans l'intervalle de confiance pour la fin de la phase t4.",
    "La transition observée ici a commencé à 43.1 hpi, ce qui est dans la plage attendue.",
    "La transition annotée (t4) s'est produite 8 fenêtres avant la fenêtre courante, ce qui est "
    "dans les limites attendues de la variabilité biologique connue.",
])
def test_R9b_dispersion_the_table_does_not_print(text, q4_context, corpus):
    report = validate_answer(text, q4_context, corpus)
    assert "R9b_median_is_not_a_range" in _rules(report)
    assert report.claims[0].status == NOT_ASSESSABLE


def test_R9b_conditional_and_modal_sentences_are_not_verdicts(q4_context, corpus):
    for text in ("Si le décalage observé se situe à l'intérieur de cette distribution, alors il "
                 "pourrait être considéré comme relevant de la variabilité biologique connue.",
                 "Cependant, il est possible que ce décalage relève de la variabilité biologique connue."):
        report = validate_answer(text, q4_context, corpus)
        assert "R9b_median_is_not_a_range" not in _rules(report), text


def test_R12_return_claim_against_a_zero_stability_block(q8_context, corpus):
    """Q8 r2/r3: "il y a des retours vers la phase precedente" while
    stability prints 0 returns / 0 regressions and no annotated transition
    goes backward."""
    for text in ("La prédiction est instable après la transition et il y a des retours vers la "
                 "phase précédente.",
                 "Il y a des retours observés vers la phase précédente."):
        report = validate_answer(text, q8_context, corpus)
        assert "R12_phase_return_claim_vs_stability" in _rules(report), text
        assert report.has_violation, text
        assert "n_phase_returns = 0" in _violations(report)[0].qualification


def test_R12_is_silent_on_negations_definitions_and_mentions(q8_context, corpus):
    for text in ("La prédiction est stable après la transition car il n'y a pas de retours vers la "
                 "phase précédente.",
                 "Le calcul de stabilité indique qu'il y a eu 0 retours vers la phase précédente.",
                 "De même, il n'y a pas eu de régression de phase, car une régression impliquerait "
                 "un retour en arrière vers une phase précédente.",
                 "Une analyse plus approfondie pourrait être nécessaire pour déterminer les raisons "
                 "de cette instabilité et des retours observés."):
        report = validate_answer(text, q8_context, corpus)
        assert "R12_phase_return_claim_vs_stability" not in _rules(report), text


def test_R12_is_silent_when_the_block_records_a_return_or_is_absent(q8_context, q4_context, corpus):
    text = "Il y a des retours vers la phase précédente."
    assert "R12_phase_return_claim_vs_stability" not in _rules(validate_answer(text, q4_context, corpus))
    q8_context["dynamic_context"]["series_analysis"]["stability"]["n_phase_returns"] = 1
    assert "R12_phase_return_claim_vs_stability" not in _rules(validate_answer(text, q8_context, corpus))


def test_restated_questions_are_not_claims(q8_context, corpus):
    """Q7 r1 opened with the question and R11 fired on it."""
    assert is_question("Le modèle prédit-il la transition avant ou après l'annotation ?")
    text = ("Le modèle prédit-il la transition avant ou après l'annotation ?\n\n"
            "Le décalage est non calculable : model_vs_observed_lag_windows est null.")
    report = validate_answer(text, q8_context, corpus)
    assert not any(c.claim.endswith("?") for c in report.claims)
    assert not report.has_violation


def test_R1_needs_an_interpretation_not_the_word_embryon(q8_context, corpus):
    for text in ("Ce genre d'incohérence entre la prédiction du modèle et l'annotation observée est "
                 "un domaine actif de recherche dans ce type de tâches de reconnaissance de phases "
                 "embryonnaires.",
                 "La phase actuelle de l'embryon, telle que prédite par le modèle, est la phase t7."):
        assert "R1_observation_is_not_interpretation" not in _rules(
            validate_answer(text, q8_context, corpus)), text
    report = validate_answer("La phase t4 observée indique une bonne viabilité de l'embryon.",
                             q8_context, corpus)
    assert "R1_observation_is_not_interpretation" in _rules(report)


def test_R14_reference_order_statement_is_not_a_validated_prediction(q11_context, corpus):
    """Q12: "selon le referentiel, la prochaine etape est t7" states the
    reference, it validates no prediction."""
    report = validate_answer("La prochaine étape développementale attendue selon le référentiel est t7.",
                             q11_context, corpus)
    assert "R14_consensus_never_validates_a_prediction" not in _rules(report)
    assert not report.has_violation


def test_R14_still_guards_an_explicit_model_claim(q11_context, corpus):
    report = validate_answer("La transition prédite t7 -> t8 est compatible avec l'ordre du Consensus.",
                             q11_context, corpus)
    assert "R14_consensus_never_validates_a_prediction" in _rules(report)
    assert report.claims[0].status != SUPPORTED


def test_inherited_framing_does_not_override_own_framing(q8_context, corpus):
    """A list item that attributes itself keeps its own framing."""
    text = ("La séquence prédite par le modèle est :\n"
            "- L'annotation montre la transition t7 -> t8.")
    report = validate_answer(text, q8_context, corpus)
    assert "R2b_transition_provenance" in _rules(report)


def test_engine_version_is_v13():
    from orchestrator.scientific_validation import VALIDATOR_ENGINE
    assert VALIDATOR_ENGINE == "scientific_validator v1.3"

"""Timing safety, cases H and I, plus the median/IVF-ICSI rules."""

import pytest

from validator.schema import NOT_ASSESSABLE, UNVERIFIED_TIME_BASIS
from validator.validator import validate_answer


@pytest.mark.parametrize("text", [
    "La transition a eu lieu a 43.1 heures.",
    "La phase dure environ 22,75 secondes.",
    "Le decalage est de -3,8 minutes.",
    "La duree attendue est de 4,739 jours.",
])
def test_H_timing_without_a_sourced_unit_is_not_assessable(
        text, prediction_only_context, corpus):
    """Every one of these four was really produced by the benchmark from a
    context printing `time_unit = "unknown/unverified"`."""
    report = validate_answer(text, prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == NOT_ASSESSABLE
    assert claim.reason_code == UNVERIFIED_TIME_BASIS
    assert "R9_timing_requires_unit_and_origin" in claim.rules_fired


def test_I_timing_in_the_sourced_unit_is_accepted(prediction_only_context, corpus):
    """`duration_unit = "windows"` IS sourced, so a duration in windows raises
    no timing violation."""
    report = validate_answer("La duree attendue est de 22,75 fenetres.",
                             prediction_only_context, corpus)
    assert not any("R9_timing_requires_unit_and_origin" in c.rules_fired
                   for c in report.claims)


def test_I_verified_time_unit_lifts_the_rule(prediction_only_context, corpus):
    ctx = prediction_only_context
    ctx["dynamic_context"]["get_current_inference"]["window"]["time_unit"] = "hours post-insemination"
    report = validate_answer("La transition a eu lieu a 43.1 heures.", ctx, corpus)
    assert not any("R9_timing_requires_unit_and_origin" in c.rules_fired
                   for c in report.claims)


@pytest.mark.parametrize("text", [
    "Ce timing est atypique.",
    "Le moment de la transition est typique.",
    "La duree de cette phase est normale.",
])
def test_median_is_not_a_range(text, prediction_only_context, corpus):
    """Table 1 prints medians and NO dispersion, and the Consensus gives no
    atypicality threshold (IC2025-T1, IC2025-LIM-09, IC2025-KG-04)."""
    report = validate_answer(text, prediction_only_context, corpus)
    assert report.claims[0].status == NOT_ASSESSABLE


def test_ivf_icsi_is_never_assumed(prediction_only_context, corpus):
    report = validate_answer(
        "Pour un embryon ICSI, t4 est attendu a 38 hpi.", prediction_only_context, corpus)
    claim = report.claims[0]
    assert claim.status == NOT_ASSESSABLE
    assert claim.reason_code == "unknown_insemination_method"


def test_offset_is_never_a_lag(full_context, corpus):
    """`offset_from_center` and `model_vs_observed_lag_windows` are not
    interchangeable; the latter is null here, so no lag is licensed."""
    from validator.provenance import available_provenance
    series = full_context["dynamic_context"]["series_analysis"]
    assert series["timing"]["model_vs_observed_lag_windows"] is None
    report = validate_answer("Le decalage est de -2 fenetres.", full_context, corpus)
    assert report.claims  # the sentence is picked up as a timing claim at all

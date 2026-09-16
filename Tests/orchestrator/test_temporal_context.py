"""Tests for the window-blocked temporal context + its deterministic
series analysis (Training/orchestrator/temporal_context.py, 2026-09-02
post-benchmark context restructuring).

Checks exactly what the restructuring is supposed to guarantee:
  - no window is lost, none is invented;
  - the chronological order is preserved and stated;
  - MODEL-DERIVED / OBSERVED / DERIVED provenance is never merged;
  - no value can appear without its own window identifier;
  - top-k probabilities stay internally consistent and self-declared;
  - every derived quantity is arithmetically correct AND visible to
    grounding_check as a KNOWN value.

Synthetic data only -- no torch, no GPU, no Reporting API, no LLM.
"""

from __future__ import annotations

import math

import pytest

from orchestrator import prompt, temporal_context
from orchestrator.temporal_context import (
    PROVENANCE_DERIVED, derive_series_analysis, render_inference_history, render_series_analysis,
)

CENTER = 156

# (window, model_phase, phase_probabilities, next_phase_distribution, observed_phase)
_SERIES = [
    (151, "t4", {"t4": 0.84, "t6": 0.09, "t5": 0.05}, {"t4": 0.77, "t6": 0.22}, "t4"),
    (152, "t4", {"t4": 0.81, "t6": 0.12, "t5": 0.05}, {"t4": 0.72, "t6": 0.26}, "t4"),
    (153, "t4", {"t4": 0.74, "t6": 0.20, "t5": 0.05}, {"t4": 0.63, "t6": 0.36}, "t4"),
    (154, "t4", {"t4": 0.64, "t6": 0.30, "t5": 0.05}, {"t4": 0.51, "t6": 0.47}, "t4"),
    (155, "t4", {"t4": 0.53, "t6": 0.42, "t5": 0.04}, {"t6": 0.58, "t4": 0.40}, "t4"),
    (156, "t6", {"t6": 0.57, "t4": 0.38, "t5": 0.04}, {"t6": 0.77, "t4": 0.21}, "t4"),
    (157, "t6", {"t6": 0.70, "t4": 0.26, "t5": 0.03}, {"t6": 0.86, "t4": 0.13}, "t6"),
    (158, "t6", {"t6": 0.75, "t4": 0.19, "t5": 0.03}, {"t6": 0.89, "t4": 0.08}, "t6"),
    (159, "t4", {"t4": 0.52, "t6": 0.46, "t5": 0.02}, {"t6": 0.63, "t4": 0.36}, "t6"),
    (160, "t6", {"t6": 0.80, "t4": 0.15, "t5": 0.02}, {"t6": 0.93, "t4": 0.05}, "t6"),
    (161, "t6", {"t6": 0.86, "t4": 0.11, "t5": 0.02}, {"t6": 0.95, "t4": 0.04}, "t6"),
]

_PHASE_INDEX = {"t4": 5, "t5": 6, "t6": 7}


def _entry(window, model_phase, probabilities, next_distribution, observed_phase):
    top = max(probabilities.values())
    return {
        "window_start": window,
        "offset_from_center": window - CENTER,
        "is_center": window == CENTER,
        "window_start_time": 40.0 + window * 0.25,
        "window_end_time": 40.0 + (window + 1) * 0.25,
        "time_available": True,
        "model_derived": {
            "current_phase": model_phase,
            "current_phase_index": _PHASE_INDEX[model_phase],
            "phase_probability": top,
            "phase_probabilities": dict(probabilities),
            # entropy of the RENDERED entries only -- enough to give the tests a
            # monotone, checkable series; the real one comes from the full posterior.
            "entropy": round(-sum(p * math.log(p) for p in probabilities.values()), 6),
            "most_likely_next_phase": max(next_distribution, key=next_distribution.get),
            "next_phase_probability": max(next_distribution.values()),
            "next_phase_distribution": dict(next_distribution),
        },
        "observed": {"ground_truth_phase": observed_phase, "consistency_flag": 0},
    }


def _history(series=None, center=CENTER, warnings=None):
    entries = [_entry(*row) for row in (series if series is not None else _SERIES)]
    return {
        "sample": {"sample_id": f"Patient_319:{center}", "video_name": "Patient_319",
                   "patient_id": "319", "split": "val"},
        "center_window": center, "requested_before": 5, "requested_after": 5,
        "n_windows_before": 5, "n_windows_after": 5, "n_windows": len(entries),
        "window_starts": [e["window_start"] for e in entries],
        "entries": entries,
        "model": {"model_name": "semi_hmm", "model_version": "semi_hmm_weekend_phaseF@dmax268"},
        "warnings": list(warnings or []),
    }


@pytest.fixture
def history():
    return _history()


@pytest.fixture
def analysis(history):
    return derive_series_analysis(history)


@pytest.fixture
def rendered(history, analysis):
    return render_inference_history(history) + "\n" + render_series_analysis(analysis)


# --- A. no window lost, none invented, order preserved ----------------------

def test_every_window_gets_its_own_block(rendered):
    for window, *_ in _SERIES:
        assert f"WINDOW {window}" in rendered, f"window {window} lost from the rendered context"


def test_no_window_outside_the_band_is_invented(rendered):
    real = {w for w, *_ in _SERIES}
    for candidate in range(140, 175):
        if candidate not in real:
            assert f"WINDOW {candidate}" not in rendered


def test_window_blocks_appear_in_ascending_chronological_order(rendered):
    positions = [rendered.index(f"WINDOW {w}") for w, *_ in _SERIES]
    assert positions == sorted(positions)


def test_chronological_order_is_stated_explicitly_not_only_implied(rendered):
    assert "ordre chronologique croissant" in rendered


def test_out_of_order_entries_are_re_sorted_never_narrated_out_of_order():
    shuffled = _history()
    shuffled["entries"] = list(reversed(shuffled["entries"]))
    rendered = render_inference_history(shuffled)
    positions = [rendered.index(f"WINDOW {w}") for w, *_ in _SERIES]
    assert positions == sorted(positions)
    assert derive_series_analysis(shuffled)["window_starts"] == [w for w, *_ in _SERIES]


def test_analysis_window_starts_match_the_history_exactly(history, analysis):
    assert analysis["window_starts"] == history["window_starts"]
    assert analysis["n_windows"] == len(history["entries"])
    assert set(analysis["per_window"]) == {str(w) for w, *_ in _SERIES}


# --- B. every value carries its window identifier ---------------------------

def test_no_value_line_appears_without_a_window_prefix(rendered):
    inside_block = False
    for line in render_inference_history(_history()).splitlines():
        if line.startswith("WINDOW "):
            inside_block = True
            continue
        if not line.strip() or line.startswith("==="):
            inside_block = False
            continue
        if inside_block and " = " in line or (inside_block and " : " in line and "  " in line[:4]):
            assert line.strip().startswith("W"), f"value line without window identifier: {line!r}"


def test_each_windows_values_are_prefixed_with_that_windows_own_id(rendered):
    for window, model_phase, probabilities, _, observed_phase in _SERIES:
        assert f"W{window}.model_phase = {model_phase}" in rendered
        assert f"W{window}.observed_phase = {observed_phase}" in rendered
        assert f"W{window}.phase_probability = {max(probabilities.values())}" in rendered


def test_a_value_is_never_attached_to_a_neighbouring_window():
    # W155's leading probability (0.53) must never be rendered as W156's.
    rendered = render_inference_history(_history())
    assert "W156.phase_probability = 0.53" not in rendered
    assert "W155.phase_probability = 0.53" in rendered
    assert "W156.phase_probability = 0.57" in rendered


def test_center_window_is_explicitly_marked(rendered):
    assert f"WINDOW {CENTER}  [FENETRE CENTRALE" in rendered


# --- C. provenance separation ----------------------------------------------

def test_model_observed_and_derived_are_three_labelled_blocks(rendered):
    assert "MODEL-DERIVED" in rendered
    assert "OBSERVED" in rendered
    assert "DERIVED (calcul deterministe Python" in rendered


def test_model_and_observed_phase_disagree_without_being_merged(rendered):
    # window 156: the model already says t6, the annotation still says t4.
    assert "W156.model_phase = t6" in rendered
    assert "W156.observed_phase = t4" in rendered


def test_observed_block_is_labelled_as_never_a_prediction(rendered):
    assert "jamais une prediction" in rendered


def test_analysis_declares_itself_neither_model_output_nor_annotation(analysis):
    assert analysis["provenance"] == PROVENANCE_DERIVED
    assert analysis["is_model_output"] is False
    assert analysis["is_observed_annotation"] is False


def test_derived_gap_is_not_rendered_inside_the_model_derived_block():
    block = render_inference_history(_history()).split("WINDOW 156")[1].split("WINDOW 157")[0]
    model_part, derived_part = block.split("DERIVED (calcul deterministe Python")
    assert "probability_gap" not in model_part
    assert "probability_gap" in derived_part


# --- D. top-k probability consistency ---------------------------------------

def test_rendered_top_k_probabilities_are_descending_and_lead_with_the_model_phase(rendered):
    for line in rendered.splitlines():
        if ".phase_probabilities :" not in line:
            continue
        values = [float(part.split("=")[1]) for part in line.split(":", 1)[1].split("|")]
        assert values == sorted(values, reverse=True), line


def test_second_phase_is_the_runner_up_of_the_rendered_distribution(analysis):
    for window, _, probabilities, _, _ in _SERIES:
        ranked = sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
        entry = analysis["per_window"][str(window)]
        assert entry["second_phase"] == ranked[1][0]
        assert entry["second_phase_probability"] == ranked[1][1]


def test_truncated_distributions_are_disclosed_with_their_omitted_mass():
    truncated = _history(series=[(151, "t4", {"t4": 0.6, "t6": 0.2}, {"t4": 0.9}, "t4"),
                                  (152, "t4", {"t4": 0.6, "t6": 0.2}, {"t4": 0.9}, "t4")],
                          center=151)
    warnings = derive_series_analysis(truncated)["warnings"]
    assert any("truncated" in w for w in warnings)
    assert any("0.2" in w for w in warnings)


def test_untruncated_distribution_produces_no_truncation_warning():
    full = _history(series=[(151, "t4", {"t4": 0.7, "t6": 0.3}, {"t4": 1.0}, "t4"),
                            (152, "t4", {"t4": 0.7, "t6": 0.3}, {"t4": 1.0}, "t4")], center=151)
    assert not any("truncated" in w for w in derive_series_analysis(full)["warnings"])


# --- E. the deterministic computations are arithmetically right -------------

def test_probability_gap_matches_the_two_leading_probabilities(analysis):
    for window, _, probabilities, _, _ in _SERIES:
        ranked = sorted(probabilities.values(), reverse=True)
        assert analysis["per_window"][str(window)]["probability_gap"] == \
            pytest.approx(ranked[0] - ranked[1])


def test_gap_minimum_is_the_real_minimum(analysis):
    series = analysis["probability_gap"]["series"]
    assert analysis["probability_gap"]["minimum"] == min(series.values())
    assert series[str(analysis["probability_gap"]["minimum_window"])] == \
        analysis["probability_gap"]["minimum"]


def test_crossing_window_is_the_first_model_phase_change(analysis):
    assert analysis["convergence"]["crossing_window"] == 156


def test_convergence_start_is_the_run_of_strictly_decreasing_gaps_before_crossing(analysis):
    start = analysis["convergence"]["convergence_start_window"]
    series = analysis["probability_gap"]["series"]
    run = [w for w, *_ in _SERIES if start <= w <= 155]
    gaps = [series[str(w)] for w in run]
    assert gaps == sorted(gaps, reverse=True)


def test_model_vs_observed_lag_is_signed_and_correct(analysis):
    # the model flips t4->t6 at window 156, the annotation at 157 -> lag = -1
    assert analysis["timing"]["model_vs_observed_lag_windows"] == -1


def test_lag_is_null_and_explained_when_no_matching_change_pair_exists():
    flat = _history(series=[(151, "t4", {"t4": 0.9, "t6": 0.1}, {"t4": 0.9}, "t4"),
                            (152, "t4", {"t4": 0.9, "t6": 0.1}, {"t4": 0.9}, "t4")], center=151)
    timing = derive_series_analysis(flat)["timing"]
    assert timing["model_vs_observed_lag_windows"] is None
    assert "not computable" in timing["lag_definition"]


def test_phase_return_and_regression_are_detected_with_both_window_ids(analysis):
    stability = analysis["stability"]
    assert stability["n_phase_returns"] >= 1
    assert stability["n_phase_regressions"] == 1
    regression = stability["phase_regressions"][0]
    assert (regression["from_window"], regression["to_window"]) == (158, 159)
    assert (regression["from_phase"], regression["to_phase"]) == ("t6", "t4")


def test_forward_skip_is_detected_and_a_backward_change_is_never_called_a_skip(analysis):
    skips = analysis["stability"]["phase_skips"]
    assert {(s["from_window"], s["to_window"]) for s in skips} == {(155, 156), (159, 160)}
    assert all(s["phase_distance"] >= 2 for s in skips)


def test_post_center_stability_names_the_offending_window(analysis):
    stability = analysis["stability"]
    assert stability["is_stable_after_center"] is False
    assert stability["windows_after_center_differing_from_center_phase"] == [159]


def test_stability_is_true_when_every_post_center_window_keeps_the_phase():
    stable = _history(series=[(155, "t4", {"t4": 0.9, "t6": 0.1}, {"t4": 0.9}, "t4"),
                              (156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6"),
                              (157, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6")])
    assert derive_series_analysis(stable)["stability"]["is_stable_after_center"] is True


def test_entropy_extrema_and_near_far_comparison_are_correct(analysis):
    entropy = analysis["entropy"]
    series = {int(w): v for w, v in entropy["series"].items()}
    assert entropy["maximum"] == max(series.values())
    assert entropy["minimum"] == min(series.values())
    assert entropy["at_center_window"] == series[CENTER]
    near = [v for w, v in series.items() if abs(w - CENTER) <= 1]
    far = [v for w, v in series.items() if abs(w - CENTER) >= 3]
    assert entropy["mean_near_center_offset_le_1"] == pytest.approx(sum(near) / len(near))
    assert entropy["mean_far_from_center_offset_ge_3"] == pytest.approx(sum(far) / len(far))
    # both means are rounded to 6 decimals before subtraction, so compare at that scale
    assert entropy["near_minus_far"] == pytest.approx(
        sum(near) / len(near) - sum(far) / len(far), abs=1e-6)


def test_entropy_series_covers_every_window(analysis):
    assert set(analysis["entropy"]["series"]) == {str(w) for w, *_ in _SERIES}


# --- F. determinism, degenerate inputs, integration -------------------------

def test_analysis_and_rendering_are_deterministic(history):
    assert derive_series_analysis(history) == derive_series_analysis(_history())
    assert render_inference_history(history) == render_inference_history(_history())


def test_empty_band_returns_nothing_rather_than_a_fabricated_series():
    assert derive_series_analysis({"entries": []}) == {}
    assert derive_series_analysis({}) == {}
    assert "aucune fenetre" in render_inference_history({"entries": []})


def test_single_window_band_has_no_change_no_crossing_no_lag():
    single = _history(series=[(156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6")])
    result = derive_series_analysis(single)
    assert result["convergence"]["crossing_window"] is None
    assert result["timing"]["model_vs_observed_lag_windows"] is None
    assert any("crossing_window is null" in w for w in result["warnings"])


def test_non_contiguous_band_is_flagged_rather_than_silently_renumbered():
    gapped = _history(series=[(151, "t4", {"t4": 0.9, "t6": 0.1}, {"t4": 0.9}, "t4"),
                              (156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6")])
    rendered = render_inference_history(gapped)
    assert "non contigue" in rendered
    assert "WINDOW 152" not in rendered


def test_missing_time_metadata_is_stated_not_faked():
    no_time = _history(series=[(156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6")])
    no_time["entries"][0]["time_available"] = False
    assert "aucune donnee temporelle" in render_inference_history(no_time)


def test_missing_observed_annotation_renders_as_null_never_as_the_model_phase():
    unannotated = _history(series=[(156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, "t6")])
    unannotated["entries"][0]["observed"]["ground_truth_phase"] = None
    assert "W156.observed_phase = null" in render_inference_history(unannotated)


def test_prompt_renders_the_series_structurally_not_by_index_flattening(history, analysis):
    body = prompt._format_dynamic_context(
        {"get_inference_history": history, "series_analysis": analysis})
    assert "WINDOW 156" in body
    assert "entries[5]" not in body
    assert "get_inference_history.entries" not in body


def test_prompt_still_flattens_every_other_tool_unchanged(history):
    body = prompt._format_dynamic_context({
        "get_transition_events": {"n_transitions": 1, "transitions": [
            {"from_phase": "t4", "to_phase": "t6", "window_start": 156, "is_skip": True}]},
        "get_inference_history": history,
    })
    assert "get_transition_events.transitions[0].from_phase = t4" in body
    assert "WINDOW 156" in body


def test_prompt_states_the_center_window_and_single_window_record_are_the_same_window(history):
    body = prompt._format_dynamic_context({
        "get_current_inference": {"window": {"window_start": CENTER},
                                   "current_state": {"current_phase": "t6"}},
        "get_inference_history": history,
    })
    assert "RECONCILIATION" in body
    assert f"meme fenetre que W{CENTER}" in body


def test_no_reconciliation_note_when_the_two_records_are_different_windows(history):
    body = prompt._format_dynamic_context({
        "get_current_inference": {"window": {"window_start": 999},
                                   "current_state": {"current_phase": "t6"}},
        "get_inference_history": history,
    })
    assert "RECONCILIATION" not in body


def test_derived_numbers_are_visible_to_the_grounding_check(history, analysis):
    """A derived quantity the LLM restates must be verifiable exactly like a
    tool-returned one -- which is why series_analysis is a dynamic_context
    entry, not prompt-only text."""
    from orchestrator import grounding_check
    context = {"question": "q", "document_context": [],
               "dynamic_context": {"get_inference_history": history, "series_analysis": analysis},
               "warnings": []}
    answer = (f"L'écart minimal est de {analysis['probability_gap']['minimum']} à la fenêtre "
              f"{analysis['probability_gap']['minimum_window']}, et le décalage est de "
              f"{analysis['timing']['model_vs_observed_lag_windows']} fenêtre.")
    assert grounding_check.check_grounding(answer, context).ungrounded_numbers == []


def test_an_invented_number_is_still_caught_after_restructuring(history, analysis):
    from orchestrator import grounding_check
    context = {"question": "q", "document_context": [],
               "dynamic_context": {"get_inference_history": history, "series_analysis": analysis},
               "warnings": []}
    result = grounding_check.check_grounding("L'écart minimal est de 0.4242.", context)
    assert "0.4242" in result.ungrounded_numbers


# --- G. Q9: the PREDICTED sequence must be present and unconfusable ---------
#
# Q9 ("Y a-t-il un saut de phase ou une régression dans la séquence PREDITE ?")
# previously received one model window plus get_transition_events' OBSERVED
# transition -- whose only skip belongs to the annotation, not to the model.
# These tests pin the fix: both sequences reach the context, labelled, and the
# skip/regression counts are stated to be about the model's sequence.

def test_both_phase_sequences_are_exposed_window_by_window(analysis):
    sequences = analysis["sequences"]
    assert sequences["model_derived_phase_sequence"] == \
        {str(w): phase for w, phase, *_ in _SERIES}
    assert sequences["observed_annotation_phase_sequence"] == \
        {str(w): observed for w, _, _, _, observed in _SERIES}


def test_the_two_sequences_actually_differ_and_are_not_collapsed(analysis):
    sequences = analysis["sequences"]
    model = sequences["model_derived_phase_sequence"]
    observed = sequences["observed_annotation_phase_sequence"]
    assert model != observed, "the fixture must exercise a real model/annotation disagreement"
    assert model["156"] == "t6" and observed["156"] == "t4"


def test_the_sequence_note_forbids_answering_from_the_annotation(analysis):
    note = analysis["sequences"]["note"]
    assert "model_derived_phase_sequence ONLY" in note
    assert "get_transition_events" in note


def test_rendered_sequences_are_labelled_by_provenance(rendered):
    assert "MODEL-DERIVED  model_phase_sequence" in rendered
    assert "OBSERVED       observed_phase_sequence" in rendered


def test_skip_and_regression_counts_are_named_as_being_about_the_predicted_sequence(rendered):
    assert "n_phase_skips_in_predicted_sequence" in rendered
    assert "n_phase_regressions_in_predicted_sequence" in rendered
    assert "SEQUENCE PREDITE" in rendered


def test_stability_block_declares_it_says_nothing_about_the_annotation(analysis, rendered):
    provenance = analysis["stability"]["definitions"]["provenance_of_this_whole_block"]
    assert provenance.startswith("MODEL-DERIVED")
    assert "get_transition_events" in provenance
    assert "MODEL-DERIVED uniquement" in rendered


def test_sequences_survive_a_missing_annotation_without_borrowing_the_model_phase():
    unannotated = _history(series=[(155, "t4", {"t4": 0.9, "t6": 0.1}, {"t4": 0.9}, "t4"),
                                    (156, "t6", {"t6": 0.9, "t4": 0.1}, {"t6": 0.9}, None)])
    sequences = derive_series_analysis(unannotated)["sequences"]
    assert sequences["model_derived_phase_sequence"]["156"] == "t6"
    assert sequences["observed_annotation_phase_sequence"]["156"] is None

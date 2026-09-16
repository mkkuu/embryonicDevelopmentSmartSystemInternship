"""
Tests for the dynamic_context priority reordering
(docs/RAG_PROMPT_REORDERING_AB_TEST.md, Option A of
docs/RAG_CONTEXT_FAILURE_ANALYSIS.md sec 7): compact/important scalar
fields are rendered BEFORE bulky same-type distributions
(phase_probabilities, next_phase_distribution, duration_quantiles),
which are deferred to the end of the flattened prompt -- every value
still preserved verbatim, only its render POSITION changes. No LLM, no
torch, no chromadb.
"""

import json

from orchestrator import prompt
from orchestrator.context_builder import SOURCE_REPORTING_API, ToolCallResult, build_context
from orchestrator.router import classify

_PHASE_PROBS = {
    "t2": 0.001, "t3": 0.002, "t4": 0.004, "t5": 0.01, "t6": 0.5643, "t7": 0.05,
    "t8": 0.03, "t9+": 0.02, "tM": 0.001, "tSB": 0.001, "tB": 0.001, "tEB": 0.001,
    "tHB": 0.001, "tPB2": 0.001, "tPNf": 0.001,
}
_DURATION_QUANTILES = {"0.1": 5, "0.25": 8, "0.5": 12, "0.75": 18, "0.9": 25}  # 5 entries,
# same as the real Semi-HMM's own DURATION_QUANTILE_LEVELS, unlike the 3-entry fixture in
# test_prompt_dynamic_context_flattening.py -- deliberately real-shaped for this fix.

_INFERENCE_RECORD = {
    "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
               "patient_id": "Patient_319", "split": "val"},
    "window": {"window_start": 156, "window_start_time": 43.1, "window_end_time": 44.8,
               "window_mid_time": 43.95, "window_duration": 1.7, "time_available": True,
               "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": 0},
    "current_state": {
        "current_phase": "t7", "current_phase_index": 8, "phase_probability": 0.7609993694101455,
        "phase_probabilities": _PHASE_PROBS, "entropy": 0.5493,
    },
    "next_phase": {
        "next_phase_distribution": dict(_PHASE_PROBS),
        "most_likely_next_phase": "t7", "next_phase_probability": 0.76,
    },
    "duration": {
        "duration_available": True, "expected_duration": 12.3,
        "duration_quantiles": _DURATION_QUANTILES, "duration_unit": "windows",
    },
    "model": {"model_name": "semi_hmm", "model_version": "dmax=268,negative_binomial",
              "model_configuration": {"dmax": 268}, "model_source": "Results/evaluation/x",
              "inference_timestamp": "2026-08-31T00:00:00"},
    "ground_truth": {"ground_truth_phase": "t4", "consistency_flag": 1},
    "warnings": [],
}

_TRANSITION_EVENTS = {
    "video_id": "Patient_319", "split": "val", "window": 156, "n_transitions": 1,
    "transitions": [{
        "from_phase": "t4", "to_phase": "t6", "from_phase_index": 5, "to_phase_index": 7,
        "observed": True, "provenance": "observed_annotation", "window_start": 156, "window_end": 157,
        "frame_start": 174, "frame_end": 175, "frame_mapping_available": True,
        "window_start_time": 43.1, "window_end_time": 46.9, "time_available": True,
        "phase_distance": 2, "is_skip": True,
    }],
}


def _real_shaped_context():
    """Mirrors the real Patient_319/window=156 case from
    docs/RAG_CONTEXT_FAILURE_ANALYSIS.md -- both get_current_inference
    AND get_transition_events called, in that order (matches
    plan_tools()'s own real ordering)."""
    results = {
        "get_current_inference": ToolCallResult(
            tool_name="get_current_inference", source_type=SOURCE_REPORTING_API,
            ok=True, value=_INFERENCE_RECORD,
        ),
        "get_transition_events": ToolCallResult(
            tool_name="get_transition_events", source_type=SOURCE_REPORTING_API,
            ok=True, value=_TRANSITION_EVENTS,
        ),
    }
    return build_context("Quelle transition de phase est observée ici ?", classify("q"), results)


def _all_leaves(value, out):
    if isinstance(value, dict):
        for v in value.values():
            _all_leaves(v, out)
    elif isinstance(value, list):
        for v in value:
            _all_leaves(v, out)
    else:
        out.append(value)


# --- Unit tests for _is_homogeneous_scalar_distribution() directly ---

def test_a_15_entry_float_distribution_is_deferred():
    assert prompt._is_homogeneous_scalar_distribution(_PHASE_PROBS) is True


def test_a_5_entry_int_distribution_is_deferred():
    assert prompt._is_homogeneous_scalar_distribution(_DURATION_QUANTILES) is True


def test_a_4_entry_identity_dict_is_not_deferred():
    # SampleInfo's own 4 string fields -- same field count class as a
    # "distribution," but an identity/record block, not an enumeration.
    sample = {"sample_id": "x#val", "video_name": "x", "patient_id": "x", "split": "val"}
    assert prompt._is_homogeneous_scalar_distribution(sample) is False


def test_a_mixed_type_record_is_never_deferred_regardless_of_field_count():
    # One TransitionEvent's own dict -- 16 fields, str+int+bool+float mixed.
    transition = _TRANSITION_EVENTS["transitions"][0]
    assert len(transition) > 4  # more fields than _DISTRIBUTION_MIN_SIZE
    assert prompt._is_homogeneous_scalar_distribution(transition) is False


def test_a_small_homogeneous_dict_at_or_below_threshold_is_not_deferred():
    small = {"a": 1, "b": 2, "c": 3}  # 3 entries, all int -- at the threshold, not above it
    assert prompt._is_homogeneous_scalar_distribution(small) is False


# --- Test A: all information present before is still present after ---

def test_a_no_leaf_value_is_lost_across_both_tools():
    leaves = []
    _all_leaves(_INFERENCE_RECORD, leaves)
    _all_leaves(_TRANSITION_EVENTS, leaves)
    text = prompt.build_user_prompt(_real_shaped_context())
    for leaf in leaves:
        if leaf is None or leaf == [] or leaf == "":
            continue
        expected = ("true" if leaf else "false") if isinstance(leaf, bool) else str(leaf)
        assert expected in text, f"leaf value {leaf!r} missing after reordering"


# --- Test B: priority fields appear earlier ---

def test_b_transition_event_fields_now_appear_before_phase_probabilities():
    text = prompt.build_user_prompt(_real_shaped_context())
    idx_transition = text.find("get_transition_events.transitions[0].from_phase")
    idx_phase_probs = text.find("get_current_inference.current_state.phase_probabilities.t2")
    idx_next_phase_dist = text.find("get_current_inference.next_phase.next_phase_distribution.t2")
    assert idx_transition != -1 and idx_phase_probs != -1 and idx_next_phase_dist != -1
    assert idx_transition < idx_phase_probs
    assert idx_transition < idx_next_phase_dist


def test_b_current_phase_and_ground_truth_appear_before_the_big_distributions():
    text = prompt.build_user_prompt(_real_shaped_context())
    idx_current_phase = text.find("get_current_inference.current_state.current_phase =")
    idx_ground_truth = text.find("get_current_inference.ground_truth.ground_truth_phase =")
    idx_phase_probs = text.find("get_current_inference.current_state.phase_probabilities.t2")
    assert idx_current_phase < idx_phase_probs
    assert idx_ground_truth < idx_phase_probs


def test_b_real_diagnostic_position_improves_dramatically():
    """Reproduces the exact real measurement from
    docs/RAG_CONTEXT_FAILURE_ANALYSIS.md: get_transition_events' answer
    sat at ~82% of the prompt before this fix. After the fix, on an
    equivalent real-shaped context, it must sit well before the midpoint."""
    text = prompt.build_user_prompt(_real_shaped_context())
    idx = text.find("get_transition_events.transitions[0].from_phase")
    position_fraction = idx / len(text)
    assert position_fraction < 0.5


# --- Test C: no "observed" field is removed ---

def test_c_all_observed_transition_fields_survive():
    text = prompt.build_user_prompt(_real_shaped_context())
    for field, value in _TRANSITION_EVENTS["transitions"][0].items():
        expected = ("true" if value else "false") if isinstance(value, bool) else str(value)
        assert f"get_transition_events.transitions[0].{field} = {expected}" in text


# --- Test D: no "model-derived" field is removed ---

def test_d_all_15_phase_probabilities_survive():
    text = prompt.build_user_prompt(_real_shaped_context())
    for phase, prob in _PHASE_PROBS.items():
        assert f"get_current_inference.current_state.phase_probabilities.{phase} = {prob}" in text


def test_d_all_15_next_phase_distribution_entries_survive():
    text = prompt.build_user_prompt(_real_shaped_context())
    for phase, prob in _PHASE_PROBS.items():
        assert f"get_current_inference.next_phase.next_phase_distribution.{phase} = {prob}" in text


def test_d_all_5_duration_quantiles_survive():
    text = prompt.build_user_prompt(_real_shaped_context())
    for level, value in _DURATION_QUANTILES.items():
        assert f'get_current_inference.duration.duration_quantiles["{level}"] = {value}' in text


# --- Test E: nested structures remain correctly flattened ---

def test_e_list_of_dicts_still_flattened_per_index():
    chain = [{"phase": "t5", "duration": 10}, {"phase": "t6", "duration": 15}]
    results = {"get_trajectory": ToolCallResult(
        tool_name="get_trajectory", source_type=SOURCE_REPORTING_API, ok=True,
        value={"sample": {"video_name": "Patient_319"}, "n_windows": 2,
               "window_starts": [0, 8], "transition_chain": chain, "model": {"model_version": "v1"}},
    )}
    ctx = build_context("q", classify("q"), results)
    text = prompt.build_user_prompt(ctx)
    assert "get_trajectory.transition_chain[0].phase = t5" in text
    assert "get_trajectory.transition_chain[1].duration = 15" in text


def test_e_scalar_list_still_compact_not_exploded():
    text = prompt.build_user_prompt(_real_shaped_context())
    assert text.count("get_transition_events.transitions[") >= 1  # exploded (list of dicts)
    # window_starts-style scalar lists (not present in this fixture) are covered by the
    # pre-existing test_prompt_dynamic_context_flattening.py::test_scalar_list_is_compact...
    # -- unaffected by this fix (deferral only applies to dicts, never lists).


# --- Test F: numeric values remain identical ---

def test_f_float_values_are_byte_identical_to_python_str_repr():
    text = prompt.build_user_prompt(_real_shaped_context())
    assert "0.7609993694101455" in text  # phase_probability, full precision
    assert str(_PHASE_PROBS["t6"]) in text
    assert str(_PHASE_PROBS["t9+"]) in text


# --- Test G: the format remains deterministic ---

def test_g_reordered_output_is_deterministic():
    ctx = _real_shaped_context()
    assert prompt.build_user_prompt(ctx) == prompt.build_user_prompt(ctx)


def test_g_reordering_produces_the_exact_same_multiset_of_lines_as_before(monkeypatch):
    """Direct proof that this is a pure reorder: force _DISTRIBUTION_MIN_SIZE
    impossibly high (nothing ever deferred) to reconstruct the OLD,
    pre-fix single-pass output, and compare its line SET against the
    real, reordered output -- must be identical, only order differs."""
    ctx = _real_shaped_context()
    new_output = prompt.build_user_prompt(ctx)

    monkeypatch.setattr(prompt, "_DISTRIBUTION_MIN_SIZE", 10_000)
    old_shaped_output = prompt.build_user_prompt(ctx)

    new_lines = sorted(new_output.splitlines())
    old_lines = sorted(old_shaped_output.splitlines())
    assert new_lines == old_lines

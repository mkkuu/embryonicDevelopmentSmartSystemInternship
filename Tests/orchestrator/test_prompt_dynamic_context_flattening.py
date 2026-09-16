"""Tests for prompt.py's dynamic_context flattening (docs/PROJECT_CHECKPOINT.md
Phase 6 Priority 1: a small local LLM failed to locate one specific
nested value -- e.g. `t6`'s own probability -- inside a raw `json.dumps`
of a 15-entry `phase_probabilities` dict). No LLM, no torch, no chromadb."""

import json

from orchestrator import grounding_check, prompt
from orchestrator.context_builder import SOURCE_REPORTING_API, ToolCallResult, build_context
from orchestrator.router import classify

# A realistic get_current_inference payload, shaped exactly like
# reporting/schemas.py's InferenceRecord.to_dict() -- 15-way phase
# distributions, a duration-quantiles dict, nested provenance.
_PHASE_PROBS = {
    "t2": 0.001, "t3": 0.002, "t4": 0.004, "t5": 0.01, "t6": 0.5643, "t7": 0.05,
    "t8": 0.03, "t9+": 0.02, "tM": 0.001, "tSB": 0.001, "tB": 0.001, "tEB": 0.001,
    "tHB": 0.001, "tPB2": 0.001, "tPNf": 0.001,
}
_INFERENCE_RECORD = {
    "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
               "patient_id": "Patient_319", "split": "val"},
    "window": {"window_start": 42, "window_start_time": None, "window_end_time": None,
               "window_mid_time": None, "window_duration": None, "time_available": False,
               "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": None},
    "current_state": {
        "current_phase": "t6", "current_phase_index": 4, "phase_probability": 0.5643,
        "phase_probabilities": _PHASE_PROBS, "entropy": 1.23,
    },
    "next_phase": {
        "next_phase_distribution": dict(_PHASE_PROBS),
        "most_likely_next_phase": "t6", "next_phase_probability": 0.5643,
    },
    "duration": {
        "duration_available": True, "expected_duration": 12.3,
        "duration_quantiles": {"0.1": 5, "0.5": 12, "0.9": 20}, "duration_unit": "windows",
    },
    "model": {"model_name": "semi_hmm", "model_version": "dmax=268,negative_binomial,alpha=1.0,rho=0.3,class_weight=None",
              "model_configuration": {"dmax": 268, "n_states": 15}, "model_source": "Results/evaluation/x",
              "inference_timestamp": "2026-08-27T00:00:00"},
    "ground_truth": {"ground_truth_phase": None, "consistency_flag": None},
    "warnings": [],
}


def _build_dynamic_context(tool_name="get_current_inference", value=None):
    value = _INFERENCE_RECORD if value is None else value
    results = {
        tool_name: ToolCallResult(tool_name=tool_name, source_type=SOURCE_REPORTING_API, ok=True, value=value),
    }
    return build_context("Quelle est la probabilité actuelle de t6 ?", classify("q"), results)


def _all_leaves(value, out):
    if isinstance(value, dict):
        for v in value.values():
            _all_leaves(v, out)
    elif isinstance(value, list):
        for v in value:
            _all_leaves(v, out)
    else:
        out.append(value)


def test_deeply_nested_scalar_gets_its_own_scannable_line():
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert "get_current_inference.current_state.phase_probabilities.t6 = 0.5643" in text


def test_next_phase_distribution_t6_is_distinguishable_from_current_state_t6():
    # The real failure mode risked confusing which "t6" number was meant --
    # the flattened path must disambiguate current_state vs. next_phase.
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert "current_state.phase_probabilities.t6 = 0.5643" in text
    assert "next_phase.next_phase_distribution.t6 = 0.5643" in text


def test_no_leaf_value_is_lost_by_flattening():
    leaves = []
    _all_leaves(_INFERENCE_RECORD, leaves)
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    for leaf in leaves:
        if leaf is None or leaf == [] or leaf == "":
            continue
        if isinstance(leaf, bool):
            expected = "true" if leaf else "false"  # _format_scalar's JSON-style rendering
        else:
            expected = str(leaf)
        assert expected in text, f"leaf value {leaf!r} missing from flattened prompt text"


def test_flattening_is_deterministic():
    ctx = _build_dynamic_context()
    assert prompt.build_user_prompt(ctx) == prompt.build_user_prompt(ctx)


def test_scalar_list_is_compact_not_exploded_per_element():
    window_starts = list(range(0, 5270, 10))  # 527 entries, mimics a real trajectory
    ctx = _build_dynamic_context(tool_name="get_trajectory", value={
        "sample": {"video_name": "Patient_319"}, "n_windows": len(window_starts),
        "window_starts": window_starts, "transition_chain": [], "model": {"model_version": "v1"},
    })
    text = prompt.build_user_prompt(ctx)
    assert text.count("window_starts") == 1  # one line, not 527
    assert f"({len(window_starts)} item(s))" in text
    assert str(window_starts[0]) in text and str(window_starts[-1]) in text


def test_list_of_dicts_is_still_flattened_per_index():
    chain = [{"phase": "t5", "duration": 10}, {"phase": "t6", "duration": 15}]
    ctx = _build_dynamic_context(tool_name="get_trajectory", value={
        "sample": {"video_name": "Patient_319"}, "n_windows": 2,
        "window_starts": [0, 8], "transition_chain": chain, "model": {"model_version": "v1"},
    })
    text = prompt.build_user_prompt(ctx)
    assert "get_trajectory.transition_chain[0].phase = t5" in text
    assert "get_trajectory.transition_chain[1].duration = 15" in text


def test_empty_dynamic_context_message_unchanged():
    ctx = build_context("q", classify("q"), {})
    text = prompt.build_user_prompt(ctx)
    assert "aucune donnée dynamique récupérée" in text


def test_bool_and_null_leaves_render_json_style():
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert "get_current_inference.duration.duration_available = true" in text
    assert "get_current_inference.ground_truth.ground_truth_phase = null" in text


def test_flattened_value_restated_verbatim_is_grounded():
    # The whole point of this transformation: a value now easy to find in
    # the flattened prompt text must ALSO already be grounded by the
    # independent, prompt-text-agnostic grounding_check (it only ever reads
    # context["dynamic_context"] directly, never the formatted prompt) --
    # proves the fix helps a real provider without weakening the check.
    ctx = _build_dynamic_context()
    answer_text = "La probabilité actuelle de t6 est 0.5643."
    result = grounding_check.check_grounding(answer_text, ctx)
    assert result.grounded


def test_flattening_does_not_change_raw_dynamic_context_seen_by_grounding_check():
    # grounding_check must keep operating on the untouched raw dict --
    # prompt.py's formatting is presentation-only.
    ctx = _build_dynamic_context()
    assert ctx["dynamic_context"]["get_current_inference"] == _INFERENCE_RECORD


def test_grounding_check_result_is_unaffected_by_which_formatter_is_active(monkeypatch):
    # Stronger decoupling proof than the identity check above: even if
    # prompt.py's formatter were swapped for something else entirely,
    # grounding_check.check_grounding() must produce the exact same
    # verdict, because it never calls into prompt.py at all -- it reads
    # context["dynamic_context"] directly, every time.
    ctx = _build_dynamic_context()
    answer_text = "La probabilité actuelle de t6 est 0.5643."
    baseline = grounding_check.check_grounding(answer_text, ctx)

    monkeypatch.setattr(prompt, "_format_dynamic_context", lambda dynamic_context: "BOGUS TEXT, UNRELATED")
    after_swap = grounding_check.check_grounding(answer_text, ctx)

    assert after_swap.grounded == baseline.grounded
    assert after_swap.checked_numbers == baseline.checked_numbers
    assert after_swap.ungrounded_numbers == baseline.ungrounded_numbers


def test_dict_keys_containing_dots_use_bracket_notation_to_avoid_path_ambiguity():
    # duration_quantiles' own keys ("0.1", "0.5", "0.9") contain "." --
    # the same character used as the path separator. Joined naively
    # ("path.0.1") this would be indistinguishable from two extra nesting
    # levels ("0" -> "1"). The flattener must bracket such a key instead,
    # keeping the path a single, unambiguous pointer into the original dict.
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert 'get_current_inference.duration.duration_quantiles["0.1"] = 5' in text
    assert 'get_current_inference.duration.duration_quantiles["0.5"] = 12' in text
    assert 'get_current_inference.duration.duration_quantiles["0.9"] = 20' in text
    # And the ambiguous, un-bracketed form must NOT appear.
    assert "duration_quantiles.0.1 = 5" not in text


def test_phase_key_with_plus_sign_renders_on_its_own_unambiguous_line():
    # "t9+" is a real phase name (docs/RAG_PHASE_3_REPORT.md's own
    # convention) -- "+" is not the path separator, so no bracketing is
    # needed, but the line must still be a single, clean, greppable entry.
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert "get_current_inference.current_state.phase_probabilities.t9+ = 0.02" in text


def test_string_leaf_values_render_literally_without_json_quoting():
    # A string leaf (e.g. current_phase="t6") must render as bare text,
    # not as a JSON-quoted '"t6"' -- json.dumps was exactly the
    # representation being replaced, so no leaf should still go through it.
    ctx = _build_dynamic_context()
    text = prompt.build_user_prompt(ctx)
    assert "get_current_inference.current_state.current_phase = t6" in text
    assert '"t6"' not in text
    assert "get_current_inference.sample.video_name = Patient_319" in text


def test_high_precision_float_values_are_preserved_exactly():
    # The transformation must never round/reformat a number -- grounding_check's
    # own KNOWN set is built from str(value) on the SAME raw floats, so any
    # divergence here (rounding, scientific notation, trailing-zero
    # stripping) would silently break groundedness for a provider that
    # restates a value verbatim from this text.
    value = {
        "sample": {"video_name": "Patient_319"},
        "current_state": {"phase_probability": 0.6641144553128034},
        "next_phase": {"next_phase_probability": 0.6636264445421354},
        "duration": {"expected_duration": 9.8303440728163},
    }
    ctx = _build_dynamic_context(value=value)
    text = prompt.build_user_prompt(ctx)
    assert "current_state.phase_probability = 0.6641144553128034" in text
    assert "next_phase.next_phase_probability = 0.6636264445421354" in text
    assert "duration.expected_duration = 9.8303440728163" in text
    # And the KNOWN set grounding_check builds from the same raw dict must
    # contain the identical string -- proving prompt text and grounding
    # truth can never silently diverge in how a float is stringified.
    known = grounding_check._known_numbers_from_dynamic_context(ctx["dynamic_context"])
    assert "0.6641144553128034" in known
    assert "0.6636264445421354" in known
    assert "9.8303440728163" in known

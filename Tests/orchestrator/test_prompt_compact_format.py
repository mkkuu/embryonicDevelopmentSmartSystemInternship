"""COMPACT_CONTEXT_V1 rendering -- safety + provenance tests.

Pure: no LLM, no torch, no chromadb, no GPU, no network. Every context here is
built by the real `context_builder.build_context()` from tool payloads shaped
like the real Reporting API's (`Training/reporting/schemas.py`), with the same
anchor the frozen benchmark uses (Patient_319 / val / window 156).

What these pin, in the order the mission states them:
  1-3. OBSERVED / MODEL-DERIVED / DERIVED blocks survive the compact rendering;
  4.   provenance stays explicit;
  5.   the temporal quantities are never mixed or renamed;
  6.   every RAG chunk stays traceable (source/section/status printed);
  7.   no expected answer is injected by the compaction;
  8.   Q3's MODEL-DERIVED information is preserved, and the annotation is NOT
       substituted for it;
  9.   the 15 frozen questions are untouched by this change.
"""

import math
import re

import pytest

from orchestrator import compact_context, prompt
from orchestrator.context_builder import (
    SOURCE_DOCUMENT,
    SOURCE_REPORTING_API,
    ToolCallResult,
    build_context,
)
from orchestrator.router import classify
from orchestrator.temporal_context import derive_series_analysis

CURRENT_INFERENCE = {
    "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
               "patient_id": "Patient_319", "split": "val"},
    "window": {"window_start": 156, "window_start_time": 43.1, "window_end_time": 44.8,
               "window_mid_time": 43.95, "window_duration": 1.7, "time_available": True,
               "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": 0},
    "current_state": {"current_phase": "t7", "current_phase_index": 8,
                      "phase_probability": 0.7609993694101455,
                      "phase_probabilities": {"t5": 0.0017052671933268549, "t6": 1.1459e-05,
                                              "t7": 0.7609993694101455, "t8": 0.23679880956344662,
                                              "t9+": 0.00043476617122871367},
                      "entropy": 0.5638288292962345},
    "next_phase": {"next_phase_distribution": {"t7": 0.7313823792655003, "t8": 0.2612672088018441},
                   "most_likely_next_phase": "t7", "next_phase_probability": 0.7313823792655003},
    "duration": {"duration_available": True, "expected_duration": 22.747089383925076,
                 "duration_quantiles": {"0.1": 1, "0.25": 3, "0.5": 12, "0.75": 30, "0.9": 59},
                 "duration_unit": "windows"},
    "model": {"model_name": "semi_hmm", "model_version": "", "model_source": "",
              "inference_timestamp": "2026-09-11T10:00:00+00:00"},
    "ground_truth": {"ground_truth_phase": "t4", "consistency_flag": 0},
    "warnings": [],
}

TRANSITION_EVENTS = {
    "video_id": "Patient_319", "split": "val", "window": 156, "n_transitions": 1,
    "transitions": [{"from_phase": "t4", "to_phase": "t6", "window_start": 157,
                     "transition_time": 44.8, "is_skip": True, "observed": True,
                     "provenance": "observed_annotation"}],
}

# The multi-window series and its deterministic analysis are built by the REAL
# production functions (`temporal_context.derive_series_analysis`) over a
# synthetic `get_inference_history` payload shaped like the Reporting API's --
# never a hand-written analysis dict, which could drift from the renderer.
CENTER = 156
_PHASE_INDEX = {"t7": 8, "t8": 9}
_SERIES = [
    (155, "t7", {"t7": 0.66, "t8": 0.33}, {"t7": 0.64, "t8": 0.35}, "t4"),
    (156, "t7", {"t7": 0.76, "t8": 0.23}, {"t7": 0.73, "t8": 0.26}, "t4"),
    (157, "t8", {"t8": 0.92, "t7": 0.07}, {"t8": 0.92, "t7": 0.07}, "t6"),
]


def _entry(window, model_phase, probabilities, next_distribution, observed_phase):
    return {
        "window_start": window,
        "offset_from_center": window - CENTER,
        "is_center": window == CENTER,
        "window_start_time": 40.0 + window * 0.02,
        "window_end_time": 40.0 + (window + 1) * 0.02,
        "time_available": True,
        "model_derived": {
            "current_phase": model_phase,
            "current_phase_index": _PHASE_INDEX[model_phase],
            "phase_probability": max(probabilities.values()),
            "phase_probabilities": dict(probabilities),
            "entropy": round(-sum(p * math.log(p) for p in probabilities.values()), 6),
            "most_likely_next_phase": max(next_distribution, key=next_distribution.get),
            "next_phase_probability": max(next_distribution.values()),
            "next_phase_distribution": dict(next_distribution),
        },
        "observed": {"ground_truth_phase": observed_phase, "consistency_flag": 0},
    }


INFERENCE_HISTORY = {
    "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319",
               "patient_id": "Patient_319", "split": "val"},
    "center_window": CENTER, "requested_before": 1, "requested_after": 1,
    "n_windows_before": 1, "n_windows_after": 1, "n_windows": len(_SERIES),
    "window_starts": [row[0] for row in _SERIES],
    "entries": [_entry(*row) for row in _SERIES],
    "model": {"model_name": "semi_hmm", "model_version": "", "model_source": "",
              "inference_timestamp": "2026-09-11T10:00:00+00:00"},
    "warnings": [],
}

SERIES_ANALYSIS = derive_series_analysis(INFERENCE_HISTORY)

NARRATIVE_DOC = {
    "content": ("**If the project fully succeeds** (a 2-year-horizon scenario):\n"
                "- *Title*: \"A single hidden coordinate governs embryo development.\"\n"),
    "metadata": {"source": "docs/RESEARCH_BLUEPRINT.md", "section": "Part VII — Publication Strategy",
                 "status": "authoritative", "authoritative": True},
    "score": 0.55,
}

FACTUAL_DOC = {
    "content": ("The 15 phases, in chronological order: tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, "
                "t7, t8, t9+, tM, tSB, tB, tEB.\n"),
    "metadata": {"source": "docs/SCIENTIFIC_REPORT.md", "section": "2. Dataset",
                 "status": "authoritative", "authoritative": True},
    "score": 0.61,
}


def _full_context(question="Cette transition est-elle compatible avec l'ordre attendu ?"):
    results = {
        "get_current_inference": ToolCallResult("get_current_inference", SOURCE_REPORTING_API,
                                                True, value=CURRENT_INFERENCE),
        "get_transition_events": ToolCallResult("get_transition_events", SOURCE_REPORTING_API,
                                                True, value=TRANSITION_EVENTS),
        "get_inference_history": ToolCallResult("get_inference_history", SOURCE_REPORTING_API,
                                               True, value=INFERENCE_HISTORY),
        "series_analysis": ToolCallResult("series_analysis", SOURCE_REPORTING_API,
                                          True, value=SERIES_ANALYSIS),
        "retrieve_documents": ToolCallResult("retrieve_documents", SOURCE_DOCUMENT, True,
                                             value=[NARRATIVE_DOC, FACTUAL_DOC]),
    }
    return build_context(question, classify(question), results,
                         tools_called=list(results))


# ---------------------------------------------------------------------------
# format selection
# ---------------------------------------------------------------------------

def test_default_format_is_the_historical_one_and_is_unchanged():
    ctx = _full_context()
    assert prompt.resolve_context_format() == prompt.CONTEXT_FORMAT_V2
    assert prompt.build_user_prompt(ctx) == prompt.build_user_prompt(ctx, "v2")
    text = prompt.build_user_prompt(ctx)
    assert "CONTEXTE DOCUMENT (RAG" in text
    assert "[DOC 1]" in text
    assert "[RAG DOCUMENT 1]" not in text
    assert "PROVENANCE DES BLOCS" not in text


def test_unknown_format_fails_loudly():
    with pytest.raises(ValueError):
        prompt.resolve_context_format("shorter")


def test_env_var_selects_the_format(monkeypatch):
    monkeypatch.setenv(prompt.CONTEXT_FORMAT_ENV_VAR, "compact")
    assert prompt.resolve_context_format() == prompt.CONTEXT_FORMAT_COMPACT
    # an explicit argument still wins over the environment
    assert prompt.resolve_context_format("v2") == prompt.CONTEXT_FORMAT_V2


def test_compact_rendering_is_deterministic():
    ctx = _full_context()
    assert prompt.build_user_prompt(ctx, "compact") == prompt.build_user_prompt(ctx, "compact")


# ---------------------------------------------------------------------------
# 1-3. OBSERVED / MODEL-DERIVED / DERIVED all survive
# ---------------------------------------------------------------------------

def test_observed_block_survives_the_compact_rendering():
    text = prompt.build_user_prompt(_full_context(), "compact")
    assert "get_current_inference.ground_truth.ground_truth_phase = t4" in text
    assert "get_transition_events.transitions[0].from_phase = t4" in text
    assert "get_transition_events.transitions[0].to_phase = t6" in text
    assert "provenance = observed_annotation" in text


def test_model_derived_block_survives_the_compact_rendering():
    text = prompt.build_user_prompt(_full_context(), "compact")
    assert "get_current_inference.current_state.current_phase = t7" in text
    assert "get_current_inference.current_state.phase_probability = 0.7609993694101455" in text
    assert "get_current_inference.next_phase.most_likely_next_phase = t7" in text
    assert "get_current_inference.duration.expected_duration = 22.747089383925076" in text


def test_derived_block_survives_the_compact_rendering():
    text = prompt.build_user_prompt(_full_context(), "compact")
    assert "ANALYSE DETERMINISTE DE LA SERIE" in text
    assert "derived_deterministic" in text
    for derived_field in ("probability_gap_minimum", "convergence_start_window",
                          "n_phase_returns", "n_phase_skips_in_predicted_sequence",
                          "entropy_near_minus_far"):
        assert derived_field in text
    # a DERIVED value itself, exactly as the renderer prints it
    gap_minimum = SERIES_ANALYSIS["probability_gap"]["minimum"]
    assert f"probability_gap_minimum = {gap_minimum}" in text
    # a DERIVED quantity that is not computable on this band stays null, never
    # estimated (the band has no window at |offset| >= 3)
    assert "entropy_near_minus_far = null" in text


def test_dynamic_half_is_byte_identical_between_the_two_formats():
    """The single variable is the DOCUMENT half: not one dynamic value may be
    moved, rounded, reordered or dropped by the compact renderer."""
    ctx = _full_context()
    marker = "CONTEXTE DYNAMIQUE"
    v2_dynamic = prompt.build_user_prompt(ctx, "v2").split(marker, 1)[1]
    compact_dynamic = prompt.build_user_prompt(ctx, "compact").split(marker, 1)[1]
    assert v2_dynamic == compact_dynamic


# ---------------------------------------------------------------------------
# 4. provenance stays explicit
# ---------------------------------------------------------------------------

def test_provenance_legend_names_the_classes_without_stating_any_value():
    text = prompt.build_user_prompt(_full_context(), "compact")
    legend = text.split("PROVENANCE DES BLOCS", 1)[1].split("CONTEXTE DYNAMIQUE", 1)[0]
    for klass in ("OBSERVED", "MODEL-DERIVED", "DERIVED", "DOCUMENT"):
        assert klass in legend
    # no value, no phase name, no window id, no probability in the legend
    assert not re.search(r"\bt\d+\+?\b|\btPB2\b|\btEB\b|\bPatient_\d+\b", legend)
    assert not re.search(r"\d\.\d{3,}", legend)


def test_legend_never_moves_a_field_from_one_class_to_another():
    text = prompt.build_user_prompt(_full_context(), "compact")
    legend = text.split("PROVENANCE DES BLOCS", 1)[1].split("CONTEXTE DYNAMIQUE", 1)[0]
    observed_line = [l for l in legend.splitlines() if l.startswith("- OBSERVED")]
    model_lines = [l for l in legend.splitlines() if l.startswith("- MODEL-DERIVED")]
    assert any("ground_truth" in l for l in observed_line)
    assert any("get_transition_events" in l for l in observed_line)
    assert any("current_state" in l for l in model_lines)
    # the annotation is never announced as a model output, nor the reverse
    assert not any("ground_truth" in l for l in model_lines)
    assert not any("current_state" in l for l in observed_line)


def test_document_block_prints_source_section_status_for_every_chunk():
    text = prompt.build_user_prompt(_full_context(), "compact")
    block = text.split("CONTEXTE SCIENTIFIQUE", 1)[1].split("PROVENANCE DES BLOCS", 1)[0]
    assert block.count("[RAG DOCUMENT ") == 2
    assert "source: docs/SCIENTIFIC_REPORT.md" in block
    assert "section: 2. Dataset" in block
    assert "status: authoritative" in block
    # the narrative chunk is dropped but still named, with its rule
    assert "source: docs/RESEARCH_BLUEPRINT.md" in block
    assert "regle S1" in block


# ---------------------------------------------------------------------------
# 5. temporal safety
# ---------------------------------------------------------------------------

def test_temporal_quantities_are_all_present_and_not_renamed():
    text = prompt.build_user_prompt(_full_context(), "compact")
    assert "window_start = 156" in text
    assert "window_start_time = 43.1" in text
    assert "window_end_time = 44.8" in text
    assert "offset_from_center" in text
    lag = SERIES_ANALYSIS["timing"]["model_vs_observed_lag_windows"]
    assert f"model_vs_observed_lag_windows = {'null' if lag is None else lag}" in text
    # an offset is never renamed into a lag: both names are present, and each
    # window block carries its own offset, in the renderer's own format
    for offset in (-1, 0, 1):
        assert f"offset_from_center={offset}" in text
    assert "WINDOW 156  [FENETRE CENTRALE / CENTER]" in text


def test_time_unit_is_never_invented():
    text = prompt.build_user_prompt(_full_context(), "compact")
    assert "time_unit = unknown/unverified" in text
    assert "time_unit = hours" not in text


# ---------------------------------------------------------------------------
# 7-8. no answer injection, Q3 provenance
# ---------------------------------------------------------------------------

def test_compaction_injects_no_new_content_into_the_document_block():
    """Every non-marker line of the compact document block is a verbatim line
    of some retrieved chunk."""
    ctx = _full_context()
    text = prompt.build_user_prompt(ctx, "compact")
    block = text.split("[RAG DOCUMENT 1]", 1)[1].split("PROVENANCE DES BLOCS", 1)[0]
    source_lines = set()
    for chunk in ctx["document_context"]:
        source_lines.update((chunk["content"] or "").splitlines())
    structural = ("[RAG DOCUMENT", "source:", "section:", "status:", "content:", "(note:",
                  "(document ecarte", "CONTEXTE SCIENTIFIQUE")
    for line in block.splitlines():
        if not line.strip() or line.startswith(structural):
            continue
        assert line in source_lines, f"line not traceable to a retrieved chunk: {line!r}"


def test_q3_keeps_its_model_derived_phase_and_the_annotation_is_not_substituted():
    """Q3 is the project's provenance test case: the context says the model
    believes t7 while the annotation says t4. The compact rendering must keep
    BOTH, each under its own name -- never overwrite one with the other."""
    question = "Quelle est la phase actuelle predite par le modele ?"
    text = prompt.build_user_prompt(_full_context(question), "compact")
    assert "current_state.current_phase = t7" in text      # MODEL-DERIVED, preserved
    assert "ground_truth.ground_truth_phase = t4" in text  # OBSERVED, still separate
    assert "current_state.current_phase = t4" not in text  # never corrected/substituted


def test_no_success_criterion_or_expected_answer_reaches_the_prompt():
    from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS

    text = prompt.build_user_prompt(_full_context(), "compact")
    for q in FIXED_QUESTIONS:
        for field in (q.success_criterion, q.expected_behavior):
            if field:
                assert field not in text


# ---------------------------------------------------------------------------
# 9. the frozen questions are untouched
# ---------------------------------------------------------------------------

def test_the_fifteen_questions_are_unchanged_by_this_experiment():
    """Hash of the 15 frozen question strings."""
    import hashlib
    import json

    from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS

    assert len(FIXED_QUESTIONS) == 15
    payload = json.dumps([q.question for q in FIXED_QUESTIONS],
                         ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Pinned against the question strings recorded in the real artefacts of
    # 2026-09-07 (fixed_question_benchmark_*.json) and 2026-09-11
    # (analysis_three_conditions_v1.json), verified equal string by string.
    assert digest.startswith("9bd4ed03ac5bf5cb"), (
        "the 15 frozen questions changed -- this experiment must never touch them")


def test_compact_rules_are_not_question_specific():
    """No rule may name a question, an anchor, a phase or a value: the filter
    must be a generic narrative filter, not a per-question special case."""
    tables = (compact_context.SECTION_RULES + compact_context.LINE_RULES
              + compact_context.KEEP_RULES)
    for _rule_id, markers, _reason in tables:
        for marker in markers:
            assert not re.fullmatch(r"t\d+\+?", marker)
            assert "patient_" not in marker
            assert "q1" not in marker and "q15" not in marker

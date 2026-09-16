"""COMPACT_CONTEXT_V2 rendering -- provenance, traceability and safety pins.

Pure: no LLM, no torch, no chromadb, no GPU, no network. Contexts are built
by the real `context_builder.build_context()` from the same synthetic
Reporting-API-shaped payloads as the compact_v1 tests (anchor Patient_319 /
val / window 156: model t7 -> t8, annotation t4 -> t6).
"""

import copy
import json
import re

import pytest

from orchestrator import compact_v2_render as v2r
from orchestrator import prompt
from orchestrator.compact_context import select_relevant_chunks
from orchestrator.context_builder import (
    SOURCE_AUDIT,
    SOURCE_DOCUMENT,
    SOURCE_REPORTING_API,
    SOURCE_SCIENTIFIC,
    ToolCallResult,
    build_context,
)
from orchestrator.router import classify
from orchestrator.scientific_reference import retrieve_scientific_reference
from test_prompt_compact_format import (
    CURRENT_INFERENCE,
    FACTUAL_DOC,
    INFERENCE_HISTORY,
    NARRATIVE_DOC,
    SERIES_ANALYSIS,
    TRANSITION_EVENTS,
)

Q3 = "Quelle était la phase dominante juste avant la transition ?"
Q13 = "Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques disponibles ?"

# The real 12-transition full history of Patient_319 (windows only), as stored
# in the compact_v1 artefact -- the shape that made Q1 fail 6/6.
FULL_HISTORY_WINDOWS = [(8, "tPB2", "tPNa"), (62, "tPNa", "tPNf"), (73, "tPNf", "t2"),
                        (115, "t2", "t3"), (119, "t3", "t4"), (156, "t4", "t6"),
                        (176, "t6", "t7"), (180, "t7", "t8"), (246, "t8", "t9+"),
                        (386, "t9+", "tM"), (438, "tM", "tSB"), (493, "tSB", "tB")]


def _full_history_events():
    return {
        "video_id": "Patient_319", "split": "val", "window": None,
        "n_transitions": len(FULL_HISTORY_WINDOWS),
        "transitions": [
            {"from_phase": a, "to_phase": b, "window_start": w, "window_end": w + 1,
             "window_start_time": float(w) / 4, "window_end_time": float(w) / 4 + 2.0,
             "observed": True, "provenance": "observed_annotation",
             "phase_distance": 2 if (a, b) == ("t4", "t6") else 1,
             "is_skip": (a, b) == ("t4", "t6")}
            for w, a, b in FULL_HISTORY_WINDOWS],
    }


def _results(question, *, history=False, docs=False, scientific=False, transition_context=False,
             full_history=False, trajectory=False):
    results = {
        "get_current_inference": ToolCallResult("get_current_inference", SOURCE_REPORTING_API,
                                                True, value=copy.deepcopy(CURRENT_INFERENCE)),
        "get_transition_events": ToolCallResult(
            "get_transition_events", SOURCE_REPORTING_API, True,
            value=_full_history_events() if full_history else copy.deepcopy(TRANSITION_EVENTS)),
    }
    if history:
        results["get_inference_history"] = ToolCallResult(
            "get_inference_history", SOURCE_REPORTING_API, True, value=copy.deepcopy(INFERENCE_HISTORY))
        results["series_analysis"] = ToolCallResult(
            "series_analysis", SOURCE_REPORTING_API, True, value=copy.deepcopy(SERIES_ANALYSIS))
    if trajectory:
        results["get_trajectory"] = ToolCallResult("get_trajectory", SOURCE_REPORTING_API, True, value={
            "sample": {"sample_id": "Patient_319#val", "video_name": "Patient_319"},
            "n_windows": 527, "window_starts": list(range(527)),
            "transition_chain": [{"phase": "t4", "observed_duration_windows": 37,
                                  "duration_probability_at_observed": 0.012,
                                  "next_phase_distribution": {"t5": 0.7, "t6": 0.2, "t7": 0.05,
                                                              "t8": 0.03, "t9+": 0.02}}],
            "model": {"model_version": "v"},
        })
    if transition_context:
        payload = {k: r.value for k, r in results.items()}
        results["transition_context"] = ToolCallResult(
            "transition_context", SOURCE_REPORTING_API, True,
            value=v2r.derive_transition_context(payload))
    if docs:
        kept, audit = select_relevant_chunks([NARRATIVE_DOC, FACTUAL_DOC], top_k=5)
        results["retrieve_documents"] = ToolCallResult("retrieve_documents", SOURCE_DOCUMENT, True,
                                                       value=kept)
        results["retrieval_audit"] = ToolCallResult("retrieval_audit", SOURCE_AUDIT, True, value=audit)
    if scientific:
        results["scientific_reference"] = ToolCallResult(
            "scientific_reference", SOURCE_SCIENTIFIC, True,
            value=retrieve_scientific_reference(question))
    return results


def _context(question=Q3, **kw):
    results = _results(question, **kw)
    return build_context(question, classify(question), results, tools_called=list(results))


def _render(question=Q3, **kw):
    return prompt.build_user_prompt(_context(question, **kw), v2r.CONTEXT_FORMAT_COMPACT_V2)


_LABELS = (v2r.LABEL_REFERENCE, v2r.LABEL_OBSERVED, v2r.LABEL_MODEL, v2r.LABEL_SERIES,
           v2r.LABEL_DERIVED, v2r.LABEL_IDENTIFIERS, v2r.LABEL_SCIENTIFIC, v2r.LABEL_PROJECT_DOCS,
           v2r.LABEL_LIMITATIONS, v2r.LABEL_WARNINGS)


def _label_above(text: str, line: str) -> str:
    """The local label governing `line` -- the last label printed before it."""
    assert line in text, line
    before = text.split(line, 1)[0]
    positions = [(before.rfind(label), label) for label in _LABELS if label in before]
    assert positions, f"no label above {line!r}"
    return max(positions)[1]


def _value_lines(text: str):
    return {l for l in text.splitlines() if re.match(r"^[A-Za-z_][\w.\[\]\"]* = ", l)}


# ---------------------------------------------------------------------------
# registration and isolation of the historical formats
# ---------------------------------------------------------------------------

def test_compact_v2_is_a_registered_format_and_the_others_are_untouched():
    assert prompt.resolve_context_format("compact_v2") == v2r.CONTEXT_FORMAT_COMPACT_V2
    ctx = _context(docs=True)
    v2 = prompt.build_user_prompt(ctx, "v2")
    compact = prompt.build_user_prompt(ctx, "compact")
    assert v2r.LABEL_OBSERVED not in v2 and v2r.LABEL_OBSERVED not in compact
    assert "PROVENANCE DES BLOCS" in compact and "PROVENANCE DES BLOCS" not in v2


def test_rendering_is_deterministic():
    ctx = _context(history=True, docs=True, scientific=True)
    a = prompt.build_user_prompt(ctx, "compact_v2")
    assert a == prompt.build_user_prompt(ctx, "compact_v2")


# ---------------------------------------------------------------------------
# local labels, no global legend  (the Q3 regression of compact_v1)
# ---------------------------------------------------------------------------

def test_local_labels_replace_the_global_legend():
    text = _render(history=True)
    assert "PROVENANCE DES BLOCS" not in text
    for label in (v2r.LABEL_OBSERVED, v2r.LABEL_MODEL, v2r.LABEL_DERIVED, v2r.LABEL_LIMITATIONS):
        assert label in text


def test_q3_model_phase_sits_under_the_model_label_and_annotation_under_observed():
    """The Q3 case: `current_phase = t7` must be readable ONLY as a prediction,
    `ground_truth_phase = t4` ONLY as the annotation, and neither is
    substituted for the other."""
    text = _render()
    assert _label_above(text, "get_current_inference.current_state.current_phase = t7") == v2r.LABEL_MODEL
    assert _label_above(text, "get_current_inference.ground_truth.ground_truth_phase = t4") == v2r.LABEL_OBSERVED
    assert _label_above(text, "get_transition_events.transitions[0].from_phase = t4") == v2r.LABEL_OBSERVED
    assert "current_state.current_phase = t4" not in text
    assert "ground_truth_phase = t7" not in text


def test_observed_block_comes_before_model_block_and_model_label_says_not_an_observation():
    text = _render()
    assert text.index(v2r.LABEL_OBSERVED) < text.index(v2r.LABEL_MODEL)
    assert "jamais une observation biologique" in v2r.LABEL_MODEL
    assert "jamais une prediction" in v2r.LABEL_OBSERVED


def test_every_model_value_is_under_the_model_label():
    text = _render()
    for line in ("get_current_inference.next_phase.most_likely_next_phase = t7",
                 "get_current_inference.duration.expected_duration = 22.747089383925076",
                 "get_current_inference.current_state.phase_probabilities.t7 = 0.7609993694101455"):
        assert _label_above(text, line) == v2r.LABEL_MODEL


def test_reference_and_identifier_blocks_are_neutral():
    text = _render()
    assert _label_above(text, "get_current_inference.window.window_start = 156") == v2r.LABEL_REFERENCE
    assert _label_above(text, "get_current_inference.model.model_name = semi_hmm") == v2r.LABEL_IDENTIFIERS
    assert "ni observation ni prediction" in v2r.LABEL_REFERENCE


# ---------------------------------------------------------------------------
# traceability: every value line is a v2 line, no value invented
# ---------------------------------------------------------------------------

def test_every_v2_value_line_survives_and_no_value_line_is_invented():
    ctx = _context(history=True, docs=True)
    v2_lines = _value_lines(prompt.build_user_prompt(ctx, "v2"))
    c2_lines = _value_lines(prompt.build_user_prompt(ctx, "compact_v2"))
    assert v2_lines <= c2_lines, v2_lines - c2_lines
    assert c2_lines <= v2_lines, c2_lines - v2_lines


def test_derived_transition_context_lines_are_labelled_derived_and_traceable():
    ctx = _context(full_history=True, transition_context=True)
    text = prompt.build_user_prompt(ctx, "compact_v2")
    line = "transition_context.last_annotated_transition_at_or_before_reference_window.to_phase = t6"
    assert _label_above(text, line) == v2r.LABEL_DERIVED
    assert "transition_context.provenance = derived_deterministic" in text
    # every derived value already exists verbatim in the OBSERVED block
    assert "get_transition_events.transitions[5].to_phase = t6" in text


def test_series_block_and_analysis_are_rendered_by_the_unchanged_renderers():
    text = _render(history=True)
    assert "=== SERIE TEMPORELLE MULTI-FENETRES" in text
    assert "ANALYSE DETERMINISTE DE LA SERIE" in text
    assert "WINDOW 156  [FENETRE CENTRALE / CENTER]  offset_from_center=0" in text
    assert _label_above(text, "=== SERIE TEMPORELLE MULTI-FENETRES (source : Reporting API — "
                        "get_inference_history) ===") == v2r.LABEL_SERIES
    assert "NOTE DE RECONCILIATION" in text


# ---------------------------------------------------------------------------
# derive_transition_context (the Q1 defect)
# ---------------------------------------------------------------------------

def test_transition_context_selects_last_at_or_before_and_next_after_the_window():
    payload = {"get_current_inference": copy.deepcopy(CURRENT_INFERENCE),
               "get_transition_events": _full_history_events()}
    tc = v2r.derive_transition_context(payload)
    last = tc["last_annotated_transition_at_or_before_reference_window"]
    nxt = tc["next_annotated_transition_after_reference_window"]
    assert tc["reference_window"] == 156
    assert (last["from_phase"], last["to_phase"], last["transitions_index"]) == ("t4", "t6", 5)
    assert last["is_skip"] is True and last["window_end_time"] == 41.0
    assert (nxt["from_phase"], nxt["to_phase"], nxt["window_start"]) == ("t6", "t7", 176)
    assert tc["n_annotated_transitions_at_or_before_reference_window"] == 6
    assert tc["n_annotated_transitions_after_reference_window"] == 6
    assert tc["provenance"] == "derived_deterministic"


def test_transition_context_is_empty_without_events_or_window():
    assert v2r.derive_transition_context({"get_current_inference": CURRENT_INFERENCE}) == {}
    # a single bracketing transition: nothing to select, no redundant block
    assert v2r.derive_transition_context({"get_current_inference": CURRENT_INFERENCE,
                                          "get_transition_events": TRANSITION_EVENTS}) == {}
    events = _full_history_events()
    assert v2r.derive_transition_context({"get_transition_events": events}) == {}  # window None
    assert v2r.derive_transition_context({}) == {}


def test_transition_context_never_names_a_model_phase():
    payload = {"get_current_inference": copy.deepcopy(CURRENT_INFERENCE),
               "get_transition_events": _full_history_events()}
    serialised = json.dumps(v2r.derive_transition_context(payload))
    assert "current_phase" not in serialised and "phase_probabilit" not in serialised
    assert "0.7609993694101455" not in serialised


# ---------------------------------------------------------------------------
# scientific reference vs project documentation
# ---------------------------------------------------------------------------

def _corpus_available():
    return retrieve_scientific_reference(Q13)["available"]


@pytest.mark.skipif(not _corpus_available(), reason="Istanbul corpus not on this machine")
def test_scientific_block_is_separate_from_project_docs_and_labelled_external():
    text = _render(Q13, docs=True, scientific=True)
    sci = text.split(v2r.LABEL_SCIENTIFIC, 1)[1].split(v2r.LABEL_PROJECT_DOCS, 1)[0]
    docs = text.split(v2r.LABEL_PROJECT_DOCS, 1)[1].split(v2r.LABEL_LIMITATIONS, 1)[0]
    assert "IC2025-" in sci and "[REFERENCE IC2025-" in sci and "[LIMITATION LIEE IC2025-" in sci
    assert "IC2025-" not in docs
    assert "[RAG DOCUMENT" in docs and "[RAG DOCUMENT" not in sci
    assert "jamais une mesure sur cet embryon" in v2r.LABEL_SCIENTIFIC
    assert "Istanbul" in v2r.LABEL_SCIENTIFIC and "Istanbul" not in v2r.LABEL_PROJECT_DOCS


@pytest.mark.skipif(not _corpus_available(), reason="Istanbul corpus not on this machine")
def test_scientific_block_text_is_verbatim_corpus_text():
    from validator.corpus import Corpus

    corpus = Corpus()
    ctx = _context(Q13, docs=True, scientific=True)
    text = prompt.build_user_prompt(ctx, "compact_v2")
    for entry in ctx["scientific_context"]["blocks"]:
        block = corpus.get(entry["block_id"])
        assert block is not None
        # the rendered body is the corpus text minus only its own duplicated heading
        body = block.text
        prefix = f"### {block.block_id} — {block.subject}"
        if body.startswith(prefix):
            body = body[len(prefix):].lstrip(" -")
        assert body[:80] in text


def test_no_scientific_block_when_the_context_carries_none():
    text = _render(docs=True)
    assert v2r.LABEL_SCIENTIFIC not in text
    assert v2r.LABEL_PROJECT_DOCS in text


def test_narrative_chunk_is_skipped_before_rendering_and_the_decision_is_visible():
    ctx = _context(docs=True)
    assert [c["source"] for c in ctx["document_context"]] == ["docs/SCIENTIFIC_REPORT.md"]
    assert ctx["retrieval_audit"][0]["decision"] == "skipped"
    assert ctx["retrieval_audit"][0]["rule"] == "D1"
    text = prompt.build_user_prompt(ctx, "compact_v2")
    assert "candidat(s) de documentation narrative ecarte(s)" in text
    assert "hidden coordinate" not in text          # the narrative content never reaches the prompt
    assert "tPB2, tPNa, tPNf" in text                # the factual chunk does, verbatim


def test_project_doc_chunk_keeps_source_section_status():
    text = _render(docs=True)
    assert "[RAG DOCUMENT 1] source: docs/SCIENTIFIC_REPORT.md | section: 2. Dataset | status: authoritative" in text


# ---------------------------------------------------------------------------
# limitations, warnings, long lists, trajectory split
# ---------------------------------------------------------------------------

def test_limitations_block_cites_the_context_fields_by_path():
    text = _render(history=True)
    lim = text.split(v2r.LABEL_LIMITATIONS, 1)[1].split(v2r.LABEL_WARNINGS, 1)[0]
    assert "get_current_inference.window.time_unit = unknown/unverified" in lim
    assert "get_current_inference.duration.duration_unit = windows" in lim
    assert "model_vs_observed_lag_windows = null" in lim
    assert "time_unit = hours" not in text


def test_warnings_block_is_last_and_never_dropped():
    ctx = _context()
    ctx["warnings"] = ["outil X indisponible"]
    text = prompt.build_user_prompt(ctx, "compact_v2")
    assert text.rstrip().endswith("- outil X indisponible")


def test_long_contiguous_list_is_summarised_losslessly_and_a_gap_prints_it_in_full():
    text = _render(trajectory=True)
    assert "get_trajectory.window_starts = 527 valeurs entieres contigues de 0 a 526" in text
    ctx = _context(trajectory=True)
    ctx["dynamic_context"]["get_trajectory"]["window_starts"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    text = prompt.build_user_prompt(ctx, "compact_v2")
    assert "get_trajectory.window_starts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14] (14 item(s))" in text


def test_transition_chain_is_split_between_annotation_and_model():
    text = _render(trajectory=True)
    assert _label_above(text, "get_trajectory.transition_chain[0].phase = t4") == v2r.LABEL_OBSERVED
    assert _label_above(text, "get_trajectory.transition_chain[0].observed_duration_windows = 37") == v2r.LABEL_OBSERVED
    assert _label_above(text, "get_trajectory.transition_chain[0].duration_probability_at_observed = 0.012") == v2r.LABEL_MODEL
    assert _label_above(text, "get_trajectory.transition_chain[0].next_phase_distribution.t5 = 0.7") == v2r.LABEL_MODEL


# ---------------------------------------------------------------------------
# no answer injection
# ---------------------------------------------------------------------------

def test_no_success_criterion_or_expected_answer_reaches_the_prompt():
    from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS

    text = _render(history=True, docs=True, scientific=True)
    for q in FIXED_QUESTIONS:
        for field in (q.success_criterion, q.expected_behavior, q.expected_information):
            if field:
                assert field not in text


def test_labels_carry_no_value_no_phase_no_number():
    for label in _LABELS:
        assert not re.search(r"\bt\d+\+?\b|\btPB2\b|\btEB\b|\d\.\d", label)

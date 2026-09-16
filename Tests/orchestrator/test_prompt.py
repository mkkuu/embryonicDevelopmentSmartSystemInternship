"""Pure tests for prompt.py -- no LLM, no torch, no chromadb."""

from orchestrator import prompt
from orchestrator.context_builder import build_context
from orchestrator.router import classify


def test_system_prompt_is_bounded_and_covers_the_contract_rules():
    text = prompt.build_system_prompt()
    # Task's own "ne pas faire un prompt gigantesque". The bound was 3000 for
    # the prohibition-only v1; v2 (2026-09-07) adds the reading method and
    # is bounded at 7000 chars (~2.3k tokens): the largest real benchmark
    # prompt is ~10.3k tokens, so system + user still fit num_ctx=16384.
    assert len(text) < 7000
    for keyword in ("invente", "CONTEXTE", "cite", "avertissement", "GRU", "raisonnement"):
        assert keyword.lower() in text.lower()


def test_system_prompt_v2_gives_reading_rules_not_only_prohibitions():
    """The 2026-09-03 audit: every value was present, the LLM still refused
    or swapped fields. v2 must tell the model HOW to read the context, by
    naming the rendered blocks/fields it must look at and the quantities it
    must never interchange."""
    text = prompt.build_system_prompt()
    for field_name in ("offset_from_center", "window_start_time", "model_vs_observed_lag_windows",
                       "model_phase_sequence", "n_phase_returns", "convergence_start_window",
                       "entropy_near_minus_far", "second_phase", "expected_duration"):
        assert field_name in text, f"the prompt must name the field {field_name!r}"
    assert "null" in text and "non calculable" in text
    assert "MODEL-DERIVED" in text and "OBSERVED" in text and "DERIVED" in text


def test_system_prompt_v2_contains_no_scientific_value():
    """Reading rules only: no probability, phase name, window id, time or
    expected answer may be written into the prompt."""
    import re
    text = prompt.build_system_prompt()
    assert not re.search(r"\bt\d+\+?\b|\btPB2\b|\btPN[af]\b|\btM\b|\btSB\b|\btB\b|\btEB\b", text), \
        "a phase name is written in the prompt"
    assert not re.search(r"\bW\d{2,}\b|\b0[.,]\d{3,}\b|\b\d{2,3}[.,]\d h\b", text), \
        "a window id / probability / time value is written in the prompt"
    assert "Patient_" not in text


def test_system_prompt_is_deterministic():
    assert prompt.build_system_prompt() == prompt.build_system_prompt()


def test_user_prompt_labels_document_and_dynamic_context_separately():
    ctx = build_context("q", classify("q"), {})
    text = prompt.build_user_prompt(ctx)
    assert "CONTEXTE DOCUMENT" in text
    assert "CONTEXTE DYNAMIQUE" in text
    assert "AVERTISSEMENTS" in text


def test_user_prompt_includes_question_verbatim():
    text = prompt.build_user_prompt(build_context("Qu'est-ce que le Semi-HMM ?", classify("q"), {}))
    assert "Qu'est-ce que le Semi-HMM ?" in text


def test_user_prompt_empty_context_says_so_explicitly():
    text = prompt.build_user_prompt(build_context("q", classify("q"), {}))
    assert "aucun document" in text.lower()
    assert "aucune donnée dynamique" in text.lower()


def test_user_prompt_surfaces_warnings():
    from orchestrator.context_builder import SOURCE_DOCUMENT, ToolCallResult

    results = {
        "retrieve_documents": ToolCallResult(
            tool_name="retrieve_documents", source_type=SOURCE_DOCUMENT, ok=False,
            error="index unavailable", error_type="ToolUnavailableError",
        ),
    }
    ctx = build_context("q", classify("q"), results)
    text = prompt.build_user_prompt(ctx)
    assert "index unavailable" in text

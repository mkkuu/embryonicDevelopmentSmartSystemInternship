"""Tests for grounding_check.py's comma/period decimal-separator
normalization (docs/PROJECT_CHECKPOINT.md 2026-08-27 Priority 1): a real
LLM answer restating a genuine dynamic_context value with a French
decimal comma (e.g. "1,0" for a known "1.0") was being flagged
"ungrounded" -- a false positive on a correctly-transcribed real value,
not a hallucination. No LLM, no torch, no chromadb."""

from orchestrator import grounding_check


def _context(document_context=None, dynamic_context=None, question="q", provenance=None):
    return {
        "question": question,
        "document_context": document_context or [],
        "dynamic_context": dynamic_context or {},
        "warnings": [],
        "provenance": provenance or [],
    }


# --- _canonical_number_form unit behavior -----------------------------------

def test_canonical_form_replaces_comma_with_period():
    assert grounding_check._canonical_number_form("1,0") == "1.0"
    assert grounding_check._canonical_number_form("0,6641144553128034") == "0.6641144553128034"


def test_canonical_form_is_idempotent_on_period_forms():
    assert grounding_check._canonical_number_form("1.0") == "1.0"
    assert grounding_check._canonical_number_form("268") == "268"


def test_canonical_form_preserves_sign():
    assert grounding_check._canonical_number_form("-0,5") == "-0.5"
    assert grounding_check._canonical_number_form("-0.5") == "-0.5"


def test_canonical_form_preserves_trailing_percent():
    assert grounding_check._canonical_number_form("7,2%") == "7.2%"


# --- the 10 requested scenarios ----------------------------------------------

def test_1_comma_decimal_matching_known_dot_value_is_grounded():
    ctx = _context(dynamic_context={"get_model_metadata": {"model_configuration": {"transition_smoothing_alpha": 1.0}}})
    result = grounding_check.check_grounding("Le parametre alpha est de 1,0.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_2_comma_decimal_matching_high_precision_known_value_is_grounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.6641144553128034}}})
    result = grounding_check.check_grounding(
        "La probabilite actuelle est de 0,6641144553128034 pour cette phase.", ctx,
    )
    assert result.grounded is True


def test_3_genuinely_different_value_remains_ungrounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.5}}})
    result = grounding_check.check_grounding("La probabilite est de 0.6.", ctx)
    assert result.grounded is False
    assert "0.6" in result.ungrounded_numbers


def test_4_negative_comma_decimal_matching_known_negative_value_is_grounded():
    ctx = _context(dynamic_context={"get_current_inference": {"some_delta": -0.5}})
    result = grounding_check.check_grounding("L'ecart est de -0,5.", ctx)
    assert result.grounded is True


def test_5_scientific_notation_is_not_silently_equated_to_anything():
    # _NUMBER_PATTERN has no exponent support -- "6.64e-1" is not matched
    # as one token (it splits into "6.64" and "-1"), so this fix must not
    # accidentally create a false equivalence between a real known value
    # (0.664) and an answer written in scientific notation for a
    # DIFFERENT real magnitude. Confirms current (unchanged) behavior:
    # scientific notation is neither specially supported nor falsely
    # matched by the new normalization.
    ctx = _context(dynamic_context={"get_current_inference": {"phase_probability": 0.664}})
    result = grounding_check.check_grounding("La valeur mesuree est de 6.64e-1.", ctx)
    # "6.64" (extracted whole) does not equal known "0.664" -- correctly
    # flagged, not silently accepted via any exponent-aware normalization
    # this fix does not add.
    assert "6.64" in result.ungrounded_numbers


def test_6_phase_names_are_never_treated_as_numbers():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"current_phase": "t6"}}})
    result = grounding_check.check_grounding("La phase actuelle est t6, avec t9+ en approche.", ctx)
    # "6" and "9" embedded in phase tokens must not appear as ungrounded
    # numeric claims (pre-existing _strip_phase_tokens behavior, unaffected
    # by the comma/period normalization added here).
    assert "6" not in result.ungrounded_numbers
    assert "9" not in result.ungrounded_numbers
    assert "t9+" in result.ungrounded_phase_tokens  # not present in dynamic_context/doc/question -- correctly flagged


def test_7_absent_value_stays_ungrounded_regardless_of_separator():
    ctx = _context(dynamic_context={"get_current_inference": {"phase_probability": 0.42}})
    result_comma = grounding_check.check_grounding("La probabilite est de 0,99.", ctx)
    result_dot = grounding_check.check_grounding("La probabilite est de 0.99.", ctx)
    assert result_comma.grounded is False
    assert result_dot.grounded is False


def test_8_normalization_never_conflates_a_different_number():
    # known = "1.5" only. An answer number with no comma at all ("15")
    # must never be matched via the comma-removal path -- canonicalizing
    # a comma-free token is a no-op, so "15" stays "15", never "1.5".
    ctx = _context(dynamic_context={"get_current_inference": {"some_value": 1.5}})
    result = grounding_check.check_grounding("Le compte est de 15.", ctx)
    assert "15" in result.ungrounded_numbers
    # And the reverse: "0.1" must never be treated as equal to "0.10".
    ctx2 = _context(dynamic_context={"get_current_inference": {"some_value": 0.1}})
    result2 = grounding_check.check_grounding("La valeur est de 0.10.", ctx2)
    assert "0.10" in result2.ungrounded_numbers


def test_9_provenance_only_numbers_are_not_treated_as_known():
    # context["provenance"] is metadata ABOUT facts, never itself read as
    # a fact source by check_grounding (unchanged by this fix) -- a number
    # appearing only there, not in dynamic_context's own leaves, must
    # still be reported ungrounded.
    ctx = _context(
        dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.2}}},
        provenance=[{"tool": "get_current_inference", "model_version": "alpha=9.9"}],
    )
    result = grounding_check.check_grounding("Le parametre alpha est de 9,9.", ctx)
    # ungrounded_numbers always reports the LLM's own original text
    # verbatim (never the canonicalized form) -- "9,9" as written, not "9.9".
    assert "9,9" in result.ungrounded_numbers


def test_10_preexisting_real_benchmark_case_now_resolves():
    # The exact real case from docs/LLM_EVALUATION.md sec 0a/0b:
    # model_configuration's alpha/rho embedded in model_version, restated
    # by the LLM with French decimal commas.
    ctx = _context(dynamic_context={
        "get_model_metadata": {
            "model_version": "dmax=268,negative_binomial,alpha=1.0,rho=0.3,class_weight=None",
        },
    })
    answer = "Le modele utilise alpha=1,0 et rho=0,3, avec dmax=268."
    result = grounding_check.check_grounding(answer, ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_preexisting_exact_match_behavior_is_unchanged():
    # A pre-existing, already-passing case (dot-to-dot exact match) must
    # keep working identically after this change.
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.42}}})
    result = grounding_check.check_grounding("La probabilite est de 0.42.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_document_text_comma_numbers_are_now_normalized_via_extraction():
    # UPDATED (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md sec 10):
    # doc_text comma-decimal numbers are now recognized via extraction-based
    # normalization (_known_numbers_from_document_text_canonical), not the
    # blind whole-string substitution this original test's comment warned
    # against -- see Tests/orchestrator/test_grounding_document_numeric_normalization.py
    # for the full new-behavior test suite. This test now documents the
    # RESOLVED case, not the boundary (the boundary tests -- "t6, puis t7",
    # "1, 2, 3" -- moved to that file).
    ctx = _context(document_context=[{"content": "Le taux mesure est de 0,44 sur ce benchmark."}])
    result = grounding_check.check_grounding("Le taux est 0.44.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []

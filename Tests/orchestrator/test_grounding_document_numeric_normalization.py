"""Tests for grounding_check.py's comma/period decimal-separator
normalization on the RAG DOCUMENT-TEXT side (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md
sec 6/10 -- the P1 carried over from the 2026-08-27 dynamic_context-only
fix, addressed this session). Real cases observed in earlier benchmark
runs: an LLM answer citing genuine `document_context` values (`p=0,29`,
`Brier=0,11483`) with a French decimal comma was flagged "ungrounded"
even though the identical value exists, period-form, verbatim in the
retrieved document. No LLM, no torch, no chromadb -- `check_grounding()`
and its helpers are pure functions over plain dicts/strings.

Companion file: Tests/orchestrator/test_grounding_numeric_normalization.py
(the original dynamic_context-side fix, unchanged by this session)."""

from orchestrator import grounding_check


def _context(document_context=None, dynamic_context=None, question="q", provenance=None):
    return {
        "question": question,
        "document_context": document_context or [],
        "dynamic_context": dynamic_context or {},
        "warnings": [],
        "provenance": provenance or [],
    }


# --- Unicode minus sign (U+2212) normalization ------------------------------
# Real docs (docs/GLOBAL_MODEL_COMPARISON.md) write negative deltas with a
# typographic minus sign ("Brier −0.0004"), not ASCII hyphen-minus. Found
# while reproducing the exact real "-0,0004" case end-to-end with real
# document content (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md sec 11).

def test_canonical_form_normalizes_unicode_minus_to_ascii_hyphen():
    assert grounding_check._canonical_number_form("−0.0004") == "-0.0004"
    assert grounding_check._canonical_number_form("-0.0004") == "-0.0004"  # idempotent


def test_extraction_recognizes_unicode_minus_as_a_sign():
    assert grounding_check.extract_numbers("Brier −0.0004 sur ce benchmark.") == ["−0.0004"]


def test_doc_text_real_document_unicode_minus_matches_answer_ascii_hyphen():
    # The exact real end-to-end case: docs/GLOBAL_MODEL_COMPARISON.md's own
    # verbatim prose (Unicode minus), an LLM answer restating it with a
    # French comma AND an ordinary ASCII hyphen (how any provider actually
    # writes a negative number).
    doc_content = (
        "small, real, directionally consistent gain (AUROC +0.005, Brier "
        "−0.0004, ECE −0.00005) but not trajectory-significant (p=0.29)"
    )
    ctx = _context(document_context=[
        {"content": doc_content, "source": "docs/GLOBAL_MODEL_COMPARISON.md", "section": "10"},
    ])
    result = grounding_check.check_grounding(
        "Le gain observe est AUROC +0,005 et Brier -0,0004.", ctx,
    )
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_unspaced_subtraction_ambiguity_is_now_consistent_across_both_sign_characters():
    # NOT a new risk category (documented in _NUMBER_PATTERN's own comment):
    # an unspaced subtraction expression immediately adjacent to a digit was
    # ALREADY misread as a negative literal for the ASCII hyphen before this
    # session ("window_size-1=7" -> "-1"). This test documents that the
    # Unicode minus now behaves the SAME way (parity), not a new, worse
    # failure mode.
    assert grounding_check.extract_numbers("window_size-1=7") == ["-1", "7"]
    assert grounding_check.extract_numbers("window_size−1=7") == ["−1", "7"]  # raw extraction keeps
    # the original character (U+2212) -- only _canonical_number_form (used
    # for KNOWN-set comparison, never for what's reported back) normalizes
    # it to ASCII "-".
    assert grounding_check._canonical_number_form("−1") == "-1"
    # Spaced subtraction (the far more common real style in this project's
    # docs) is unaffected either way -- never merged into a signed token.
    assert grounding_check.extract_numbers("0.7638 − 0.7501") == ["0.7638", "0.7501"]


# --- extraction-boundary unit tests (the safety mechanism itself) ----------

def test_extraction_merges_comma_decimal_with_no_space_into_one_token():
    assert grounding_check.extract_numbers("Le taux est de 0,005.") == ["0,005"]
    assert grounding_check.extract_numbers("L'ecart est de -0,0004 environ.") == ["-0,0004"]


def test_extraction_does_not_merge_comma_followed_by_space():
    # The exact safety boundary this fix relies on: a comma immediately
    # followed by a SPACE (ordinary punctuation, list enumeration) never
    # forms a single numeric token, unlike a comma immediately followed
    # by a digit (a genuine decimal separator).
    assert grounding_check.extract_numbers("La phase passe de t6, puis t7.") == ["6", "7"]
    assert grounding_check.extract_numbers("Les valeurs sont 1, 2, 3 dans l'ordre.") == ["1", "2", "3"]


def test_known_numbers_from_document_text_canonicalizes_only_merged_tokens():
    doc_text = "Le taux mesure est de 0,44 ; les rangs sont 1, 2, 3."
    known = grounding_check._known_numbers_from_document_text_canonical(doc_text)
    assert "0.44" in known  # the genuine decimal, canonicalized
    # "1", "2", "3" pass through unchanged (no comma inside any of these
    # single-digit tokens to canonicalize) -- never merged into "1.2.3"
    # or any other spurious combined value.
    assert known == {"0.44", "1", "2", "3"}


def test_known_numbers_from_document_text_leaves_ambiguous_thousands_forms_unresolved():
    # Explicitly OUT OF SCOPE, same posture as the original dynamic_context
    # fix (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md sec 8): a
    # thousands-grouped/compound-separator form is not correctly parsed by
    # _NUMBER_PATTERN at all (pre-existing, unchanged) -- it splits into
    # fragments rather than one coherent value, on BOTH the dynamic_context
    # and document_text sides equally. Documented here, not silently
    # assumed to work.
    assert grounding_check.extract_numbers("Total: 1 234,56 euros.") == ["1", "234,56"]
    assert grounding_check.extract_numbers("Total: 1,234.56 dollars.") == ["1,234", "56"]


# --- check_grounding() integration: the real reported cases -----------------

def test_doc_text_period_form_matches_answer_comma_form_p_value():
    # Reconstructs the exact real case from docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md
    # sec 6: a genuine value written period-form in a real project doc
    # (docs/MODEL_COMPARISON.md's own style), restated by the LLM with a
    # French decimal comma.
    ctx = _context(document_context=[
        {"content": "Le test statistique donne p=0.29 pour cette comparaison.",
         "source": "docs/MODEL_COMPARISON.md", "section": "7"},
    ])
    result = grounding_check.check_grounding("La valeur rapportee est p=0,29.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_doc_text_period_form_matches_answer_comma_form_brier_score():
    # The second real case from sec 6: Brier=0,11483.
    ctx = _context(document_context=[
        {"content": "Le score Brier obtenu est de 0.11483 sur le jeu de validation.",
         "source": "docs/SCIENTIFIC_REPORT.md", "section": "3"},
    ])
    result = grounding_check.check_grounding("Le Brier score est de 0,11483.", ctx)
    assert result.grounded is True


def test_doc_text_comma_form_matches_answer_dot_form_reverse_direction():
    # Symmetry check: the fix is a canonical-form SET comparison, not
    # order-dependent -- works equally if the DOCUMENT happens to use a
    # comma and the LLM restates with a period.
    ctx = _context(document_context=[{"content": "Le taux observe est de 0,44."}])
    result = grounding_check.check_grounding("Le taux est 0.44.", ctx)
    assert result.grounded is True


def test_doc_negative_comma_decimal_matches_dot_form():
    ctx = _context(document_context=[{"content": "La variation est de -0.0004 par etape."}])
    result = grounding_check.check_grounding("La variation est de -0,0004.", ctx)
    assert result.grounded is True


def test_doc_1_25_equivalence():
    ctx = _context(document_context=[{"content": "Le ratio calcule est de 1.25."}])
    result = grounding_check.check_grounding("Le ratio est 1,25.", ctx)
    assert result.grounded is True


def test_doc_percentage_equivalence():
    ctx = _context(document_context=[{"content": "Le taux de reussite est de 7.2%."}])
    result = grounding_check.check_grounding("Le taux est de 7,2%.", ctx)
    assert result.grounded is True


# --- false-positive guards: what must NOT become grounded -------------------

def test_doc_list_enumeration_does_not_falsely_ground_an_unrelated_number():
    # "1, 2, 3" (comma-space) must never be read as a merged value -- an
    # answer citing a genuinely different, absent number must stay
    # ungrounded.
    ctx = _context(document_context=[{"content": "Les rangs observes sont 1, 2, 3 dans cet ordre."}])
    result = grounding_check.check_grounding("Le rang mesure est de 23.", ctx)
    assert "23" in result.ungrounded_numbers


def test_doc_genuinely_different_value_stays_ungrounded_regardless_of_separator():
    ctx = _context(document_context=[{"content": "Le score est de 0.29 pour ce modele."}])
    result_comma = grounding_check.check_grounding("Le score est 0,30.", ctx)
    result_dot = grounding_check.check_grounding("Le score est 0.30.", ctx)
    assert result_comma.grounded is False
    assert result_dot.grounded is False


def test_doc_normalization_never_conflates_a_comma_free_answer_number():
    # A comma-free answer token is a no-op under canonicalization -- must
    # never spuriously match a differently-valued known document number.
    ctx = _context(document_context=[{"content": "La valeur de reference est 1,5."}])
    result = grounding_check.check_grounding("Le total est de 15.", ctx)
    assert "15" in result.ungrounded_numbers


def test_doc_already_dot_form_values_behave_identically_to_before_the_fix():
    # Regression safety: a pre-existing dot-to-dot exact match (already
    # working before this session) must be completely unaffected.
    ctx = _context(document_context=[{"content": "La duree moyenne est de 42.0 fenetres."}])
    result = grounding_check.check_grounding("La duree est de 42.0 fenetres.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


# --- unchanged/untouched surfaces --------------------------------------------

def test_ungrounded_numbers_still_report_the_llm_original_text_verbatim():
    # Full transparency preserved: even when a number IS matched via the
    # new canonical doc-text set, ungrounded_numbers (for numbers that do
    # NOT match) must still report the original text, never a
    # canonicalized rewrite -- and checked_numbers always includes every
    # extracted token regardless of grounded status.
    ctx = _context(document_context=[{"content": "Le score est de 0.29."}])
    result = grounding_check.check_grounding("Le score rapporte est p=0,29, un autre est 0,99.", ctx)
    assert "0,29" in result.checked_numbers
    assert "0,99" in result.ungrounded_numbers  # genuinely absent, stays as originally written


def test_dynamic_context_fix_and_document_text_fix_compose_correctly():
    # A single answer citing one number from dynamic_context (comma form)
    # and one number from document_context (comma form) must have BOTH
    # resolved -- the two KNOWN sets are independent, not mutually exclusive.
    ctx = _context(
        dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.664}}},
        document_context=[{"content": "Le Brier score de reference est 0.11483."}],
    )
    result = grounding_check.check_grounding(
        "La probabilite est de 0,664 et le Brier score associe est 0,11483.", ctx,
    )
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_provenance_only_document_numbers_still_not_treated_as_known():
    # context["provenance"] is not read by check_grounding at all (pre-
    # existing, unchanged) -- a number appearing only there must still be
    # reported ungrounded regardless of separator.
    ctx = _context(
        document_context=[{"content": "Voir le rapport pour les details."}],
        provenance=[{"tool": "retrieve_documents", "value": "9,9"}],
    )
    result = grounding_check.check_grounding("La valeur est de 9,9.", ctx)
    assert "9,9" in result.ungrounded_numbers

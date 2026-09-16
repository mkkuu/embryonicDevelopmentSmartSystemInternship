"""Pure tests for grounding_check.py -- no LLM, no torch, no chromadb."""

from orchestrator import grounding_check


def _context(document_context=None, dynamic_context=None, question="q", warnings=None):
    return {
        "question": question,
        "document_context": document_context or [],
        "dynamic_context": dynamic_context or {},
        "warnings": warnings or [],
    }


def test_extract_numbers_finds_decimals_and_percents():
    assert grounding_check.extract_numbers("P=0.72, soit 72% environ, sur 15 fenetres") == \
        ["0.72", "72%", "15"]


def test_number_present_in_dynamic_context_is_grounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.42}}})
    result = grounding_check.check_grounding("La probabilite est de 0.42.", ctx)
    assert result.grounded is True
    assert result.ungrounded_numbers == []


def test_number_absent_from_context_is_ungrounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.42}}})
    result = grounding_check.check_grounding("La probabilite est de 0.999.", ctx)
    assert result.grounded is False
    assert "0.999" in result.ungrounded_numbers


def test_number_present_in_document_text_is_grounded():
    ctx = _context(document_context=[{"content": "Le taux mesure est de 0.44 sur ce benchmark."}])
    result = grounding_check.check_grounding("Le taux est 0.44.", ctx)
    assert result.grounded is True


def test_small_common_numbers_are_never_flagged():
    ctx = _context()
    result = grounding_check.check_grounding("Il y a 2 sources et 1 avertissement.", ctx)
    assert result.ungrounded_numbers == []


def test_phase_token_present_in_dynamic_context_is_grounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"current_phase": "t6"}}})
    result = grounding_check.check_grounding("La phase actuelle est t6.", ctx)
    assert result.grounded is True
    assert "t6" in result.checked_phase_tokens


def test_phase_token_present_in_question_is_grounded():
    ctx = _context(question="Pourquoi t6 correspond-il a cette phase ?")
    result = grounding_check.check_grounding("t6 est une phase precoce.", ctx)
    assert result.grounded is True


def test_phase_token_absent_everywhere_is_ungrounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"current_phase": "t6"}}})
    result = grounding_check.check_grounding("La phase actuelle est t9+.", ctx)
    assert result.grounded is False
    assert "t9+" in result.ungrounded_phase_tokens


def test_empty_answer_is_trivially_grounded():
    ctx = _context()
    result = grounding_check.check_grounding("", ctx)
    assert result.grounded is True
    assert result.checked_numbers == []


def test_has_any_context_false_when_both_empty():
    assert grounding_check.has_any_context(_context()) is False


def test_has_any_context_true_when_dynamic_present():
    ctx = _context(dynamic_context={"get_model_metadata": {"model_name": "semi_hmm"}})
    assert grounding_check.has_any_context(ctx) is True


def test_estimate_confidence_unknown_when_no_context():
    ctx = _context()
    result = grounding_check.check_grounding("", ctx)
    assert grounding_check.estimate_confidence(ctx, result) == "unknown"


def test_estimate_confidence_low_when_ungrounded():
    ctx = _context(dynamic_context={"get_current_inference": {"current_state": {"phase_probability": 0.1}}})
    result = grounding_check.check_grounding("La probabilite est de 0.999.", ctx)
    assert grounding_check.estimate_confidence(ctx, result) == "low"


def test_estimate_confidence_medium_when_grounded_but_warnings_present():
    ctx = _context(document_context=[{"content": "x"}], warnings=["some known gap"])
    result = grounding_check.check_grounding("texte sans nombre", ctx)
    assert grounding_check.estimate_confidence(ctx, result) == "medium"


def test_estimate_confidence_high_when_grounded_and_no_warnings():
    ctx = _context(document_context=[{"content": "x"}])
    result = grounding_check.check_grounding("texte sans nombre", ctx)
    assert grounding_check.estimate_confidence(ctx, result) == "high"


def test_no_information_message_is_the_canonical_french_sentence():
    assert "ne dispose pas de suffisamment d'informations" in grounding_check.NO_INFORMATION_MESSAGE


# --- Live phase-conflict check (docs/RAG_LLM_QUALITY_REPORT.md P0 item 1) ---
# A wrong phase name that also happens to appear in retrieved document text
# (e.g. a vocabulary table enumerating every phase) must not be accepted as
# grounded just because it is "known anywhere" -- when this turn actually
# fetched a live phase, the claim must overlap with that live data.

def test_wrong_phase_matching_doc_text_but_not_live_data_is_ungrounded():
    """Reproduces the real C2 finding: LLM states 't4', real live phase is
    't7', and 't4' is ALSO present in retrieved document text (e.g. a
    phase-vocabulary table) -- previously grounded=True, zero warnings."""
    ctx = _context(
        dynamic_context={"get_current_inference": {"current_state": {"current_phase": "t7"}}},
        document_context=[{"content": "Les phases incluent t3, t4, t6, t7, t9+ dans cet ordre."}],
    )
    result = grounding_check.check_grounding("La phase actuelle est t4.", ctx)
    assert result.grounded is False
    assert "t4" in result.ungrounded_phase_tokens
    assert "t4" in result.conflicting_phase_tokens


def test_correct_live_phase_plus_extra_doc_phase_is_still_grounded():
    """An answer correctly citing the live phase AND an additional phase
    from documentation (e.g. "what typically follows") must not be
    penalized -- the live-conflict check is 'any overlap', not 'every
    token must be live'."""
    ctx = _context(
        dynamic_context={"get_current_inference": {"current_state": {"current_phase": "tPNf"}}},
        document_context=[{"content": "Apres tPNf, la phase suivante est generalement t9+."}],
    )
    result = grounding_check.check_grounding(
        "La phase actuelle est tPNf, qui succede generalement a t9+.", ctx,
    )
    assert result.grounded is True
    assert result.conflicting_phase_tokens == []


def test_doc_only_phase_still_grounded_when_no_live_phase_data_this_turn():
    """Pure DOCUMENTARY question, no dynamic_context at all -- the
    live-conflict check must not fire when there is no live data to
    conflict with; behavior is unchanged from before this fix."""
    ctx = _context(document_context=[{"content": "t6 est une phase precoce du developpement."}])
    result = grounding_check.check_grounding("t6 est une phase precoce.", ctx)
    assert result.grounded is True
    assert result.conflicting_phase_tokens == []


def test_live_phase_conflict_reported_separately_from_unknown_phase():
    """conflicting_phase_tokens (known via doc text but disjoint from live
    data) and a phase token unknown anywhere at all both land in
    ungrounded_phase_tokens, but only the former is also surfaced in
    conflicting_phase_tokens -- callers that need to distinguish "wrong
    but plausible" from "never mentioned anywhere" can."""
    ctx = _context(
        dynamic_context={"get_current_inference": {"current_state": {"current_phase": "t7"}}},
        document_context=[{"content": "t4 est mentionne ici, sans rapport avec ce patient."}],
    )
    result = grounding_check.check_grounding("La phase est t4, ou peut-etre t9+.", ctx)
    assert "t4" in result.conflicting_phase_tokens
    assert "t9+" not in result.conflicting_phase_tokens
    assert set(result.ungrounded_phase_tokens) == {"t4", "t9+"}


# --- Event/Transition RAG axis (docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md
# sec 9) -- get_transition_events's from_phase/to_phase are plain string
# VALUES inside its returned dict (never dict keys, unlike the
# phase_probabilities/next_phase_distribution blind spot found in P2.4,
# docs/RAG_LLM_IMPROVEMENT_REPORT.md), so the EXISTING
# _known_phase_tokens()/_dynamic_phase_tokens() machinery (both walk dict
# VALUES via _collect_strings) already grounds them correctly with ZERO
# grounding_check.py code change -- these tests prove that, rather than
# assume it.

def test_transition_event_phase_tokens_are_grounded_via_existing_mechanism():
    ctx = _context(dynamic_context={"get_transition_events": {
        "video_id": "Patient_319", "split": "val", "n_transitions": 1,
        "transitions": [{"from_phase": "t5", "to_phase": "t7", "observed": True,
                          "provenance": "observed_annotation", "phase_distance": 2, "is_skip": True}],
    }})
    result = grounding_check.check_grounding(
        "La transition observee va de t5 a t7, ce qui constitue un saut de phase.", ctx,
    )
    assert result.grounded is True
    assert result.ungrounded_phase_tokens == []


def test_fabricated_transition_phase_not_in_observed_list_is_ungrounded():
    """A phase name that never appears anywhere in get_transition_events'
    own output (nor document_context, nor the question) must still be
    caught as ungrounded -- same mechanism as every other tool, no
    special-casing needed for this new tool."""
    ctx = _context(dynamic_context={"get_transition_events": {
        "video_id": "Patient_319", "split": "val", "n_transitions": 1,
        "transitions": [{"from_phase": "t5", "to_phase": "t7", "observed": True,
                          "provenance": "observed_annotation", "phase_distance": 2, "is_skip": True}],
    }})
    result = grounding_check.check_grounding("La transition observee va de t5 a t9.", ctx)
    assert result.grounded is False
    assert "t9" in result.ungrounded_phase_tokens


# --- Relational grounding (docs/RELATIONAL_GROUNDING.md) -- token-level
# grounding alone cannot distinguish a real observed transition PAIR from
# two individually-real tokens combined into a fabricated relation. Real
# bug found via real validation (docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md
# sec 10, question Q3): "il y a une transition entre phases t7 et t4"
# passed grounded=True even though the real transition is t4 -> t6, not
# t7 -> t4 (both t7 and t4 individually appear somewhere this turn).

def _transition_ctx(from_phase="t4", to_phase="t6", is_skip=True, extra_dynamic=None):
    dynamic_context = {"get_transition_events": {
        "video_id": "Patient_319", "split": "val", "window": 156, "n_transitions": 1,
        "transitions": [{"from_phase": from_phase, "to_phase": to_phase, "observed": True,
                          "provenance": "observed_annotation",
                          "phase_distance": 2 if is_skip else 1, "is_skip": is_skip}],
    }}
    if extra_dynamic:
        dynamic_context.update(extra_dynamic)
    return _context(dynamic_context=dynamic_context)


# TEST 1 (task sec 12): correct ordered pair -> GROUNDED.
def test_correct_transition_pair_is_grounded():
    ctx = _transition_ctx("t4", "t6")
    result = grounding_check.check_grounding("La transition observee est t4 -> t6.", ctx)
    assert result.grounded is True
    assert result.ungrounded_transition_claims == []
    assert {"from_phase": "t4", "to_phase": "t6"} in result.checked_transition_claims


# TEST 2 (task sec 12): fabricated pair, both tokens individually real
# elsewhere -> UNGROUNDED. Reproduces the exact real Q3 bug, using the
# exact real Q3 wording ("entre ... et").
def test_fabricated_transition_pair_is_ungrounded_reproduces_real_q3():
    ctx = _transition_ctx("t4", "t6", extra_dynamic={
        "get_current_inference": {"current_state": {"current_phase": "t7"}},
    })
    result = grounding_check.check_grounding(
        "Il y a une transition entre phases t7 et t4.", ctx,
    )
    assert result.grounded is False
    assert {"from_phase": "t7", "to_phase": "t4"} in result.ungrounded_transition_claims


def test_fabricated_transition_pair_is_ungrounded_with_vers_connector():
    """Same fabrication, task's own 'vers' phrasing -- confirms the fix
    is not narrowly tied to one specific connector wording."""
    ctx = _transition_ctx("t4", "t6", extra_dynamic={
        "get_current_inference": {"current_state": {"current_phase": "t7"}},
    })
    result = grounding_check.check_grounding("La transition observee est t7 vers t4.", ctx)
    assert result.grounded is False
    assert {"from_phase": "t7", "to_phase": "t4"} in result.ungrounded_transition_claims


# TEST 3 (task sec 12): real transition + correct is_skip claim -> GROUNDED.
def test_correct_skip_claim_is_grounded():
    ctx = _transition_ctx("t4", "t6", is_skip=True)
    result = grounding_check.check_grounding("t4 -> t6 est un saut de phase.", ctx)
    assert result.grounded is True
    assert result.skip_claim_mismatches == []


# TEST 4 (task sec 12): a DIFFERENT (fabricated) pair claimed as a skip -> UNGROUNDED.
def test_wrong_pair_claimed_as_skip_is_ungrounded():
    ctx = _transition_ctx("t4", "t6", is_skip=True)
    result = grounding_check.check_grounding("t4 -> t5 est un skip.", ctx)
    assert result.grounded is False
    assert {"from_phase": "t4", "to_phase": "t5"} in result.ungrounded_transition_claims


def test_real_pair_wrongly_claimed_as_skip_is_flagged_via_is_skip_mismatch():
    """A real, correctly-matched pair, but the REAL transition is NOT a
    skip -- the LLM's own "skip" language must never be taken at face
    value (task sec 10: "ne pas deduire is_skip uniquement du texte du
    LLM"); the real is_skip field is what decides."""
    ctx = _transition_ctx("t4", "t6", is_skip=False)
    result = grounding_check.check_grounding("t4 -> t6 est un skip.", ctx)
    assert result.grounded is False
    assert result.skip_claim_mismatches == [{"from_phase": "t4", "to_phase": "t6", "real_is_skip": False}]


# TEST 5 (task sec 12): single-token claim, no relation asserted -> GROUNDED
# (unaffected by the relational check -- no connector, no pair extracted).
def test_single_token_observed_claim_is_grounded():
    ctx = _transition_ctx("t4", "t6")
    result = grounding_check.check_grounding("La phase t6 est observee.", ctx)
    assert result.grounded is True
    assert result.checked_transition_claims == []


# TEST 6 (task sec 12): single-token claim, unsupported anywhere -> UNGROUNDED.
def test_single_token_unsupported_claim_is_ungrounded():
    ctx = _transition_ctx("t4", "t6")
    result = grounding_check.check_grounding("t7 est la phase observee.", ctx)
    assert result.grounded is False
    assert "t7" in result.ungrounded_phase_tokens


# --- Anti-overcorrection (task sec 7): enumeration must never be misread
# as a transition claim.

def test_enumeration_with_et_but_no_entre_is_not_a_transition_claim():
    ctx = _transition_ctx("t4", "t6")
    result = grounding_check.check_grounding("Les phases t4 et t6 apparaissent ici.", ctx)
    assert result.grounded is True
    assert result.checked_transition_claims == []  # relational check never fired


def test_enumeration_with_comma_is_not_a_transition_claim():
    ctx = _transition_ctx("t4", "t6")
    result = grounding_check.check_grounding(
        "Les phases incluent t3, t4, t6, t7, t9+ dans cet ordre.", ctx,
    )
    assert result.checked_transition_claims == []


# --- Graceful no-op when get_transition_events was not called this turn
# (nothing real to verify a relation against -- same discipline as the
# existing live-conflict check).

def test_relational_check_does_not_fire_when_transition_events_not_called():
    ctx = _context(dynamic_context={"get_current_inference": {
        "current_state": {"current_phase": "t7"}, "ground_truth": {"ground_truth_phase": "t4"},
    }})
    result = grounding_check.check_grounding("t7 vers t4.", ctx)
    assert result.ungrounded_transition_claims == []
    assert result.checked_transition_claims == []


def test_relational_check_flags_a_claim_against_a_real_but_empty_transition_list():
    """get_transition_events WAS called and genuinely found zero
    transitions -- a claimed pair must still be flagged (real information,
    not an absence of information), distinct from the "not called at all"
    case above."""
    ctx = _context(dynamic_context={"get_transition_events": {
        "video_id": "Patient_319", "split": "val", "window": 0, "n_transitions": 0, "transitions": [],
    }})
    result = grounding_check.check_grounding("La transition est t4 -> t6.", ctx)
    assert result.grounded is False
    assert {"from_phase": "t4", "to_phase": "t6"} in result.ungrounded_transition_claims


def test_extract_transition_claims_covers_the_five_task_phrasings():
    """docs/RELATIONAL_GROUNDING.md sec 9 -- the 5 phrasings explicitly
    named in the task brief, checked directly against the extractor."""
    assert grounding_check.extract_transition_claims("t4 -> t6") == [("t4", "t6")]
    assert grounding_check.extract_transition_claims("t4 vers t6") == [("t4", "t6")]
    assert grounding_check.extract_transition_claims("de t4 a t6") == [("t4", "t6")]
    assert grounding_check.extract_transition_claims("transition t4-t6") == [("t4", "t6")]
    assert grounding_check.extract_transition_claims("passage de t4 a t6") == [("t4", "t6")]

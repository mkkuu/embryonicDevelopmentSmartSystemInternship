"""Deterministic tests for the two grounding false negatives fixed on
2026-09-07d (`Training/orchestrator/grounding_check.py`).

Both were found by the 2026-09-07c replication and reproduced in 4/4 runs
on byte-identical context, so neither is LLM variance:

  A. an exact MODEL-DERIVED transition pair (`series_analysis.convergence.
     all_model_phase_changes`) was always ungrounded, because the only
     reference set was the ANNOTATED transition list;
  B. a phase name reachable only as a KEY of `phase_probabilities` was
     invisible, because the string collector walks dict VALUES.

Pure: no LLM, no torch, no chromadb, no GPU. Fixtures mirror the real
payload shapes (`Training/reporting/schemas.py`,
`Training/orchestrator/temporal_context.py`) at the frozen benchmark
anchor Patient_319 / val / window 156 -- the values are real, but nothing
here reads the dataset.
"""

from orchestrator import grounding_check


def _context(document_context=None, dynamic_context=None, question="q", warnings=None):
    return {
        "question": question,
        "document_context": document_context or [],
        "dynamic_context": dynamic_context or {},
        "warnings": warnings or [],
    }


# The model's own posterior at the anchor: t7 dominant, t8 runner-up, then a
# 13-state tail whose mass is negligible (t4 is 3.8e-25 -- the phase the
# live-conflict check must KEEP rejecting).
_PHASE_PROBABILITIES = {
    "tPB2": 0.0, "tPNa": 5.017534741302858e-202, "tPNf": 4.6124866171006075e-191,
    "t2": 2.922956161138927e-35, "t3": 1.9324091965945717e-32,
    "t4": 3.7983819633435514e-25, "t5": 0.0017052671933268549,
    "t6": 1.1459818142527895e-05, "t7": 0.7609993694101455,
    "t8": 0.23679880956344662, "t9+": 0.00043476617122871367,
    "tM": 5.0318367115227474e-05, "tSB": 7.441233506711718e-09,
    "tB": 4.679757741290383e-10, "tEB": 1.5673849175679434e-09,
}


def _current_inference(current_phase="t7", probabilities=None):
    return {"get_current_inference": {
        "current_state": {"current_phase": current_phase,
                          "phase_probability": 0.7609993694101455,
                          "phase_probabilities": probabilities if probabilities is not None
                          else dict(_PHASE_PROBABILITIES)},
        "next_phase": {"most_likely_next_phase": current_phase},
    }}


def _series_analysis(model_changes=(("t7", "t8"),), skips=()):
    """`temporal_context.derive_series_analysis()`'s shape, reduced to the
    two blocks the relational check reads."""
    return {"series_analysis": {
        "provenance": "derived_deterministic",
        "convergence": {"all_model_phase_changes": [
            {"from_window": 156, "to_window": 157, "from_phase": f, "to_phase": t}
            for f, t in model_changes]},
        "stability": {
            "n_phase_skips": len(skips),
            "phase_skips": [{"from_window": 156, "to_window": 157,
                             "from_phase": f, "to_phase": t, "phase_distance": 2}
                            for f, t in skips],
            "phase_returns": [], "n_phase_returns": 0,
            "phase_regressions": [], "n_phase_regressions": 0,
        },
    }}


def _observed_events(pairs=(("t4", "t6"),), is_skip=True):
    return {"get_transition_events": {
        "video_id": "Patient_319", "split": "val", "window": 156,
        "n_transitions": len(pairs),
        "transitions": [{"from_phase": f, "to_phase": t, "observed": True,
                         "provenance": "observed_annotation",
                         "phase_distance": 2 if is_skip else 1, "is_skip": is_skip}
                        for f, t in pairs],
    }}


def _provenance(result):
    return {(c["from_phase"], c["to_phase"]): c["matches"]
            for c in result.transition_claim_provenance}


# --- A. MODEL-DERIVED transition pairs --------------------------------------

def test_1_exact_model_derived_pair_is_grounded():
    """THE fixed false negative: `t7 -> t8` is the model's own change, printed
    in the prompt, and was reported ungrounded 4/4 before this fix."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(),
                                    **_observed_events()})
    result = grounding_check.check_grounding(
        "Le modele predit une transition de t7 a t8 entre les deux fenetres.", ctx)
    assert result.ungrounded_transition_claims == []
    assert result.grounded is True
    assert _provenance(result)[("t7", "t8")] == "model"


def test_2_model_pair_absent_from_the_context_is_ungrounded():
    """The model's real change is `t7 -> t8`; a claim about `t7 -> t9+` is a
    fabricated model relation and must stay flagged."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(),
                                    **_observed_events()})
    result = grounding_check.check_grounding("Le modele passe de t7 a t9+.", ctx)
    assert {"from_phase": "t7", "to_phase": "t9+"} in result.ungrounded_transition_claims
    assert result.grounded is False
    assert _provenance(result)[("t7", "t9+")] == "none"


def test_3_observed_pair_behaviour_is_unchanged():
    """The historical path: an annotated pair stays grounded and is now
    labelled `observed`, so the two classes remain distinguishable."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(),
                                    **_observed_events()})
    result = grounding_check.check_grounding("La transition observee est t4 -> t6.", ctx)
    assert result.ungrounded_transition_claims == []
    assert result.grounded is True
    assert _provenance(result)[("t4", "t6")] == "observed"
    # the shape the pre-existing tests assert must be preserved exactly
    assert {"from_phase": "t4", "to_phase": "t6"} in result.checked_transition_claims


def test_4_pair_matching_neither_model_nor_annotation_is_ungrounded():
    """The real Q11 hallucination of the replication: `t4 -> t7` is neither
    an annotated transition (`t4 -> t6`) nor a model change (`t7 -> t8`)."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(),
                                    **_observed_events()})
    result = grounding_check.check_grounding(
        "La transition de t4 a t7 est une progression normale.", ctx)
    assert {"from_phase": "t4", "to_phase": "t7"} in result.ungrounded_transition_claims
    assert result.grounded is False
    assert _provenance(result)[("t4", "t7")] == "none"


def test_a_pair_that_is_both_observed_and_predicted_is_labelled_both():
    ctx = _context(dynamic_context={**_current_inference(),
                                    **_series_analysis(model_changes=(("t4", "t6"),)),
                                    **_observed_events()})
    result = grounding_check.check_grounding("La transition t4 -> t6 a eu lieu.", ctx)
    assert result.grounded is True
    assert _provenance(result)[("t4", "t6")] == "both"


def test_model_pairs_are_checkable_without_get_transition_events():
    """A turn that carries series_analysis but no annotated transitions still
    has a real reference set -- the relational check must fire on it."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis()})
    good = grounding_check.check_grounding("Le modele passe de t7 a t8.", ctx)
    bad = grounding_check.check_grounding("Le modele passe de t7 a t4.", ctx)
    assert good.ungrounded_transition_claims == [] and good.grounded is True
    assert {"from_phase": "t7", "to_phase": "t4"} in bad.ungrounded_transition_claims


def test_empty_model_change_list_still_flags_a_claimed_model_pair():
    """`all_model_phase_changes == []` is real information ("the predicted
    phase never changes in this band"), not an absence of information."""
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(model_changes=())})
    result = grounding_check.check_grounding("Le modele passe de t7 a t8.", ctx)
    assert {"from_phase": "t7", "to_phase": "t8"} in result.ungrounded_transition_claims
    assert result.grounded is False


# --- A bis. accepting model pairs must NOT weaken skip detection ------------

def test_skip_claimed_on_a_model_pair_that_is_not_a_skip_is_still_flagged():
    """The load-bearing guard. A run of the 2026-09-07c replication answered
    "il y a un saut de phase" about `t7 -> t8` -- a distance-1 change with
    `n_phase_skips = 0`. Accepting model pairs without this check would have
    converted that hallucination into a grounded answer."""
    ctx = _context(dynamic_context={**_current_inference(),
                                    **_series_analysis(model_changes=(("t7", "t8"),), skips=()),
                                    **_observed_events()})
    result = grounding_check.check_grounding(
        "Il y a un saut de phase dans la sequence predite : t7 -> t8.", ctx)
    assert result.skip_claim_mismatches == [
        {"from_phase": "t7", "to_phase": "t8", "real_is_skip": False}]
    assert result.grounded is False


def test_skip_claimed_on_a_real_model_skip_is_grounded():
    ctx = _context(dynamic_context={**_current_inference(),
                                    **_series_analysis(model_changes=(("t5", "t7"),),
                                                       skips=(("t5", "t7"),))})
    result = grounding_check.check_grounding(
        "La sequence predite comporte un saut de phase t5 -> t7.", ctx)
    assert result.skip_claim_mismatches == []
    assert result.grounded is True


# --- B. phase names that exist only as distribution KEYS --------------------

def test_5_phase_present_as_a_phase_probabilities_key_is_recognised():
    """THE fixed false negative: `t8` is the model's second most probable
    phase and appears nowhere as a value; the correct answer to the frozen
    benchmark's Q4 was reported ungrounded 4/4 before this fix."""
    ctx = _context(dynamic_context={**_current_inference(), **_observed_events()})
    result = grounding_check.check_grounding(
        "La deuxieme phase la plus probable est t8.", ctx)
    assert result.ungrounded_phase_tokens == []
    assert result.conflicting_phase_tokens == []
    assert result.grounded is True


def test_6_phase_absent_from_every_distribution_is_not_recognised():
    """A distribution that does not contain the claimed phase must not make
    it grounded -- the recognition is per-distribution, never a blanket
    "any phase name is fine"."""
    ctx = _context(dynamic_context={**_current_inference(
        probabilities={"t7": 0.76, "t6": 0.24}), **_observed_events(pairs=(("t6", "t7"),))})
    result = grounding_check.check_grounding("La deuxieme phase est t8.", ctx)
    assert "t8" in result.ungrounded_phase_tokens
    assert result.grounded is False


def test_7_an_invented_phase_name_is_still_ungrounded():
    ctx = _context(dynamic_context={**_current_inference(), **_observed_events()})
    result = grounding_check.check_grounding("La phase est tSB.", ctx)
    assert "tsb" in result.ungrounded_phase_tokens
    assert result.grounded is False


def test_only_the_top_2_of_a_distribution_count_not_the_whole_posterior():
    """The restriction that keeps the live-conflict check alive. `t4` carries
    p=3.8e-25 in the real posterior: it is in the dict, but it is not a claim
    the model is making. If every key counted, no wrong phase could ever be
    flagged again."""
    ctx = _context(dynamic_context=_current_inference())
    ranked = set()
    grounding_check._phase_distribution_ranked_tokens(ctx["dynamic_context"], ranked)
    assert ranked == {"t7", "t8"}
    for tail_phase in ("t5", "t9+", "tm", "tpb2"):
        assert tail_phase not in ranked


def test_the_c2_live_conflict_case_is_still_caught():
    """Regression guard for docs/RAG_LLM_QUALITY_REPORT.md P0 item 1: a wrong
    phase that is merely enumerated in retrieved documentation must stay
    ungrounded even though it is now also a (tail) key of the posterior."""
    ctx = _context(
        dynamic_context=_current_inference(),
        document_context=[{"content": "Les phases incluent t3, t4, t6, t7, t9+ dans cet ordre."}],
    )
    result = grounding_check.check_grounding("La phase actuelle est t4.", ctx)
    assert result.grounded is False
    assert "t4" in result.ungrounded_phase_tokens
    assert "t4" in result.conflicting_phase_tokens


def test_a_dict_whose_keys_are_not_all_phase_tokens_is_never_promoted():
    """Narrowness guard: only a dict whose keys are ALL phase tokens and whose
    values are ALL numbers qualifies. Anything else is walked normally, so no
    ordinary dict key can become a phase token."""
    ranked = set()
    grounding_check._phase_distribution_ranked_tokens(
        {"metrics": {"t7": 0.9, "brier": 0.1}, "labels": {"t7": "phase courante"},
         "flags": {"t7": True, "t8": False}}, ranked)
    assert ranked == set()


def test_8_numeric_claims_are_still_checked_exactly_as_before():
    """The numeric axis is untouched by this fix: a real value stays grounded,
    an invented one stays flagged, including inside an answer whose phase
    names are now recognised through the distribution keys."""
    ctx = _context(dynamic_context={**_current_inference(), **_observed_events()})
    real = grounding_check.check_grounding(
        "La deuxieme phase est t8, avec une probabilite de 0.23679880956344662.", ctx)
    assert real.ungrounded_numbers == []
    assert real.grounded is True

    invented = grounding_check.check_grounding(
        "La deuxieme phase est t8, avec une probabilite de 0.4242.", ctx)
    assert "0.4242" in invented.ungrounded_numbers
    assert invented.grounded is False


# --- report shape ------------------------------------------------------------

def test_provenance_is_reported_without_changing_the_existing_list_shapes():
    ctx = _context(dynamic_context={**_current_inference(), **_series_analysis(),
                                    **_observed_events()})
    result = grounding_check.check_grounding("t4 -> t6, puis t7 -> t8.", ctx)
    assert result.checked_transition_claims == [
        {"from_phase": "t4", "to_phase": "t6"}, {"from_phase": "t7", "to_phase": "t8"}]
    assert result.transition_claim_provenance == [
        {"from_phase": "t4", "to_phase": "t6", "matches": "observed"},
        {"from_phase": "t7", "to_phase": "t8", "matches": "model"}]
    assert "transition_claim_provenance" in result.to_dict()


def test_no_relational_opinion_without_any_reference_set():
    """Unchanged graceful no-op: neither get_transition_events nor
    series_analysis this turn -> no relational claim is formed."""
    ctx = _context(dynamic_context=_current_inference())
    result = grounding_check.check_grounding("t7 vers t4.", ctx)
    assert result.checked_transition_claims == []
    assert result.transition_claim_provenance == []

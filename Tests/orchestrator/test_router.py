"""Pure router tests -- no filesystem, no torch, no chromadb. router.py
has zero dependencies beyond the stdlib, so these always run."""

from orchestrator.router import (
    DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN, classify, extract_phase_tokens,
)


def test_documentary_definitional_question():
    d = classify("Qu'est-ce que le Semi-HMM ?")
    assert d.category == DOCUMENTARY


def test_dynamic_current_phase_question():
    d = classify("Quelle est la phase actuelle ?")
    assert d.category == DYNAMIC_DATA


def test_hybrid_why_model_predicts_question():
    d = classify("Pourquoi le modele predit-il cette phase ?")
    assert d.category == HYBRID


def test_unknown_out_of_domain_question():
    d = classify("Quel temps fait-il a Brest ?")
    assert d.category == UNKNOWN


def test_route_decision_to_dict_has_all_fields():
    d = classify("Qu'est-ce que le Semi-HMM ?").to_dict()
    assert set(d.keys()) == {
        "question", "category", "hybrid_signal", "dynamic_signal", "documentary_signal",
        "domain_relevant", "matched_hybrid", "matched_dynamic", "matched_documentary",
        "matched_domain",
    }


def test_classify_is_deterministic():
    q = "Pourquoi le modele est-il incertain sur cette prediction ?"
    assert classify(q).category == classify(q).category == HYBRID


def test_classify_is_case_and_accent_insensitive():
    a = classify("Qu'est-ce que le Semi-HMM ?")
    b = classify("QU'EST-CE QUE LE SEMI-HMM ?")
    assert a.category == b.category == DOCUMENTARY


def test_dynamic_and_documentary_signals_together_force_hybrid():
    # "phase actuelle" (dynamic) + "qu'est-ce que" (documentary) in one
    # question, with no explicit HYBRID_PATTERN -- the intersection rule
    # (dynamic_signal AND documentary_signal) must still promote to HYBRID.
    d = classify("Qu'est-ce que la phase actuelle represente ?")
    assert d.dynamic_signal is True
    assert d.documentary_signal is True
    assert d.category == HYBRID


def test_extract_phase_tokens_finds_phase_name():
    assert extract_phase_tokens("Pourquoi t6 correspond-il a cette phase ?") == ["t6"]


def test_extract_phase_tokens_empty_when_no_phase_named():
    assert extract_phase_tokens("Qu'est-ce que le Semi-HMM ?") == []


def test_extract_phase_tokens_handles_tpb2_and_teb():
    tokens = extract_phase_tokens("Pourquoi tPB2 et tEB n'ont-ils pas de duree estimable ?")
    assert "tpb2" in tokens
    assert "teb" in tokens


def test_extract_phase_tokens_handles_t9_plus():
    # Regression: the trailing "+" in the real phase name "t9+" is a
    # non-word character, so a plain trailing `\b` never fires right
    # after it (no word/non-word transition between two non-word chars)
    # -- PHASE_TOKEN_PATTERN previously silently fell back to matching
    # just "t9". Found via Tests/orchestrator/test_grounding_check.py.
    assert extract_phase_tokens("La phase actuelle est t9+.") == ["t9+"]

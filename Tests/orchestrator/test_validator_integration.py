"""The Scientific Validator wired into answer_question() as a control layer.

    QUESTION -> Context Builder -> LLM -> raw answer -> claim extraction ->
    provenance resolution -> scientific retrieval -> validation ->
    qualification -> final answer

Pure: every tool is monkeypatched, the LLM is a fixed-text fake, no network.
The context shapes are the real Reporting-API shapes at the frozen anchor
(model t7 -> t8, annotation t4 -> t6).
"""

import copy
import json

import pytest

from orchestrator import compact_context, scientific_validation, tools
from orchestrator.llm_provider import LLMProvider, LLMResponse
from orchestrator.orchestrator import answer_question, resolve_context_policy
from test_prompt_compact_format import CURRENT_INFERENCE, FACTUAL_DOC, NARRATIVE_DOC, TRANSITION_EVENTS

Q3 = "Quelle était la phase dominante juste avant la transition ?"
Q15 = "Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division chaotique ?"
Q11 = "Cette transition est-elle compatible avec l'ordre attendu du développement embryonnaire ?"


class FixedLLM(LLMProvider):
    """Returns exactly the text it was built with -- the answer under test."""
    name = "fixed"

    def __init__(self, text, context_format=None):
        self.text = text
        if context_format is not None:
            self.context_format = context_format

    def generate(self, context):
        return LLMResponse(text=self.text, provider=self.name, grounded=False)


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setattr(tools, "get_current_inference",
                        lambda video_id, window, split="val": copy.deepcopy(CURRENT_INFERENCE))
    monkeypatch.setattr(tools, "get_transition_events",
                        lambda video_id, split="val", window=None: copy.deepcopy(TRANSITION_EVENTS))
    candidates = [copy.deepcopy(NARRATIVE_DOC)] * 3 + [copy.deepcopy(FACTUAL_DOC)] * 4
    monkeypatch.setattr(tools, "retrieve_documents",
                        lambda query, top_k=5, **kw: [copy.deepcopy(c) for c in candidates[:top_k]])
    monkeypatch.delenv(scientific_validation.VALIDATOR_ENV_VAR, raising=False)
    yield


def _ask(text, question=Q3, **kw):
    return answer_question(question, video_id="Patient_319", window=156, split="val",
                           llm=FixedLLM(text), **kw)


# ---------------------------------------------------------------------------
# Q3 -- the critical case
# ---------------------------------------------------------------------------

def test_q3_prediction_stated_as_observation_is_qualified_not_rewritten():
    raw = "L'embryon était en t7."
    out = _ask(raw)
    validation = out["validation"]
    assert validation["enabled"] is True
    assert out["response"]["text"] == raw                       # raw answer untouched
    assert out["final_answer"].startswith(raw)                  # never rewritten
    assert "QUALIFICATION SCIENTIFIQUE" in out["final_answer"]
    claim = validation["report"]["claims"][0]
    assert claim["claim"] == raw
    assert claim["status"] == "NOT_SUPPORTED"
    assert "R2_prediction_is_not_observation" in claim["rules_fired"]
    assert "PREDICTION" in claim["qualification"]
    assert validation["provenance_valid"] is False
    assert validation["provenance_violations"][0]["rule"] == "R2_prediction_is_not_observation"


def test_q3_validator_never_substitutes_the_annotation_for_the_prediction():
    """The context DOES hold ground_truth_phase = t4, and the validator must
    still not write it: it corrects the provenance, not the answer."""
    raw = "L'embryon était en t7."
    out = _ask(raw)
    appended = out["final_answer"][len(raw):]
    assert "t4" not in appended
    assert "t4" not in json.dumps(out["validation"]["report"], ensure_ascii=False)
    assert "t7" in appended                                     # the provenance of t7 is named


def test_q3_correctly_framed_prediction_is_supported_and_final_equals_raw():
    raw = "Le modèle prédit t7 pour cette fenêtre ; l'annotation dit t4."
    out = _ask(raw)
    assert out["validation"]["provenance_valid"] is True
    assert out["validation"]["claim_validation"]["has_violation"] is False
    assert out["final_answer"] == raw
    assert out["validation"]["qualification_appended"] is False


# ---------------------------------------------------------------------------
# Q15 -- absent definition
# ---------------------------------------------------------------------------

def test_q15_reverse_cleavage_asserted_from_a_phase_return_is_not_assessable():
    raw = "Il s'agit d'un reverse cleavage, car la phase revient en arrière."
    out = _ask(raw, question=Q15)
    claim = out["validation"]["report"]["claims"][0]
    assert claim["status"] == "NOT_ASSESSABLE"
    assert claim["reason_code"] == "external_definition"
    assert "R8_definition_absent_from_corpus" in claim["rules_fired"]
    assert out["validation"]["claim_validation"]["has_violation"] is False   # NOT_ASSESSABLE is not a violation
    assert out["final_answer"].startswith(raw)
    assert "definition" in out["final_answer"].lower()


# ---------------------------------------------------------------------------
# switch OFF / ON, environment, failure isolation
# ---------------------------------------------------------------------------

def test_validator_off_returns_the_same_structure_with_raw_final_answer():
    raw = "L'embryon était en t7."
    out = _ask(raw, validate=False)
    assert out["validation"]["enabled"] is False
    assert out["validation"]["report"] is None
    assert out["validation"]["provenance_valid"] is None
    assert out["final_answer"] == raw
    assert out["context_policy"]["validator_enabled"] is False


def test_validator_off_and_on_share_byte_identical_raw_answer_and_grounding():
    raw = "L'embryon était en t7."
    off, on = _ask(raw, validate=False), _ask(raw, validate=True)
    assert off["response"] == on["response"]
    assert off["response"]["grounded"] == on["response"]["grounded"]
    assert off["context"]["dynamic_context"] == on["context"]["dynamic_context"]


def test_environment_variable_switch(monkeypatch):
    monkeypatch.setenv(scientific_validation.VALIDATOR_ENV_VAR, "off")
    assert _ask("L'embryon était en t7.")["validation"]["enabled"] is False
    monkeypatch.setenv(scientific_validation.VALIDATOR_ENV_VAR, "on")
    assert _ask("L'embryon était en t7.")["validation"]["enabled"] is True
    monkeypatch.setenv(scientific_validation.VALIDATOR_ENV_VAR, "maybe")
    with pytest.raises(ValueError):
        scientific_validation.resolve_validator_enabled()
    assert scientific_validation.resolve_validator_enabled(False) is False   # explicit wins


def test_default_is_on():
    assert scientific_validation.resolve_validator_enabled() is True


def test_a_validator_failure_never_crashes_the_turn_and_keeps_the_raw_answer(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic validator failure")

    monkeypatch.setattr(scientific_validation, "validate_answer", boom)
    raw = "L'embryon était en t7."
    out = _ask(raw)
    assert out["final_answer"] == raw
    assert out["validation"]["enabled"] is True
    assert "synthetic validator failure" in out["validation"]["error"]
    assert out["validation"]["report"] is None


def test_grounded_keeps_its_presence_meaning_whatever_the_validator_says():
    """t7 is present in the context, so the presence check passes -- and the
    validator still rejects the framing. The two verdicts are separate keys."""
    out = _ask("La phase dominante juste avant la transition était t7.")
    assert out["response"]["grounded"] is True
    assert out["validation"]["provenance_valid"] is False


def test_result_is_json_serializable_end_to_end():
    out = _ask("L'embryon était en t7.")
    json.dumps(out, ensure_ascii=False, default=str)
    for key in ("question", "route", "tool_plan", "context_policy", "context", "response",
                "validation", "final_answer"):
        assert key in out


def test_null_provider_default_path_still_works():
    out = answer_question(Q3, video_id="Patient_319", window=156)
    assert out["response"]["provider"] == "null"
    assert "validation" in out and out["final_answer"].startswith(out["response"]["text"].rstrip())


# ---------------------------------------------------------------------------
# context policy: compact_v2 adds curated retrieval + scientific reference
# ---------------------------------------------------------------------------

def test_v2_policy_context_is_unchanged_by_the_new_keys():
    out = _ask("x", question=Q11)
    ctx = out["context"]
    assert out["context_policy"]["context_format"] == "v2"
    assert ctx["scientific_context"] is None and ctx["retrieval_audit"] is None
    assert "transition_context" not in ctx["dynamic_context"]
    assert len(ctx["document_context"]) == 5          # plain top-5, narrative included
    assert ctx["document_context"][0]["source"] == "docs/RESEARCH_BLUEPRINT.md"


def test_compact_v2_policy_curates_retrieval_and_adds_scientific_and_transition_context():
    out = answer_question(Q11, video_id="Patient_319", window=156, split="val",
                          llm=FixedLLM("x", context_format="compact_v2"))
    ctx = out["context"]
    assert out["context_policy"] == {"context_format": "compact_v2", "curated_retrieval": True,
                                     "scientific_reference": True, "validator_enabled": True}
    assert [c["source"] for c in ctx["document_context"]] == ["docs/SCIENTIFIC_REPORT.md"] * 4
    assert [a["decision"] for a in ctx["retrieval_audit"]] == ["skipped"] * 3 + ["kept"] * 4
    assert ctx["scientific_context"] is not None
    assert ctx["scientific_context"]["source"] == "ISTANBUL_CONSENSUS_2025"
    # the fixture carries a single (bracketing) transition: the DERIVED
    # selection would only repeat it, so it is absent -- never a redundant block
    assert "transition_context" not in ctx["dynamic_context"]


def test_compact_v2_candidate_pool_is_larger_than_the_rendered_set(monkeypatch):
    seen = {}

    def fake_retrieve(query, top_k=5, **kw):
        seen["top_k"] = top_k
        return [copy.deepcopy(FACTUAL_DOC)] * top_k

    monkeypatch.setattr(tools, "retrieve_documents", fake_retrieve)
    answer_question(Q11, video_id="Patient_319", window=156, llm=FixedLLM("x", context_format="compact_v2"))
    assert seen["top_k"] == compact_context.DEFAULT_RETRIEVAL_CANDIDATE_POOL


def test_explicit_context_format_must_agree_with_the_provider():
    with pytest.raises(ValueError):
        resolve_context_policy("v2", FixedLLM("x", context_format="compact_v2"))
    assert resolve_context_policy("compact_v2", FixedLLM("x", context_format="compact_v2")) == "compact_v2"
    assert resolve_context_policy(None, FixedLLM("x")) == "v2"
    assert resolve_context_policy("compact_v2", FixedLLM("x")) == "compact_v2"


def test_scientific_context_counts_as_presence_for_grounding_and_as_scientific_for_the_validator():
    """A block id quoted from the [SCIENTIFIC] block is neither an invented
    token nor an embryo measurement."""
    ctx_out = answer_question(Q11, video_id="Patient_319", window=156,
                              llm=FixedLLM("Selon le Consensus (IC2025-CP1-02), la variabilité "
                                           "biologique est inhérente.", context_format="compact_v2"))
    if not ctx_out["context"]["scientific_context"]["available"]:
        pytest.skip("Istanbul corpus not on this machine")
    assert ctx_out["response"]["grounded"] is True
    assert "SCIENTIFIC" in ctx_out["validation"]["report"]["available_provenance"]

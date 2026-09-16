"""
Phase 6 end-to-end tests -- Grounded Answer Contract, citations,
hallucination boundary, timeout/failure handling. NO API key required
anywhere in this file (task sec 10's own requirement): every provider
used here is either `TemplateLLMProvider`/`NullLLMProvider` (zero
network) or a small hand-written fake `LLMProvider` subclass built for
one specific test. Real Reporting API/RAG calls are monkeypatched, same
convention as Tests/orchestrator/test_orchestrator.py.
"""

import time

import pytest

from orchestrator import tools
from orchestrator.llm_provider import LLMProvider, LLMResponse, NullLLMProvider, TemplateLLMProvider
from orchestrator.orchestrator import answer_question
from orchestrator.router import DOCUMENTARY, DYNAMIC_DATA, HYBRID, UNKNOWN


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    def fake_retrieve_documents(query, top_k=5, **kwargs):
        return [{
            "chunk_id": "doc::0001", "content": "Le Semi-HMM ajoute un modele de duree explicite.",
            "distance": 0.1, "score": 0.9,
            "metadata": {"source": "docs/HANDOFF_SEMI_HMM.md", "section": "1", "status": "authoritative",
                         "authoritative": True},
        }]

    def fake_get_current_inference(video_id, window, split="val"):
        return {
            "sample": {"sample_id": f"{video_id}#{split}"},
            "current_state": {"current_phase": "t6", "phase_probability": 0.42},
            "model": {"model_version": "fake_v1"},
        }

    monkeypatch.setattr(tools, "retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(tools, "get_current_inference", fake_get_current_inference)


# --------------------------------------------------------------------------
# Structured output / Grounded Answer Contract shape
# --------------------------------------------------------------------------

def test_response_has_the_full_grounded_answer_contract_shape():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=TemplateLLMProvider())
    response = out["response"]
    for key in ("text", "answer", "provider", "grounded", "warnings", "sources",
                "dynamic_data", "used_tools", "confidence"):
        assert key in response


def test_confidence_is_always_a_qualitative_label_never_a_number():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=TemplateLLMProvider())
    assert out["response"]["confidence"] in ("high", "medium", "low", "unknown")


# --------------------------------------------------------------------------
# Category coverage: DOCUMENTARY / DYNAMIC_DATA / HYBRID / UNKNOWN
# --------------------------------------------------------------------------

def test_documentary_answer_cites_a_document_source():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=TemplateLLMProvider())
    assert out["route"]["category"] == DOCUMENTARY
    assert out["response"]["sources"]
    assert out["response"]["sources"][0]["source"] == "docs/HANDOFF_SEMI_HMM.md"
    assert out["response"]["grounded"] is True


def test_dynamic_data_answer_restates_the_live_value():
    out = answer_question("Quelle est la phase actuelle ?", llm=TemplateLLMProvider(),
                           video_id="Patient_1", window=0)
    assert out["route"]["category"] == DYNAMIC_DATA
    assert "t6" in out["response"]["text"]
    assert out["response"]["dynamic_data"]
    assert out["response"]["grounded"] is True


def test_hybrid_answer_contains_both_document_and_dynamic_content():
    out = answer_question("Pourquoi le modele predit-il t6 pour cette fenetre ?", llm=TemplateLLMProvider(),
                           video_id="Patient_1", window=0)
    assert out["route"]["category"] == HYBRID
    assert out["response"]["sources"]
    assert out["response"]["dynamic_data"]


def test_unknown_question_never_produces_a_fabricated_answer():
    out = answer_question("Quel temps fait-il a Brest ?", llm=TemplateLLMProvider())
    assert out["route"]["category"] == UNKNOWN
    assert "ne dispose pas de suffisamment d'informations" in out["response"]["text"]


# --------------------------------------------------------------------------
# No context / contradictory context
# --------------------------------------------------------------------------

def test_no_context_produces_the_canonical_no_information_message(monkeypatch):
    def empty_retrieve(query, top_k=5, **kwargs):
        return []

    monkeypatch.setattr(tools, "retrieve_documents", empty_retrieve)
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=TemplateLLMProvider())
    assert "ne dispose pas de suffisamment d'informations" in out["response"]["text"]
    assert out["response"]["confidence"] == "unknown"


def test_contradictory_context_is_presented_not_silently_reconciled(monkeypatch):
    # Document says one number, dynamic data says another -- the template
    # provider must present both, verbatim, with their own provenance,
    # never invent a reconciled third value.
    def contradictory_retrieve(query, top_k=5, **kwargs):
        return [{
            "chunk_id": "doc::0001", "content": "L'analyse historique donne P(t6)=0.30 sur Val.",
            "distance": 0.1, "score": 0.9,
            "metadata": {"source": "docs/SCIENTIFIC_REPORT.md", "section": "16", "status": "authoritative",
                         "authoritative": True},
        }]

    def contradictory_inference(video_id, window, split="val"):
        return {"sample": {"sample_id": f"{video_id}#{split}"},
                "current_state": {"current_phase": "t6", "phase_probability": 0.72},
                "model": {"model_version": "fake_v1"}}

    monkeypatch.setattr(tools, "retrieve_documents", contradictory_retrieve)
    monkeypatch.setattr(tools, "get_current_inference", contradictory_inference)

    out = answer_question("Pourquoi le modele predit-il t6 pour cette fenetre ?", llm=TemplateLLMProvider(),
                           video_id="Patient_1", window=0)
    text = out["response"]["text"]
    assert "0.30" in text
    assert "0.72" in text
    assert out["response"]["grounded"] is True  # both numbers are real, from their own sources


# --------------------------------------------------------------------------
# Hallucination boundary -- a provider that fabricates a number is caught
# --------------------------------------------------------------------------

class _LyingProvider(LLMProvider):
    name = "liar"

    def generate(self, context):
        return LLMResponse(text="La probabilite est de 0.999, totalement fiable.",
                            provider=self.name, grounded=True)  # claims grounded=True, falsely


def test_hallucinated_number_is_caught_even_when_provider_claims_grounded():
    out = answer_question("Quelle est la phase actuelle ?", llm=_LyingProvider(),
                           video_id="Patient_1", window=0)
    assert out["response"]["grounded"] is False
    assert any("ungrounded" in w.lower() for w in out["response"]["warnings"])


class _HonestNumberProvider(LLMProvider):
    name = "honest"

    def generate(self, context):
        value = context["dynamic_context"]["get_current_inference"]["current_state"]["phase_probability"]
        return LLMResponse(text=f"La probabilite est de {value}.", provider=self.name, grounded=False)


def test_a_real_grounded_number_is_confirmed_even_if_provider_underclaims():
    out = answer_question("Quelle est la phase actuelle ?", llm=_HonestNumberProvider(),
                           video_id="Patient_1", window=0)
    assert out["response"]["grounded"] is True  # independent check upgrades a correct-but-underclaimed answer


# --------------------------------------------------------------------------
# Timeout / provider failure
# --------------------------------------------------------------------------

class _SlowProvider(LLMProvider):
    name = "slow"

    def generate(self, context):
        time.sleep(2)
        return LLMResponse(text="too late", provider=self.name, grounded=True)


def test_slow_provider_times_out_and_falls_back_honestly():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=_SlowProvider(), llm_timeout_seconds=0.2)
    assert "timed out" in out["response"]["text"]
    assert out["response"]["grounded"] is False
    assert any("provider failure" in w.lower() for w in out["response"]["warnings"])


class _CrashingProvider(LLMProvider):
    name = "crash"

    def generate(self, context):
        raise RuntimeError("simulated provider crash")


def test_crashing_provider_never_raises_out_of_answer_question():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=_CrashingProvider())  # must not raise
    assert "RuntimeError" in out["response"]["text"]
    assert out["response"]["grounded"] is False


# --------------------------------------------------------------------------
# NullLLMProvider (default, no config)
# --------------------------------------------------------------------------

def test_null_provider_remains_the_default_and_never_fabricates():
    out = answer_question("Qu'est-ce que le Semi-HMM ?")  # no llm= passed
    assert out["response"]["provider"] == "null"
    assert out["response"]["grounded"] is False


def test_null_provider_still_exposes_sources_and_dynamic_data():
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=NullLLMProvider())
    assert out["response"]["sources"]


# --------------------------------------------------------------------------
# End-to-end pipeline shape (task sec 14: QUESTION -> ROUTER -> TOOLS ->
# CONTEXT BUILDER -> LLM -> GROUNDED ANSWER -> SOURCES + WARNINGS)
# --------------------------------------------------------------------------

def test_full_pipeline_result_is_json_serializable_end_to_end():
    import json

    out = answer_question("Pourquoi le modele predit-il t6 pour cette fenetre ?", llm=TemplateLLMProvider(),
                           video_id="Patient_1", window=0)
    json.dumps(out)  # must not raise
    assert out["question"]
    assert out["route"]["category"]
    assert out["tool_plan"]["called"]
    assert out["context"]["document_context"] or out["context"]["dynamic_context"]
    assert out["response"]["text"]


def test_scientific_models_never_touched_by_this_module():
    import orchestrator.grounding_check as gc
    import orchestrator.llm_provider as lp
    import orchestrator.prompt as pr

    for mod in (gc, lp, pr):
        assert not hasattr(mod, "SemiHMMModel")
        src = mod.__file__
        with open(src, encoding="utf-8") as f:
            content = f.read()
        assert "evaluation.models" not in content

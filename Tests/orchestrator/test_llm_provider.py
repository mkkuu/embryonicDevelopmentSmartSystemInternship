"""
Pure tests for llm_provider.py -- no network call anywhere in this
module by construction, so these always run. Verifies the task's own
explicit requirement: "si aucun LLM n'est configure, les tests doivent
fonctionner."
"""

import pytest

from orchestrator.context_builder import build_context
from orchestrator.llm_provider import LLMProvider, NullLLMProvider
from orchestrator.router import classify


def test_null_llm_provider_never_raises_and_returns_a_response():
    ctx = build_context("Qu'est-ce que le Semi-HMM ?", classify("Qu'est-ce que le Semi-HMM ?"), {})
    response = NullLLMProvider().generate(ctx)
    assert response.provider == "null"
    assert response.text


def test_null_llm_provider_is_never_grounded():
    ctx = build_context("q", classify("q"), {})
    assert NullLLMProvider().generate(ctx).grounded is False


def test_null_llm_provider_reports_zero_results_honestly():
    ctx = build_context("q", classify("q"), {})
    response = NullLLMProvider().generate(ctx)
    assert "0 document chunk(s)" in response.text
    assert "0 live data result(s)" in response.text


def test_null_llm_provider_warns_it_never_synthesizes():
    ctx = build_context("q", classify("q"), {})
    response = NullLLMProvider().generate(ctx)
    assert any("never synthesizes" in w for w in response.warnings)


def test_llm_provider_is_abstract_and_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMProvider()  # abstract -- generate() has no implementation


def test_a_concrete_subclass_can_override_generate():
    class EchoProvider(LLMProvider):
        name = "echo"

        def generate(self, context):
            from orchestrator.llm_provider import LLMResponse

            return LLMResponse(text=f"echo: {context['question']}", provider=self.name, grounded=False)

    ctx = build_context("hello", classify("hello"), {})
    response = EchoProvider().generate(ctx)
    assert response.text == "echo: hello"
    assert response.provider == "echo"

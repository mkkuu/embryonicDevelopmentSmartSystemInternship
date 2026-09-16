"""orchestrator/llm_config.py -- the single resolver of which generative
model answers (Web App and benchmark alike). Pure: no Ollama, no network;
constructing OllamaLLMProvider stores configuration only."""

import pytest

from orchestrator import llm_config
from orchestrator.llm_provider import DEFAULT_NUM_CTX, OllamaLLMProvider


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(llm_config.LLM_MODEL_ENV_VAR, raising=False)
    monkeypatch.delenv(llm_config.LLM_TIMEOUT_ENV_VAR, raising=False)
    monkeypatch.delenv("EMBRYO_OLLAMA_NUM_CTX", raising=False)


# --- model ---------------------------------------------------------------

def test_default_model_is_the_current_production_model_not_a_candidate():
    """Swapping production is a decision that follows a benchmark comparison;
    the resolver must not pre-empt it."""
    assert llm_config.DEFAULT_LLM_MODEL == "llama3.2:latest"
    assert llm_config.resolve_llm_model() == "llama3.2:latest"


def test_env_var_selects_the_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mistral-nemo:12b")
    assert llm_config.resolve_llm_model() == "mistral-nemo:12b"


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mistral-nemo:12b")
    assert llm_config.resolve_llm_model("qwen3:8b") == "qwen3:8b"


def test_blank_values_are_treated_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "   ")
    assert llm_config.resolve_llm_model("") == "llama3.2:latest"
    assert llm_config.resolve_llm_model() == "llama3.2:latest"


def test_model_value_is_stripped(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  gemma3:12b \n")
    assert llm_config.resolve_llm_model() == "gemma3:12b"


# --- timeout -------------------------------------------------------------

def test_default_webapp_timeout_covers_the_measured_benchmark_maximum():
    # 33.1 s was the slowest valid answer (mistral-nemo:12b, 2026-09-02d);
    # the orchestrator's own 30 s default would have cut it.
    assert llm_config.resolve_llm_timeout_seconds() == 120.0
    assert llm_config.resolve_llm_timeout_seconds() > 33.1


def test_timeout_env_var_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")
    assert llm_config.resolve_llm_timeout_seconds() == 45.0
    assert llm_config.resolve_llm_timeout_seconds(240) == 240.0


def test_timeout_default_can_be_overridden_by_the_caller(monkeypatch):
    assert llm_config.resolve_llm_timeout_seconds(default=240.0) == 240.0


@pytest.mark.parametrize("bad", ["abc", "0", "-5", ""])
def test_invalid_timeout_env_fails_loudly_or_falls_back(monkeypatch, bad):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", bad)
    if bad == "":
        assert llm_config.resolve_llm_timeout_seconds() == 120.0
    else:
        with pytest.raises(ValueError):
            llm_config.resolve_llm_timeout_seconds()


def test_invalid_explicit_timeout_raises():
    with pytest.raises(ValueError):
        llm_config.resolve_llm_timeout_seconds(0)
    with pytest.raises(ValueError):
        llm_config.resolve_llm_timeout_seconds(-1)


# --- provider construction ----------------------------------------------

def test_build_ollama_provider_uses_the_resolved_model_and_timeout(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mistral-nemo:12b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")
    provider = llm_config.build_ollama_provider()
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model == "mistral-nemo:12b"
    assert provider.request_timeout_seconds == 90.0
    # Nothing but the generative model changes: sampling stays Ollama's own
    # default and the context window stays the documented one.
    assert provider.seed is None and provider.temperature is None
    assert provider.num_ctx == DEFAULT_NUM_CTX


def test_build_ollama_provider_passes_experiment_kwargs_through():
    provider = llm_config.build_ollama_provider("qwen3:8b", 240, seed=7, num_ctx=8192)
    assert provider.model == "qwen3:8b"
    assert provider.request_timeout_seconds == 240.0
    assert provider.seed == 7 and provider.num_ctx == 8192


def test_describe_provider_records_every_comparability_parameter():
    provider = llm_config.build_ollama_provider("mistral-nemo:12b", 240)
    description = llm_config.describe_provider(provider)
    assert description == {
        "provider": "ollama", "model": "mistral-nemo:12b", "base_url": "http://localhost:11434",
        "request_timeout_seconds": 240.0, "num_ctx": DEFAULT_NUM_CTX, "seed": None, "temperature": None,
        # Added 2026-09-11 with COMPACT_CONTEXT_V1: which RENDERING of the
        # context was sent is as answer-affecting as num_ctx, so an artefact
        # that does not record it is not comparable with one that does.
        "context_format": "v2",
    }


def test_webapp_uses_the_shared_resolver_not_a_hardcoded_model():
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent.parent / "Training" / "webapp_api" / "app.py").read_text(
        encoding="utf-8")
    assert "llm_config.build_ollama_provider()" in source
    assert 'OllamaLLMProvider(model="' not in source, "the Web App must not hardcode a model name"


def test_benchmark_runner_defaults_to_the_shared_resolver():
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent.parent / "Training" / "orchestrator"
              / "run_fixed_question_benchmark.py").read_text(encoding="utf-8")
    assert "llm_config.build_ollama_provider(args.model, args.timeout)" in source, (
        "the runner must build its provider through the shared resolver (--model > $LLM_MODEL > default)"
    )
    assert 'default="llama3.2:latest"' not in source, "no second hardcoded default model"
    assert "describe_provider" in source, "artefacts must record num_ctx/seed/temperature for comparability"

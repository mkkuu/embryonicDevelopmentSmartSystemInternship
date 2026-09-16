"""
Single place where "which generative LLM answers, and with what timeout"
is decided -- for BOTH the Web App (`Training/webapp_api/app.py`) and the
fixed-question benchmark runner
(`Training/orchestrator/run_fixed_question_benchmark.py`).

    Web App / benchmark
        -> build_ollama_provider()          (this module)
            -> OllamaLLMProvider(model=...) (llm_provider.py, unchanged)
                -> the selected model

instead of a model name hardcoded at each call site. The selection
changes NOTHING but the generative model: Router, tools, context builder,
RAG retrieval, prompt contract, grounding check and the scientific models
are untouched by anything here.

Resolution order (highest wins):
    explicit argument  >  environment variable  >  documented default

    LLM_MODEL            model tag as known to the local Ollama server
                         (default: llama3.2:latest -- the model that has
                         answered /chat since Phase 7; NOT changed here.
                         Swapping the production model is a product decision
                         that must follow a comparison on the frozen
                         Q1-Q15 benchmark, docs/RAG_FIXED_QUESTION_BENCHMARK.md).
    LLM_TIMEOUT_SECONDS  per-question wall-clock budget for the Web App
                         (default: 120). The benchmark keeps its own CLI
                         `--timeout` (240 on the last valid runs) so that a
                         benchmark artefact never silently inherits a Web App
                         setting -- both values are recorded in the artefact.

Nothing here contacts Ollama: constructing a provider stores configuration
only (`OllamaLLMProvider.__init__`), so an unreachable server or an
un-pulled model surfaces at the first real request, through the
orchestrator's existing honest-fallback path, never as an import-time
crash of the Web App.

`num_ctx`/`seed`/`temperature` are deliberately NOT exposed as environment
variables here beyond what `llm_provider.py` already offers
(`EMBRYO_OLLAMA_NUM_CTX`): a model comparison must be a single-variable
experiment, and the benchmark protocol forbids fixing seed/temperature
"to stabilise" (docs/PROJECT_CHECKPOINT.md, 2026-09-03b).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .llm_provider import OllamaLLMProvider

LLM_MODEL_ENV_VAR = "LLM_MODEL"
DEFAULT_LLM_MODEL = "llama3.2:latest"

LLM_TIMEOUT_ENV_VAR = "LLM_TIMEOUT_SECONDS"
# Why 120 and not the orchestrator's 30: on the last valid benchmark runs the
# slowest single question took 33.1 s (mistral-nemo:12b) / 20.7 s
# (llama3.2:latest) on GPU0 -- a 30 s budget would cut real answers off in
# the UI. 120 keeps a wide margin without approving CPU-offloaded models.
DEFAULT_WEBAPP_LLM_TIMEOUT_SECONDS = 120.0


def resolve_llm_model(explicit: Optional[str] = None) -> str:
    """explicit > $LLM_MODEL > DEFAULT_LLM_MODEL. A blank value is treated as
    unset rather than silently selecting a model named ''."""
    if explicit is not None and explicit.strip() != "":
        return explicit.strip()
    raw = os.environ.get(LLM_MODEL_ENV_VAR)
    if raw is not None and raw.strip() != "":
        return raw.strip()
    return DEFAULT_LLM_MODEL


def resolve_llm_timeout_seconds(explicit: Optional[float] = None,
                                default: float = DEFAULT_WEBAPP_LLM_TIMEOUT_SECONDS) -> float:
    """explicit > $LLM_TIMEOUT_SECONDS > `default`. Fails loudly on a value
    that is not a positive number -- a silently-wrong timeout is exactly the
    kind of hidden setting this module exists to end."""
    if explicit is not None:
        value = float(explicit)
        if isinstance(explicit, bool) or value <= 0:
            raise ValueError(f"LLM timeout must be a positive number of seconds, got {explicit!r}.")
        return value
    raw = os.environ.get(LLM_TIMEOUT_ENV_VAR)
    if raw is not None and raw.strip() != "":
        try:
            value = float(raw)
        except ValueError as e:
            raise ValueError(f"{LLM_TIMEOUT_ENV_VAR}={raw!r} is not a number.") from e
        if value <= 0:
            raise ValueError(f"{LLM_TIMEOUT_ENV_VAR}={raw!r} must be positive.")
        return value
    return float(default)


def build_ollama_provider(model: Optional[str] = None,
                          timeout_seconds: Optional[float] = None,
                          default_timeout_seconds: float = DEFAULT_WEBAPP_LLM_TIMEOUT_SECONDS,
                          **provider_kwargs: Any) -> OllamaLLMProvider:
    """The one constructor every caller should use. `provider_kwargs` are
    passed through untouched to `OllamaLLMProvider` (seed/temperature/
    num_ctx/base_url) for scripted experiments; the Web App passes none."""
    return OllamaLLMProvider(
        model=resolve_llm_model(model),
        request_timeout_seconds=resolve_llm_timeout_seconds(timeout_seconds, default_timeout_seconds),
        **provider_kwargs,
    )


def describe_provider(provider: OllamaLLMProvider) -> Dict[str, Any]:
    """Every generation parameter a benchmark artefact must carry to be
    comparable with another run (docs/PROJECT_CHECKPOINT.md 2026-09-03b,
    'Lacune de traçabilité': model, num_ctx, seed, temperature and the
    timeout were previously reconstructed from code and environment, not
    stored). Pure -- reads the provider object only, no network call."""
    return {
        "provider": provider.name,
        "model": provider.model,
        "base_url": provider.base_url,
        "request_timeout_seconds": provider.request_timeout_seconds,
        "num_ctx": provider.num_ctx,
        "seed": provider.seed,
        "temperature": provider.temperature,
        # Which context RENDERING was sent (prompt.CONTEXT_FORMAT_*). Recorded
        # like every other answer-affecting parameter: two artefacts that
        # differ only here are a context-format A/B, and one that does not say
        # is not comparable with one that does.
        "context_format": getattr(provider, "context_format", None),
    }

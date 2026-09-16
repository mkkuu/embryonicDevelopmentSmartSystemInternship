"""
Abstract LLM interface (task's own "Interface abstraite LLM", Phase 11)
plus the Grounded Answer Contract (Phase 6 sec 4). No EXTERNAL LLM API is
called anywhere in this repository -- see docs/LLM_ORCHESTRATION.md's
"planning only" framing and docs/LLM_PROVIDER_DECISION.md (which real
provider, if any, is authorized to receive project data -- a Decision
Required, still open as of this module).

LLMProvider exists so a future real provider is a drop-in swap, never a
rewrite of the orchestrator around it -- docs/LLM_ORCHESTRATION.md sec 6
(Offline/Privacy) explicitly leaves the concrete choice (external API vs.
local vs. hybrid) as a Decision Required; this abstraction is what makes
that decision deferrable without blocking everything built on top of it.

Three providers are registered today:

- NullLLMProvider: the default. "si aucun LLM n'est configure, les tests
  doivent fonctionner" (task's own requirement). Never synthesizes
  prose -- reports what was retrieved instead of ever fabricating an
  answer. Zero network access, zero API key.
- TemplateLLMProvider: Phase 6's initial "first provider" (task sec 7).
  Deterministic, rule-based synthesis -- restates CONTEXT content
  verbatim into a fixed template, never invents a word beyond it. Proves
  the full Grounded Answer Contract (sources/dynamic_data/used_tools/
  confidence/citations/hallucination-boundary) works end to end without
  crossing the unresolved external-data-transmission boundary. NOT a
  substitute for a real LLM's reasoning/synthesis quality -- see
  docs/LLM_PROVIDER_DECISION.md and docs/LLM_EVALUATION.md for what this
  does and does not demonstrate.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import grounding_check, prompt


class OllamaUnavailableError(Exception):
    """The local Ollama server (or the requested model) could not be
    reached -- e.g. `ollama serve` isn't running, or `ollama pull
    <model>` was never run. Caught by orchestrator.generate_grounded_response()'s
    existing broad exception handler (same fallback path as a timeout or
    any other provider failure) -- never a crash of the whole turn."""


@dataclass
class LLMResponse:
    """The Grounded Answer Contract (Phase 6 sec 4), adapted to the real
    code: `answer`/`sources`/`dynamic_data`/`warnings`/`confidence`/
    `used_tools` from the task's conceptual JSON shape, plus `provider`/
    `grounded`/`text` (kept as the pre-existing field name for backward
    compatibility with every test/caller written before this phase --
    `text` and `answer` are intentionally the same string, never two
    different truths)."""

    text: str
    provider: str
    grounded: bool  # True only when a post-hoc check (grounding_check.py) confirms every
    # number/phase-name in `text` traces to this turn's own context -- NEVER just the
    # provider's own self-report; orchestrator.answer_question() re-verifies this for
    # every provider, including ones that claim True themselves.
    warnings: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)  # citation-ready, from
    # context["document_context"] -- document/section/status/authoritative per entry
    dynamic_data: List[Dict[str, Any]] = field(default_factory=list)  # from
    # context["dynamic_context"] -- tool name + raw Reporting API output + model_version
    used_tools: List[str] = field(default_factory=list)
    confidence: str = "unknown"  # "high" | "medium" | "low" | "unknown" -- QUALITATIVE
    # only, NEVER a fabricated numeric probability (see grounding_check.estimate_confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "answer": self.text, "provider": self.provider,
            "grounded": self.grounded, "warnings": self.warnings, "sources": self.sources,
            "dynamic_data": self.dynamic_data, "used_tools": self.used_tools,
            "confidence": self.confidence,
        }


class LLMProvider(abc.ABC):
    """generate(context) -> LLMResponse. `context` is exactly
    context_builder.build_context()'s output dict -- never a raw string
    prompt assembled ad hoc, so that FACT/DOCUMENTED provenance
    (docs/LLM_ORCHESTRATION.md sec 3-4) survives into whatever the
    concrete provider does with it. A concrete provider MAY internally
    render `context` into a prompt string however it likes; it MUST NOT
    receive anything less structured than this dict as its only input."""

    name: str = "abstract"

    # True for a provider that structurally never synthesizes an answer
    # (NullLLMProvider) -- a declared property, not something
    # orchestrator.generate_grounded_response() should try to infer from
    # the text (its meta-description text mentions counts like "1
    # document chunk(s)", and a naive "no ungrounded numbers found in the
    # text" check would otherwise vacuously upgrade THAT to grounded=True,
    # contradicting "NullLLMProvider is never grounded" -- found via a
    # real test, Tests/orchestrator/test_grounded_generation.py).
    always_ungrounded: bool = False

    @abc.abstractmethod
    def generate(self, context: Dict[str, Any]) -> LLMResponse:
        raise NotImplementedError


class NullLLMProvider(LLMProvider):
    """No network call, no API key, always available, fully
    deterministic. Reports what was retrieved instead of ever writing a
    natural-language answer -- this is deliberate, not a stub to be
    embarrassed about: it is what "no LLM configured" must mean under
    docs/LLM_PROMPT_CONTRACT.md's guardrails (never invent a number,
    never invent prose that looks like a grounded answer when nothing
    was actually synthesized)."""

    name = "null"
    always_ungrounded = True

    def generate(self, context: Dict[str, Any]) -> LLMResponse:
        n_docs = len(context.get("document_context", []) or [])
        n_dynamic = len(context.get("dynamic_context", {}) or {})
        question = context.get("question", "")
        category = (context.get("route") or {}).get("category", "UNKNOWN")

        parts = [
            f"[no LLM configured] category={category}; "
            f"{n_docs} document chunk(s) and {n_dynamic} live data result(s) retrieved "
            f"for: {question!r}."
        ]
        ctx_warnings = list(context.get("warnings", []) or [])
        if ctx_warnings:
            parts.append("Warnings: " + "; ".join(ctx_warnings))

        return LLMResponse(
            text=" ".join(parts),
            provider=self.name,
            grounded=False,
            warnings=ctx_warnings + [
                "NullLLMProvider never synthesizes an answer -- configure a real "
                "LLMProvider (docs/LLM_ORCHESTRATION.md sec 6, Decision Required) to get "
                "natural-language output."
            ],
            sources=list(context.get("document_context", []) or []),
            dynamic_data=[{"tool": k, "value": v} for k, v in (context.get("dynamic_context") or {}).items()],
            used_tools=list(context.get("tools_called", []) or []),
            confidence="unknown",
        )


class TemplateLLMProvider(LLMProvider):
    """Phase 6's "first provider" (task sec 7) -- deterministic,
    rule-based, zero network, zero API key. Restates `context` content
    into a fixed French-language template; never writes a word that
    doesn't trace back to `context["document_context"]`,
    `context["dynamic_context"]`, or the question itself. This is what
    makes it safe to register as a REAL (non-null) provider before the
    external-vs-local provider decision is made
    (docs/LLM_PROVIDER_DECISION.md) -- it never transmits project data
    anywhere, and by construction it cannot hallucinate a number (every
    number it writes is copied from `context`, verified by
    `grounding_check.check_grounding()` on every call, not just
    asserted).

    NOT a substitute for a real LLM's synthesis/reasoning quality -- it
    does not paraphrase, does not resolve apparent tension between
    sources, does not answer a question CONTEXT doesn't directly address
    even if a human reader could infer the answer. Its purpose is
    narrower and more mechanical: prove the Grounded Answer Contract,
    citation system, and hallucination boundary work end to end. See
    docs/LLM_EVALUATION.md for what its own benchmark run does and does
    not demonstrate."""

    name = "template"
    MAX_CHUNK_CHARS = 400

    def generate(self, context: Dict[str, Any]) -> LLMResponse:
        question = context.get("question", "")
        document_context = context.get("document_context", []) or []
        dynamic_context = context.get("dynamic_context", {}) or {}
        ctx_warnings = list(context.get("warnings", []) or [])

        if not grounding_check.has_any_context(context):
            text = grounding_check.NO_INFORMATION_MESSAGE
        else:
            text = self._render(question, document_context, dynamic_context)

        result = grounding_check.check_grounding(text, context)
        confidence = grounding_check.estimate_confidence(context, result)

        warnings = list(ctx_warnings)
        if not result.grounded:
            # Should not happen by construction (every number/phase this provider
            # writes is copied verbatim from context) -- reported, not hidden, if
            # it ever does (e.g. a future edit to _render introduces free text).
            warnings.append(
                f"TemplateLLMProvider's own grounding check found ungrounded content: "
                f"numbers={result.ungrounded_numbers}, phases={result.ungrounded_phase_tokens}."
            )

        return LLMResponse(
            text=text, provider=self.name, grounded=result.grounded, warnings=warnings,
            sources=list(document_context), dynamic_data=[
                {"tool": k, "value": v} for k, v in dynamic_context.items()
            ],
            used_tools=list(context.get("tools_called", []) or []), confidence=confidence,
        )

    def _render(self, question: str, document_context: List[Dict], dynamic_context: Dict) -> str:
        parts = [f"Question : {question}"]

        if dynamic_context:
            parts.append("Données dynamiques (Reporting API, en direct) :")
            for tool_name, value in dynamic_context.items():
                parts.append(f"- [{tool_name}] {self._summarize_dynamic(value)}")

        if document_context:
            parts.append("Documentation (sources statiques) :")
            for chunk in document_context:
                excerpt = (chunk.get("content") or "")[: self.MAX_CHUNK_CHARS].replace("\n", " ")
                parts.append(
                    f"- [Source: {chunk.get('source')} sec {chunk.get('section')!r}] {excerpt}"
                )

        return "\n".join(parts)

    @staticmethod
    def _summarize_dynamic(value: Any) -> str:
        if not isinstance(value, dict):
            return str(value)
        current_state = value.get("current_state")
        if isinstance(current_state, dict) and "current_phase" in current_state:
            model_version = (value.get("model") or {}).get("model_version", "?")
            return (
                f"phase actuelle = {current_state['current_phase']} "
                f"(probabilite={current_state.get('phase_probability')}) "
                f"[model_version={model_version}]"
            )
        if "transition_chain" in value:
            n = len(value.get("transition_chain") or [])
            return f"trajectoire : {value.get('n_windows')} fenetres, {n} segment(s) dans transition_chain"
        if "model_version" in value:
            return f"model_version={value['model_version']}, model_name={value.get('model_name')}"
        return str(value)


# ---------------------------------------------------------------------------
# Context window (num_ctx)
# ---------------------------------------------------------------------------
# MEASURED, not assumed (2026-09-02, real GPU server, ollama 0.7.0):
#   * With no `num_ctx` in options, a ~21k-token prompt came back with
#     `prompt_eval_count = 4096` EXACTLY -- Ollama silently truncates to its
#     own default and the model never sees the rest of the prompt. Explicitly
#     sending num_ctx=4096 reproduced it identically; sending num_ctx=16384
#     gave `prompt_eval_count = 16384`. So 4096 was never a server or model
#     limit: `ollama serve` runs with no --ctx-size, the Modelfile declares no
#     num_ctx, and llama3.2's own `llama.context_length` is 131072.
#   * This is why the benchmark's biggest prompts were unusable: after the
#     2026-09-02 top_k compaction the largest assembled prompt is still ~9.8k
#     estimated tokens, i.e. >2x the silent 4096 cut.
#
# Why 16384: it covers the largest measured prompt with room for the answer
# (num_ctx bounds prompt + generated tokens together), and it fits. Measured
# footprint on the one GPU Ollama sees (CUDA_VISIBLE_DEVICES=0, TITAN V,
# 12288 MiB): 4.2 GB / 3664 MiB used at 4096, 8.4 GB / 7840 MiB at 16384 --
# ~4.4 GB still free. 32768 would extrapolate past the card and force a
# partial CPU offload, which on this hardware is the known latency wall.
DEFAULT_NUM_CTX = 16384

# Escape hatch for an operator who needs a different window without editing
# code (a bigger card, a deliberately-degraded comparison run). Read once at
# construction so a provider instance's behaviour cannot change mid-run.
NUM_CTX_ENV_VAR = "EMBRYO_OLLAMA_NUM_CTX"


def resolve_num_ctx(explicit: Optional[int] = None) -> int:
    """Precedence: explicit argument > $EMBRYO_OLLAMA_NUM_CTX > DEFAULT_NUM_CTX.

    Fails loudly on a nonsensical value rather than silently falling back --
    a silently-wrong context window is exactly the failure this constant
    exists to end."""
    if explicit is not None:
        if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit < 1:
            raise ValueError(f"num_ctx must be a positive int, got {explicit!r}.")
        return explicit
    raw = os.environ.get(NUM_CTX_ENV_VAR)
    if raw is not None and raw.strip() != "":
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(
                f"{NUM_CTX_ENV_VAR}={raw!r} is not an integer."
            ) from e
        if value < 1:
            raise ValueError(f"{NUM_CTX_ENV_VAR}={raw!r} must be a positive int.")
        return value
    return DEFAULT_NUM_CTX


class OllamaLLMProvider(LLMProvider):
    """First LOCAL-model provider (Phase 6, human decision recorded
    2026-08-26, docs/LLM_PROVIDER_DECISION.md) -- calls a local Ollama
    server only (default `http://localhost:11434`). No project data ever
    leaves the machine this process runs on -- resolves the SENSITIVE-data
    confidentiality blocker (docs/LLM_PROVIDER_DECISION.md's data
    classification: Reporting API dynamic outputs) by construction, not
    by a policy a future change could accidentally violate: this class
    has no code path capable of reaching any host other than `base_url`.

    NOT auto-enabled anywhere -- `NullLLMProvider` remains
    `orchestrator.answer_question()`'s default; using this provider is
    always an explicit `llm=OllamaLLMProvider(model=...)` argument.

    Requires, OUTSIDE this class (never done by it -- this project's
    "explicit fallback, never a silent side-effecting install"
    discipline, same as every other tool in this package):
      1. Ollama installed and running (`ollama serve`) wherever this
         process runs.
      2. The chosen model already pulled (`ollama pull <model>`).
    Neither has been done anywhere in this project as of this class being
    written -- see docs/LLM_PROVIDER_DECISION.md's "Next action"."""

    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 request_timeout_seconds: float = 30.0, seed: Optional[int] = None,
                 temperature: Optional[float] = None, num_ctx: Optional[int] = None,
                 context_format: Optional[str] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        # Optional, default None (Ollama's own default sampling, unchanged
        # behavior for every existing caller). Ollama's sampling is
        # stochastic by default -- no seed means two calls with the exact
        # same prompt can produce different text. Set this to make a
        # provider comparison (e.g. before/after a prompt-formatting
        # change) a true single-variable experiment instead of also
        # sampling new randomness on every run.
        self.seed = seed
        # Optional, default None (Ollama's own default temperature, unchanged
        # behavior for every existing caller). P1(b) DECISION (2026-08-31,
        # docs/RAG_LLM_IMPROVEMENT_REPORT.md P2.4): a lower temperature is
        # being A/B-tested as one candidate fix for LLM under-extraction with
        # correct context (docs/RAG_LLM_QUALITY_REPORT.md sec 8-9) -- not
        # assumed to help, measured against the same controlled-seed
        # methodology already used for the dynamic_context flattening fix.
        self.temperature = temperature
        # ALWAYS sent (unlike seed/temperature, which stay opt-in): omitting it
        # is what silently truncated every prompt above 4096 tokens. Resolved
        # once here so an invalid value raises at construction, not per request.
        self.num_ctx = resolve_num_ctx(num_ctx)
        # Which RENDERING of the context this provider sends (prompt.py's
        # `resolve_context_format`): the historical "v2" by default, so every
        # existing caller is byte-identical to before COMPACT_CONTEXT_V1
        # existed. Resolved once at construction, like num_ctx, so a
        # provider instance cannot change arm mid-run -- which is what makes
        # a paired A/B run a single-variable experiment.
        self.context_format = prompt.resolve_context_format(context_format)

    def generate(self, context: Dict[str, Any]) -> LLMResponse:
        import json as _json
        import urllib.error
        import urllib.request

        request_body: Dict[str, Any] = {
            "model": self.model,
            "system": prompt.build_system_prompt(),
            "prompt": prompt.build_user_prompt(context, context_format=self.context_format),
            "stream": False,
        }
        options: Dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if options:
            request_body["options"] = options
        payload = _json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = _json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise OllamaUnavailableError(
                f"Could not reach local Ollama server at {self.base_url} for model "
                f"{self.model!r}: {e}. Is `ollama serve` running, and has "
                f"`ollama pull {self.model}` been run?"
            ) from e

        text = body.get("response", "")
        check = grounding_check.check_grounding(text, context)
        return LLMResponse(
            text=text, provider=self.name, grounded=check.grounded,
            warnings=list(context.get("warnings", []) or []),
            sources=list(context.get("document_context", []) or []),
            dynamic_data=[{"tool": k, "value": v} for k, v in (context.get("dynamic_context") or {}).items()],
            used_tools=list(context.get("tools_called", []) or []),
            confidence=grounding_check.estimate_confidence(context, check),
        )

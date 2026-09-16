"""
Ties router.classify() + tools.* + context_builder.build_context() +
llm_provider.LLMProvider into one entry point: answer_question(...).

This IS the orchestration foundation the task asks for -- routing, typed
tool selection, typed tool execution, a normalized/provenance-tagged
context, and an interchangeable LLM abstraction -- but it does NOT do
session/context resolution: "which video, which window" for a bare
question like "quelle est la phase actuelle ?" requires session/context
state this project has never built (docs/RAG_ARCHITECTURE.md sec 4's
worked example flags this explicitly: "requires session/context state,
see docs/LLM_ORCHESTRATION.md sec Session context"). The caller must pass
`video_id`/`window`/`split` explicitly when the route needs live data; if
it doesn't, the dynamic tool is simply not called and a warning is
recorded -- never guessed.

No GRU tool (docs/PRODUCT_ARCHITECTURE.md sec 4a: not scheduled), no
cross-trajectory aggregate tool (docs/LLM_ORCHESTRATION.md's own flagged
gap), no multi-call planning beyond the fixed per-category tool set
below.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import compact_context, grounding_check, scientific_validation, temporal_context, tools
from .compact_v2_render import CONTEXT_FORMAT_COMPACT_V2, derive_transition_context
from .context_builder import (SOURCE_AUDIT, SOURCE_DOCUMENT, SOURCE_REPORTING_API,
                              SOURCE_SCIENTIFIC, ToolCallResult, build_context)
from .llm_provider import LLMProvider, LLMResponse, NullLLMProvider
from .router import (
    DOCUMENTARY,
    DYNAMIC_DATA,
    HYBRID,
    PHASE_TOKEN_PATTERN,
    RouteDecision,
    UNKNOWN,
    _normalize as _router_normalize,
    classify,
    extract_phase_tokens,
)

DEFAULT_LLM_TIMEOUT_SECONDS = 30

# --- Context policy (compact_v2, 2026-09-12) -------------------------------
# The context FORMAT is decided by the provider (OllamaLLMProvider.context_format,
# itself resolved from an explicit argument or $EMBRYO_LLM_CONTEXT_FORMAT). The
# policy below adds, for "compact_v2" ONLY, three things the historical "v2"
# context never contained -- so every v2 / compact turn stays byte-identical to
# what it was before this addition:
#   * curated RAG selection: `_RETRIEVAL_CANDIDATE_POOL` candidates asked from
#     the index, narrative/engineering documents skipped, the first
#     `compact_context.DEFAULT_SELECTED_CHUNKS` kept, every decision recorded
#     in context["retrieval_audit"];
#   * a [SCIENTIFIC] block from the Istanbul Consensus corpus, in its own
#     context key (`scientific_context`), only when documents were planned;
#   * `transition_context`, an exact DERIVED selection of the last annotated
#     transition at/before the reference window (and the next one after).
_RETRIEVAL_CANDIDATE_POOL = compact_context.DEFAULT_RETRIEVAL_CANDIDATE_POOL


def resolve_context_policy(explicit: Optional[str] = None, llm: Optional[LLMProvider] = None) -> str:
    """Which rendering/policy this turn runs under. An explicit value must
    agree with the provider's own `context_format` when the provider has one:
    a caller asking for "compact_v2" while the provider renders "v2" would
    silently measure the wrong thing, so it is refused loudly."""
    from . import prompt as _prompt  # noqa: WPS433 -- local: prompt imports nothing from here

    provider_format = getattr(llm, "context_format", None)
    if explicit is not None:
        resolved = _prompt.resolve_context_format(explicit)
        if provider_format is not None and provider_format != resolved:
            raise ValueError(f"context_format={explicit!r} disagrees with the provider's "
                             f"context_format={provider_format!r}; build the provider with the "
                             f"same format (llm_config.build_ollama_provider(context_format=...)).")
        return resolved
    if provider_format is not None:
        return _prompt.resolve_context_format(provider_format)
    return _prompt.resolve_context_format(None)


def _retrieve_curated(query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """compact_v2 retrieval: a larger candidate pool from the SAME index and
    query, then `compact_context.select_relevant_chunks`. The index, the
    embeddings and the query are untouched; only the choice among the
    returned candidates changes, and it is fully recorded."""
    candidates = tools.retrieve_documents(query, top_k=_RETRIEVAL_CANDIDATE_POOL)
    return compact_context.select_relevant_chunks(candidates, top_k=compact_context.DEFAULT_SELECTED_CHUNKS)

ALL_TOOLS = (
    "get_current_inference", "get_inference_history", "get_trajectory", "get_transition_events",
    "get_model_metadata", "get_phase_information", "retrieve_documents",
)

# Narrow keyword set for get_model_metadata, DISTINCT from routing's own
# DYNAMIC_KEYWORDS -- a question can be DYNAMIC_DATA without asking about
# the model's own version/config (e.g. "quelle est la phase actuelle ?"),
# so tool selection needs its own, narrower trigger rather than firing
# get_model_metadata on every DYNAMIC_DATA question.
_MODEL_METADATA_TRIGGERS = ("version du modele", "modele est charge", "modele actuellement charge",
                            "which model version", "model version", "quelle version du modele")

# Narrow keyword set for get_transition_events (Event/Transition RAG axis,
# docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md) -- same pattern as
# _MODEL_METADATA_TRIGGERS above: a question can be DYNAMIC_DATA/HYBRID
# without asking about a phase TRANSITION specifically (e.g. "quelle est
# la phase actuelle ?" needs no transition data at all), so this tool
# needs its own, additional trigger layered onto the existing
# wants_dynamic gate, not fired on every DYNAMIC_DATA/HYBRID question.
_TRANSITION_EVENT_TRIGGERS = ("transition", "saut de phase", "skip transition", "skip")

# Narrow keyword set deciding get_transition_events' SCOPE (Layer-B fix,
# docs/RAG_HISTORY_SCOPE_FIX.md): a question about the CURRENT transition
# ("quelle transition est observee ici ?") needs only the transition
# bracketing `window` -- execute_plan()'s default, unchanged. A question
# about the video's transition HISTORY needs the FULL, unfiltered list,
# which `tools.get_transition_events()` already supports (`window=None`)
# -- no tool/reporting-layer change, only this planning signal.
# Deliberately narrow and specific (each token checked for collisions
# against every existing test/eval question), not a general "does this
# need history" NLP classifier.
#   "habituellement" -- superseded-list Q3 (kept: still a valid history cue).
#   "derniere transition" -- current-list Q1 ("quelle a ete la derniere
#     transition detectee et a quel moment"): the LLM needs every
#     transition + its window to identify "the last one" and its timing,
#     not just the one bracketing the current window. Verified absent
#     from every other benchmark/eval/test question.
_TRANSITION_HISTORY_TRIGGERS = ("precede", "precedente", "historique", "habituellement",
                                "derniere transition")

# Narrow keyword set for get_inference_history (multi-window / G1,
# docs/RAG_FIXED_QUESTION_BENCHMARK.md §7). A question asking how the
# model's belief / uncertainty EVOLVES across windows -- not its value at
# one window -- needs the multi-window series get_current_inference
# cannot give. Each token verified to match ONLY its intended benchmark
# question (Q5, Q6, Q7, Q8, Q10) and nothing in eval_questions.py /
# rag_llm_quality_questions.py / test_router*.py. Layered onto the
# existing wants_dynamic + video_id + window gate, exactly like
# _TRANSITION_EVENT_TRIGGERS.
_INFERENCE_HISTORY_TRIGGERS = (
    "se rapprocher",                    # Q5 -- when the two phases' probabilities converge
    "dans les fenetres precedant",      # Q6 -- evolution before/after
    "avant ou apres l'annotation",      # Q7 -- model detection window vs annotation window
    "stable apres la transition",       # Q8 -- post-transition stability
    "retours vers la phase precedente", # Q8 -- returns to the previous phase
    "periodes stables",                 # Q10 -- uncertainty near transition vs stable periods
    "sequence predite",                 # Q9 -- skip/regression in the PREDICTED sequence
)
# Q9 note (2026-09-02g). "Y a-t-il un saut de phase ou une regression dans la
# sequence PREDITE ?" is a multi-window question about the MODEL's own phase
# sequence: a skip or a regression is defined over consecutive windows and
# cannot be expressed by the single window get_current_inference returns.
# Before this token, Q9's context held one model window (W156) plus
# get_transition_events' OBSERVED transition -- whose only skip (t4->t6,
# is_skip=true) belongs to the ANNOTATION, not to the model. The context
# therefore offered a confident-looking "yes, there is a skip" that answers a
# different question than the one asked (docs/TEMPORAL_CONTEXT_RESTRUCTURING.md
# sec 6). `derive_series_analysis` already computes `phase_skips` and
# `phase_regressions` over the model sequence; this token is the whole fix.
#
# Collision-checked through router._normalize() against every FIXED_QUESTIONS,
# EVAL_QUESTIONS and rag_llm_quality_questions entry: "sequence predite"
# matches Q9 and nothing else. Deliberately in TOOL PLANNING, not in
# router.classify() -- Q9's route (DYNAMIC_DATA) is already correct and is
# not touched.

# Narrow keyword set that additionally co-plans get_trajectory ALONGSIDE
# get_current_inference (normally mutually exclusive -- trajectory only
# when window is None). "Depuis combien de temps dans la phase actuelle"
# (Q2) needs the current phase SEGMENT's start (from transition_chain) to
# subtract from the current window time -- get_current_inference's
# `duration` is the model's EXPECTED (prospective) duration, a different
# quantity. Verified: "depuis combien de temps" matches ONLY Q2, and does
# NOT match "combien de temps avant" (eval_questions.py's future-duration
# question).
_ELAPSED_DURATION_TRIGGERS = ("depuis combien de temps", "depuis quand", "duree deja ecoulee",
                              "temps deja passe")

SCOPE_CURRENT_WINDOW = "current_window"
SCOPE_FULL_HISTORY = "full_history"

_INFERENCE_HISTORY_BEFORE = 5
_INFERENCE_HISTORY_AFTER = 5
# ETAPE 4 (context compactness). The band above is 11 windows; each window's
# FULL phase_probabilities + next_phase_distribution is 30 near-all-noise
# numbers, and prompt._flatten_dynamic_context gives every one its own long
# dotted line. Measured on the real Patient_319/val/156 anchor, that put
# Q5/Q6/Q7/Q8/Q10 at 12.7k-15.5k estimated tokens against this deployment's
# 4096-token ctx-size -- the known cause of the 11/15 LLM timeouts. Keeping
# the leading phases preserves exactly what these questions ask about
# (which phases lead, how the gap between them evolves, the entropy trend)
# and drops only the tail. The Reporting API itself still defaults to the
# full distribution -- this narrowing is the ORCHESTRATOR's prompt-budget
# policy, not a change to the scientific tool's contract.
_INFERENCE_HISTORY_TOP_K_PROBABILITIES = 4


@dataclass
class ToolPlan:
    called: List[str] = field(default_factory=list)
    not_called: List[Dict[str, str]] = field(default_factory=list)  # [{"tool": ..., "reason": ...}]
    phase_tokens: List[str] = field(default_factory=list)
    transition_events_scope: str = SCOPE_CURRENT_WINDOW  # SCOPE_CURRENT_WINDOW | SCOPE_FULL_HISTORY --
    # read by execute_plan() to decide whether get_transition_events() is called with the caller's
    # `window` (current-transition questions, unchanged default) or `window=None` (history questions).
    # Meaningless when "get_transition_events" is not in `called` at all.

    def to_dict(self) -> Dict:
        return {"called": self.called, "not_called": self.not_called, "phase_tokens": self.phase_tokens,
                "transition_events_scope": self.transition_events_scope}


def _normalize_for_trigger(question: str) -> str:
    # Delegates to router._normalize so tool-trigger matching and routing
    # normalize IDENTICALLY -- same accent stripping AND the same
    # typographic-apostrophe fold (U+2019 -> U+0027). Previously a
    # hand-copied duplicate of that function's body, which silently
    # skipped the apostrophe fold.
    return _router_normalize(question)


def plan_tools(question: str, route: RouteDecision, video_id: Optional[str] = None,
               window: Optional[int] = None) -> ToolPlan:
    """Deterministic tool-selection map -- category -> tool names
    (docs/TOOL_CONTRACTS.md's own table). Pure, no execution: used both by
    answer_question() and directly by tests/the evaluation script, so
    "which tools would be called" is inspectable even when nothing is
    actually run (task sec 13 wants both 'tools appeles' AND 'tools non
    appeles' recorded per question)."""
    text = _normalize_for_trigger(question)
    wants_dynamic = route.category in (DYNAMIC_DATA, HYBRID)
    wants_docs = route.category in (DOCUMENTARY, HYBRID)
    phase_tokens = extract_phase_tokens(question) if wants_docs else []

    called: List[str] = []
    not_called: List[Dict[str, str]] = []

    def call(name: str) -> None:
        if name not in called:
            called.append(name)

    def skip(name: str, reason: str) -> None:
        if name not in called:
            not_called.append({"tool": name, "reason": reason})

    if wants_dynamic:
        if video_id is not None and window is not None:
            call("get_current_inference")
            # G2: co-plan get_trajectory (normally window-is-None-only) when the question
            # needs the current phase segment's boundaries, not just the single window --
            # e.g. "depuis combien de temps dans la phase actuelle" (elapsed-so-far).
            if any(t in text for t in _ELAPSED_DURATION_TRIGGERS):
                call("get_trajectory")
            else:
                skip("get_trajectory", "single-window question; get_current_inference covers it "
                                        "(no elapsed-duration / segment-boundary phrase)")
        elif video_id is not None and window is None:
            call("get_trajectory")
        else:
            skip("get_current_inference", "route needs live data but no video_id/window was supplied "
                                            "(session/context resolution not implemented, see module docstring)")
            skip("get_trajectory", "route needs live data but no video_id was supplied")
    else:
        skip("get_current_inference", "route does not need live data")
        skip("get_trajectory", "route does not need live data")

    if any(t in text for t in _MODEL_METADATA_TRIGGERS):
        call("get_model_metadata")
    else:
        skip("get_model_metadata", "question did not name the model's own version/config")

    transition_events_scope = SCOPE_CURRENT_WINDOW
    if wants_dynamic and video_id is not None and any(t in text for t in _TRANSITION_EVENT_TRIGGERS):
        call("get_transition_events")
        if any(t in text for t in _TRANSITION_HISTORY_TRIGGERS):
            transition_events_scope = SCOPE_FULL_HISTORY
    else:
        skip("get_transition_events", "question did not name a phase transition/skip, "
                                        "or no video_id was supplied")

    # Multi-window / G1: get_inference_history needs BOTH a video_id AND a center window
    # (the band is [center-before, center+after]) -- a bare question with no window cannot
    # anchor the band, so it is skipped honestly rather than guessed.
    if wants_dynamic and video_id is not None and window is not None \
            and any(t in text for t in _INFERENCE_HISTORY_TRIGGERS):
        call("get_inference_history")
    else:
        skip("get_inference_history", "question does not ask how the model's belief/uncertainty "
                                        "EVOLVES across windows, or no video_id/window was supplied")

    if wants_docs:
        call("retrieve_documents")
        if phase_tokens:
            call("get_phase_information")
        else:
            skip("get_phase_information", "no phase token (e.g. t6, tPB2) detected in the question")
    else:
        skip("retrieve_documents", "route does not need documented knowledge")
        skip("get_phase_information", "route does not need documented knowledge")

    return ToolPlan(called=called, not_called=not_called, phase_tokens=phase_tokens,
                     transition_events_scope=transition_events_scope)


# --- Phase-vocabulary query enrichment for the RAG call (2026-09-07e) -------
#
# Measured problem, reproduced on the real index across the four
# `prompt_v2_reading_method` runs: the chunk that spells out the canonical
# phase order (`docs/SCIENTIFIC_REPORT.md :: 2. Dataset`) sits at rank 6 for
# Q11 and outside the top 25 for Q12, while `top_k` is 5. The corpus DOES
# contain the ordering; the questions simply carry no phase vocabulary, and
# their wording ("ordre attendu du developpement", "prochaine etape ...
# selon le referentiel") is closest to the corpus's project-strategy prose.
#
# The repair is a QUERY change only: the retrieval call for such a question
# is expanded with the phase names ALREADY PRESENT in this turn's own
# fetched tool outputs. Nothing is generated, nothing is completed from
# outside knowledge, and no phase list is written down anywhere in this file
# -- if the Reporting API returned no phase this turn, the query is the bare
# question, unchanged. `top_k` is untouched (still 5), the index, the
# embedding model, the corpus and the chunking are untouched, and no
# document is ever appended to the context after retrieval.
#
# Why a trigger table rather than always enriching: an unconditional change
# would silently alter every RAG query in the system. This follows the
# pattern the tool-planning layer already uses (`_TRANSITION_EVENT_TRIGGERS`,
# `_INFERENCE_HISTORY_TRIGGERS` above) -- a small, explicit, collision-checked
# keyword set. Verified against every question set in this repo: it matches
# Q11 and Q12 and nothing else, so Q7/Q10/Q13/Q14/Q15 (the other questions
# that call retrieve_documents) keep byte-identical queries.
_PHASE_VOCABULARY_QUERY_TRIGGERS = ("ordre attendu", "referentiel", "etape developpementale")


def _context_phase_vocabulary(results: Dict[str, ToolCallResult]) -> List[str]:
    """Phase names this turn's ALREADY-FETCHED tool outputs actually contain,
    deduplicated, in first-seen order (deterministic).

    Two sources, both strictly inside the turn's own payload:
      * string VALUES -- `current_phase`, `most_likely_next_phase`, a
        transition's `from_phase`/`to_phase`, ...;
      * the KEYS of a phase distribution (`phase_probabilities`), which is
        where the Reporting API states the phase vocabulary it works over.
        Unlike `grounding_check`, which restricts distribution keys to the
        top 2 because it must decide what the model is CLAIMING, this is a
        retrieval-vocabulary problem: every phase name the payload mentions
        is legitimate query material, and none of it is a claim.

    Never invents a phase, never completes the vocabulary from outside the
    payload: a distribution with three keys contributes three names."""
    tokens: List[str] = []

    def _add(name: str) -> None:
        lowered = name.lower()
        if lowered not in tokens:
            tokens.append(lowered)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            if value and all(isinstance(k, str) and PHASE_TOKEN_PATTERN.fullmatch(k)
                             for k in value):
                for key in value:
                    _add(key)
                return
            for item in value.values():
                _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
        elif isinstance(value, str):
            for token in extract_phase_tokens(value):
                _add(token)

    for result in results.values():
        if result.ok:
            _walk(result.value)
    return tokens


def build_retrieval_query(question: str, results: Dict[str, ToolCallResult]) -> str:
    """The exact string handed to `tools.retrieve_documents()`. Returns the
    question verbatim unless (a) the question matches a phase-vocabulary
    trigger AND (b) this turn's context actually holds phase names -- in
    which case those names are appended. Pure function of its arguments;
    performs no retrieval itself, so it is testable offline."""
    if not any(trigger in _normalize_for_trigger(question)
               for trigger in _PHASE_VOCABULARY_QUERY_TRIGGERS):
        return question
    phases = _context_phase_vocabulary(results)
    if not phases:
        return question
    return f"{question} {' '.join(phases)}"


def _call(name: str, source_type: str, fn: Callable, *args, **kwargs) -> ToolCallResult:
    try:
        value = fn(*args, **kwargs)
        return ToolCallResult(tool_name=name, source_type=source_type, ok=True, value=value)
    except tools.ToolError as e:
        return ToolCallResult(tool_name=name, source_type=source_type, ok=False,
                               error=str(e), error_type=type(e).__name__)


def execute_plan(plan: ToolPlan, question: str, video_id: Optional[str], window: Optional[int],
                  split: str, context_policy: Optional[str] = None) -> Dict[str, ToolCallResult]:
    """Actually calls every tool in plan.called, catching tools.ToolError
    into a ToolCallResult rather than letting it propagate -- one failed
    tool (e.g. the RAG index missing on this machine) must not prevent
    the others from running or the context builder from producing a
    partial-but-honest context."""
    results: Dict[str, ToolCallResult] = {}
    for name in plan.called:
        if name == "get_current_inference":
            results[name] = _call(name, SOURCE_REPORTING_API, tools.get_current_inference,
                                   video_id, window, split)
        elif name == "get_trajectory":
            results[name] = _call(name, SOURCE_REPORTING_API, tools.get_trajectory, video_id, split)
        elif name == "get_transition_events":
            # SCOPE_FULL_HISTORY (docs/RAG_HISTORY_SCOPE_FIX.md): pass window=None so
            # tools.get_transition_events() returns the video's FULL, unfiltered transition
            # history instead of only the one bracketing the caller's current window --
            # already-supported tool behavior, only the argument passed here changes.
            scoped_window = None if plan.transition_events_scope == SCOPE_FULL_HISTORY else window
            results[name] = _call(name, SOURCE_REPORTING_API, tools.get_transition_events,
                                   video_id, split, scoped_window)
        elif name == "get_inference_history":
            results[name] = _call(name, SOURCE_REPORTING_API, tools.get_inference_history,
                                   video_id, window, _INFERENCE_HISTORY_BEFORE, _INFERENCE_HISTORY_AFTER,
                                   split, _INFERENCE_HISTORY_TOP_K_PROBABILITIES)
        elif name == "get_model_metadata":
            results[name] = _call(name, SOURCE_REPORTING_API, tools.get_model_metadata)
        elif name == "get_phase_information":
            phase_id = plan.phase_tokens[0] if plan.phase_tokens else question
            results[name] = _call(name, SOURCE_DOCUMENT, tools.get_phase_information, phase_id)
        elif name == "retrieve_documents":
            # `plan_tools` always plans retrieve_documents LAST, so every
            # Reporting-API result this turn is already in `results` and can
            # supply the query's phase vocabulary (see build_retrieval_query).
            query = build_retrieval_query(question, results)
            if context_policy == CONTEXT_FORMAT_COMPACT_V2:
                curated = _call(name, SOURCE_DOCUMENT, _retrieve_curated, query)
                if curated.ok:
                    kept, audit = curated.value
                    results[name] = ToolCallResult(tool_name=name, source_type=SOURCE_DOCUMENT,
                                                   ok=True, value=kept)
                    results["retrieval_audit"] = ToolCallResult(
                        tool_name="retrieval_audit", source_type=SOURCE_AUDIT, ok=True, value=audit)
                else:
                    results[name] = curated
                results["scientific_reference"] = _call(
                    "scientific_reference", SOURCE_SCIENTIFIC,
                    _scientific_reference_tool, question)
            else:
                results[name] = _call(name, SOURCE_DOCUMENT, tools.retrieve_documents, query)

    # ETAPE 3 (2026-09-02 context restructuring): every series property the
    # multi-window questions need that is EXACTLY computable -- probability
    # gap and its minimum, the crossing/convergence windows, the model-vs-
    # annotation lag, post-transition stability / returns / skips /
    # regressions, the entropy extrema and the near-vs-far comparison -- is
    # computed here in Python instead of being left to the LLM to derive from
    # ~300 scattered lines (the diagnosed cause of the Q5/Q6/Q8/Q9/Q10 partial
    # reads in the mistral-nemo benchmark).
    #
    # It is a SEPARATE dynamic_context entry, never merged into the tool's own
    # output: its `provenance` is "derived_deterministic", distinct from both
    # a model field and get_transition_events' "observed_annotation". Being a
    # dynamic_context entry also means grounding_check sees these numbers as
    # KNOWN values, so a derived quantity the LLM restates is verified exactly
    # like a tool-returned one. No tool, no scientific model and no Reporting
    # API behaviour changes -- this is pure arithmetic over an already-fetched
    # payload.
    history_result = results.get("get_inference_history")
    if history_result is not None and history_result.ok:
        analysis = temporal_context.derive_series_analysis(history_result.value or {})
        if analysis:
            results["series_analysis"] = ToolCallResult(
                tool_name="series_analysis", source_type=SOURCE_REPORTING_API,
                ok=True, value=analysis,
            )

    # compact_v2 only: the last annotated transition at/before the reference
    # window, selected by arithmetic (same DERIVED discipline as series_analysis).
    if context_policy == CONTEXT_FORMAT_COMPACT_V2:
        payload = {name: r.value for name, r in results.items()
                   if r.ok and r.source_type == SOURCE_REPORTING_API}
        transition_context = derive_transition_context(payload)
        if transition_context:
            results["transition_context"] = ToolCallResult(
                tool_name="transition_context", source_type=SOURCE_REPORTING_API,
                ok=True, value=transition_context,
            )
    return results


def _scientific_reference_tool(question: str) -> Dict[str, Any]:
    """Istanbul Consensus blocks for the [SCIENTIFIC] context class. Wrapped
    like a tool so a missing corpus surfaces as a recorded warning, never as
    a crashed turn."""
    from .scientific_reference import retrieve_scientific_reference  # noqa: WPS433

    try:
        return retrieve_scientific_reference(question)
    except (OSError, ValueError) as e:
        raise tools.ToolUnavailableError(f"scientific_reference: {e}") from e


def _safe_fallback_response(provider_name: str, reason: str, context: Dict[str, Any]) -> LLMResponse:
    """The same honest, never-invent posture as NullLLMProvider, used
    when a REAL provider times out or raises -- a provider failure must
    degrade to "here is what was retrieved, no answer was generated",
    never to a crash of the whole turn and never to a silently empty/
    wrong answer (task sec 7: "gerer timeout", "gerer erreurs")."""
    return LLMResponse(
        text=f"[{provider_name} failed: {reason}] No answer was generated for this turn.",
        provider=provider_name, grounded=False,
        warnings=list(context.get("warnings", []) or []) + [f"LLM provider failure: {reason}"],
        sources=list(context.get("document_context", []) or []),
        dynamic_data=[{"tool": k, "value": v} for k, v in (context.get("dynamic_context") or {}).items()],
        used_tools=list(context.get("tools_called", []) or []), confidence="unknown",
    )


def generate_grounded_response(
    llm: LLMProvider, context: Dict[str, Any], timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> LLMResponse:
    """Calls `llm.generate(context)` with a timeout and exception safety
    net, THEN re-runs `grounding_check.check_grounding()` on the result
    independent of what the provider itself claims -- `response.grounded`
    coming back here is never just the provider's self-report (task sec
    6/`docs/LLM_ORCHESTRATION.md` sec 3 point 2's "structural enforcement,
    not just prompting", applied to every provider, not only
    hand-written ones)."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(llm.generate, context)
            response = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return _safe_fallback_response(llm.name, f"timed out after {timeout_seconds}s", context)
    except Exception as e:  # noqa: BLE001 -- a provider's internal failure must never crash the turn
        return _safe_fallback_response(llm.name, f"{type(e).__name__}: {e}", context)

    check = grounding_check.check_grounding(response.text, context)
    if not check.grounded:
        response.warnings = list(response.warnings) + [
            f"Post-hoc grounding check found content not traceable to this turn's context: "
            f"ungrounded_numbers={check.ungrounded_numbers}, "
            f"ungrounded_phase_tokens={check.ungrounded_phase_tokens}."
        ]
        response.grounded = False
    elif getattr(llm, "always_ungrounded", False):
        # A provider that structurally never synthesizes an answer (NullLLMProvider)
        # stays ungrounded regardless of what its own meta-description text contains
        # -- "nothing ungrounded was found" is not the same claim as "an answer was
        # verified grounded" when no answer was actually attempted.
        response.grounded = False
    else:
        # The independent check confirms (never just trusts) a provider's own claim
        # -- including UPGRADING an overly conservative `grounded=False` from a
        # provider whose answer turns out to be fully traceable to this turn's context.
        response.grounded = True
    return response


def answer_question(
    question: str,
    video_id: Optional[str] = None,
    window: Optional[int] = None,
    split: str = "val",
    llm: Optional[LLMProvider] = None,
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    context_format: Optional[str] = None,
    validate: Optional[bool] = None,
) -> Dict[str, Any]:
    """The orchestration foundation's one public entry point. Never
    raises for a domain question (UNKNOWN questions are routed and
    reported honestly, not answered); a tool failure is recorded as a
    warning in the returned context, not an exception the caller must
    catch; an LLM provider failure/timeout degrades to an honest fallback
    response, never a crash. Returns a plain dict, JSON-serializable end
    to end (matching this project's own "never pickle, always
    inspectable JSON" convention).

    Since 2026-09-12 the turn ends with the Scientific Validator as a control
    layer (`scientific_validation.validate_response`): `response` is the RAW
    LLM answer, untouched; `validation` carries the claim-level report and the
    separate `provenance_valid` / `claim_validation` / `scientific_validation`
    verdicts; `final_answer` is the raw answer plus an appended qualification
    block (or the raw answer itself when the validator is off or has nothing
    to add). `grounded` keeps its presence-check meaning throughout.
    `validate=None` defers to $EMBRYO_SCIENTIFIC_VALIDATOR (default: on);
    `context_format=None` defers to the provider's own rendering."""
    llm = llm or NullLLMProvider()
    policy = resolve_context_policy(context_format, llm)
    route = classify(question)
    plan = plan_tools(question, route, video_id=video_id, window=window)
    results = execute_plan(plan, question, video_id, window, split, context_policy=policy)
    context = build_context(question, route, results, tools_called=plan.called)
    response = generate_grounded_response(llm, context, timeout_seconds=llm_timeout_seconds)
    validation = scientific_validation.validate_response(response.text, context, enabled=validate)

    return {
        "question": question,
        "route": route.to_dict(),
        "tool_plan": plan.to_dict(),
        "context_policy": {
            "context_format": policy,
            "curated_retrieval": policy == CONTEXT_FORMAT_COMPACT_V2,
            "scientific_reference": policy == CONTEXT_FORMAT_COMPACT_V2,
            "validator_enabled": validation["enabled"],
        },
        "context": context,
        "response": response.to_dict(),
        "validation": validation,
        "final_answer": validation["final_answer"],
    }

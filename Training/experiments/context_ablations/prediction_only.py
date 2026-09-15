"""
PREDICTION-ONLY condition: the production-like context.

WHY THIS MODULE EXISTS
----------------------
`observed_only.py` answers "do the MODEL-DERIVED parts of the context degrade
the LLM's answers?" by removing the model. This module answers the *deployment*
question instead:

    In production there is no embryologist annotation. A video arrives, the
    frozen Semi-HMM produces a prediction, and the LLM must answer from that
    prediction plus the documentary corpus -- nothing else.

  FULL             OBSERVED + MODEL-DERIVED + DERIVED + RAG   (production-of-record, unchanged)
  OBSERVED-ONLY    OBSERVED + DERIVED-from-OBSERVED + RAG     (observed_only.py)
  PREDICTION-ONLY  MODEL-DERIVED + DERIVED-from-MODEL + RAG   (this module)

The annotation stays available to the EVALUATOR -- that is how the answer is
scored afterwards -- but it never reaches the provider. `filter_context()` takes
the real `context_builder.build_context()` output and returns a copy with every
annotation-bearing field removed.

MODEL-DERIVED IS NOT GROUND TRUTH
---------------------------------
Nothing here promotes a prediction to the status of an observation. A removed
OBSERVED field is REMOVED, never overwritten by its model counterpart:
`ground_truth_phase` does not become `current_phase`, and no field is renamed.
The scientific model's own errors must remain visible as errors when the
evaluator later compares MODEL-DERIVED against the annotation.

WHITELIST, NOT BLACKLIST
------------------------
Each tool's payload is rebuilt from an explicit list of KEPT keys; anything not
named is dropped. A future field added to a Reporting API schema is therefore
excluded by default rather than silently leaking the annotation into this
condition -- the conservative direction (over-removal can only hurt
PREDICTION-ONLY, never flatter it).

WHAT IS DROPPED, AND WHY (per tool)
-----------------------------------
get_current_inference  : `ground_truth` (ground_truth_phase, consistency_flag).
                         Everything the Semi-HMM produced is kept, including
                         `warnings`, which are statements about the model.
get_transition_events  : dropped WHOLESALE. Every field of a TransitionEvent is
                         annotation (`provenance="observed_annotation"`,
                         `observed=true`); `is_skip` and the phase pair are
                         properties of the annotated segmentation, not of any
                         prediction.
get_trajectory         : `transition_chain` dropped wholesale. Its `phase` and
                         `observed_duration_windows` come from the annotation's
                         own segments (see observed_only.py, which keeps exactly
                         those two as OBSERVED), and `duration_probability_at_observed`
                         is indexed on the observed duration. The segment
                         positions alone would rebuild the annotated phase
                         sequence, so no per-segment field survives.
get_inference_history  : `entries[].observed` dropped, then the WHOLE block
                         dropped -- `temporal_context._render_window_block()`
                         unconditionally prints an "OBSERVED (annotation du jeu
                         de donnees)" sub-block, so leaving the tool in place
                         would render an annotation-labelled section. Its
                         per-window MODEL-DERIVED content is carried instead by
                         `model_series_analysis` below.
series_analysis        : dropped wholesale and replaced by `model_series_analysis`,
                         a whitelisted copy. `render_series_analysis()` likewise
                         hardcodes a "Modele vs annotation" section and an
                         "OBSERVED observed_phase_sequence" line.

This mirrors exactly what `observed_only.py` does in the other direction, and
for the same reason: the renderers in `prompt.py` / `temporal_context.py` are
NOT modified by this experiment.

NO CONDITION MARKER
-------------------
Nothing is added to the context: no banner, no note, no instruction saying "you
have no annotation here". An instruction given to one condition and not to
another makes the conditions differ by more than the ablation. The shared,
frozen system prompt already carries the honest-refusal rule and is byte-identical
in every condition.

SCOPE DISCIPLINE
----------------
Nothing here modifies the dataset, the annotations, the scientific models, the
Reporting API, the RAG corpus/index, the router, the grounding checker, the
system prompt, the renderers or the 15 frozen questions. It is a pure, additive
function over an already-built context dict. The production path
(`orchestrator.answer_question()`) never imports it, and neither does
`observed_only.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

CONDITION_PREDICTION_ONLY = "PREDICTION_ONLY"

# Pre-declared per question type, never inferred from what the LLM said.
NOT_COMPUTABLE = "NON CALCULABLE DANS CETTE CONDITION"
COMPUTABLE = "CALCULABLE"
PARTIALLY_COMPUTABLE = "PARTIELLEMENT CALCULABLE"


# ---------------------------------------------------------------------------
# per-tool whitelists
# ---------------------------------------------------------------------------

# A tool absent from this map is dropped entirely -- that is how
# get_transition_events disappears.
_KEPT_TOP_LEVEL_KEYS: Dict[str, Tuple[str, ...]] = {
    # InferenceRecord minus its `ground_truth` block.
    "get_current_inference": ("sample", "window", "current_state", "next_phase",
                              "duration", "model", "warnings"),
    # TrajectorySummary: identity + window geometry only; transition_chain is
    # annotation-derived end to end.
    "get_trajectory": ("sample", "n_windows", "window_starts", "model"),
    # ModelInfo: pure model metadata.
    "get_model_metadata": ("model_name", "model_version", "model_configuration",
                           "model_source", "inference_timestamp", "n_states",
                           "duration_family", "dmax"),
    # Filtered here, then dropped by filter_dynamic_context (see module docstring).
    "get_inference_history": ("sample", "center_window", "requested_before", "requested_after",
                              "n_windows_before", "n_windows_after", "n_windows", "window_starts",
                              "entries", "model", "warnings"),
}

# One `entries` element of get_inference_history: identity, window timing, model.
_KEPT_HISTORY_ENTRY_KEYS = ("window_start", "offset_from_center", "is_center",
                            "window_start_time", "window_end_time", "time_available",
                            "model_derived")

# Top-level keys kept from `series_analysis` -> `model_series_analysis`.
# `timing` (all_observed_phase_changes, matched_change_pair,
# model_vs_observed_lag_windows) is dropped wholesale: every quantity in it is a
# comparison against the annotation. `is_observed_annotation` is dropped as a
# key whose name alone belongs to the removed provenance class.
_KEPT_SERIES_KEYS = ("provenance", "computed_from", "computed_by", "is_model_output",
                     "video_id", "split", "center_window", "n_windows", "window_starts",
                     "per_window", "sequences", "probability_gap", "convergence",
                     "stability", "entropy", "warnings")

# One `per_window` entry: everything except `observed_phase`.
_KEPT_PER_WINDOW_KEYS = ("model_phase", "model_phase_index", "phase_probability",
                         "second_phase", "second_phase_probability", "probability_gap",
                         "entropy", "next_phase", "next_phase_probability")

# `sequences`: the model's own sequence only. `observed_annotation_phase_sequence`
# and the `note` that describes it are both dropped.
_KEPT_SEQUENCE_KEYS = ("model_derived_phase_sequence",)


# Key names that must never appear anywhere in a PREDICTION-ONLY context, at any
# depth. Used by `assert_no_observed()` as an INDEPENDENT check on the filter's
# output -- not by the filter itself, so a bug in one cannot hide a bug in the
# other.
FORBIDDEN_KEYS = frozenset({
    # InferenceRecord.ground_truth
    "ground_truth", "ground_truth_phase", "consistency_flag",
    # InferenceHistoryEntry.observed and its derivatives
    "observed", "observed_phase", "observed_series_analysis",
    "observed_annotation_phase_sequence",
    # TransitionEvent (whole tool). The CONTAINERS are listed here, which is
    # what makes the guarantee structural: a TransitionEvent field cannot exist
    # without `transitions`, and `transitions` cannot exist without the tool.
    "get_transition_events", "transitions", "n_transitions",
    "is_skip", "frame_start", "frame_end", "frame_mapping_available",
    # TrajectorySummary.transition_chain
    "transition_chain", "observed_duration_windows", "duration_probability_at_observed",
    # temporal_context.derive_series_analysis -- the model-vs-annotation block
    "timing", "all_observed_phase_changes", "matched_change_pair",
    "model_vs_observed_lag_windows", "lag_definition", "is_observed_annotation",
})

# Key names that belong to BOTH provenance classes and can therefore only be
# judged by their path. `from_phase`/`to_phase` name the endpoints of a phase
# change; the annotation's changes live in `get_transition_events.transitions`
# and `series_analysis.timing.all_observed_phase_changes` (both dropped
# wholesale, and both containers are in FORBIDDEN_KEYS), while the MODEL's own
# changes live in `convergence.all_model_phase_changes`, which this condition
# keeps by design. Forbidden everywhere EXCEPT under that one container.
_PATH_SCOPED_KEYS = frozenset({
    "from_phase", "to_phase", "from_phase_index", "to_phase_index",
    "from_window", "to_window", "phase_distance",
})
_PATH_SCOPED_ALLOWED_PARENT = "all_model_phase_changes"


# String VALUES that would betray the annotation's provenance even under a
# renamed key. Checked case-insensitively against every string in the structure.
FORBIDDEN_VALUE_MARKERS = ("observed_annotation", "ground_truth")

# Substrings that must not appear in the RENDERED dynamic-context section of the
# user prompt. Deliberately restricted to that section: the frozen system prompt
# and the RAG document context legitimately discuss annotations as a concept,
# and neither carries a value for this video.
FORBIDDEN_RENDERED_MARKERS = (
    "ground_truth", "observed_phase", "consistency_flag", "observed_annotation",
    "observed_duration", "is_skip", "phase_distance",
    "OBSERVED (annotation du jeu de donnees",
    "observed_phase_sequence", "all_observed_phase_changes",
    "model_vs_observed_lag", "transition_chain", "n_transitions",
)


class ObservedLeakError(AssertionError):
    """Raised when annotation-bearing content survives the filter."""


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def _filter_tool_payload(tool_name: str, value: Any, removed: List[str]) -> Optional[Any]:
    """Whitelisted copy of one tool's payload, or None when the whole tool is
    dropped. Appends a dotted path to `removed` for every key dropped -- that
    list IS the experiment's removal evidence."""
    kept_keys = _KEPT_TOP_LEVEL_KEYS.get(tool_name)
    if kept_keys is None or not isinstance(value, dict):
        removed.append(tool_name)
        return None

    out: Dict[str, Any] = {}
    for key, sub_value in value.items():
        if key not in kept_keys:
            removed.append(f"{tool_name}.{key}")
            continue
        out[key] = sub_value

    if tool_name == "get_inference_history" and isinstance(out.get("entries"), list):
        entries = []
        for index, entry in enumerate(out["entries"]):
            if not isinstance(entry, dict):
                removed.append(f"{tool_name}.entries[{index}]")
                continue
            kept = {}
            for key, sub_value in entry.items():
                if key in _KEPT_HISTORY_ENTRY_KEYS:
                    kept[key] = sub_value
                else:
                    removed.append(f"{tool_name}.entries[{index}].{key}")
            entries.append(kept)
        out["entries"] = entries

    return out


def derive_model_series_analysis(series_analysis: Optional[Dict],
                                 removed: List[str]) -> Dict[str, Any]:
    """The PREDICTION-ONLY counterpart of the production `series_analysis`
    block. This is a WHITELISTED COPY of the block the production pipeline
    already computed -- never a re-derivation -- so the model-side arithmetic
    cannot drift between conditions. Returns {} when there is no series."""
    if not isinstance(series_analysis, dict) or not series_analysis:
        return {}

    out: Dict[str, Any] = {}
    for key, value in series_analysis.items():
        if key not in _KEPT_SERIES_KEYS:
            removed.append(f"series_analysis.{key}")
            continue
        out[key] = value

    per_window = out.get("per_window")
    if isinstance(per_window, dict):
        filtered_windows: Dict[str, Any] = {}
        for window, entry in per_window.items():
            if not isinstance(entry, dict):
                removed.append(f"series_analysis.per_window.{window}")
                continue
            kept = {}
            for key, value in entry.items():
                if key in _KEPT_PER_WINDOW_KEYS:
                    kept[key] = value
                else:
                    removed.append(f"series_analysis.per_window.{window}.{key}")
            filtered_windows[window] = kept
        out["per_window"] = filtered_windows

    sequences = out.get("sequences")
    if isinstance(sequences, dict):
        kept_sequences = {}
        for key, value in sequences.items():
            if key in _KEPT_SEQUENCE_KEYS:
                kept_sequences[key] = value
            else:
                removed.append(f"series_analysis.sequences.{key}")
        out["sequences"] = kept_sequences

    return out


def filter_dynamic_context(dynamic_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """(filtered_dynamic_context, removed_paths). Pure: the input is never
    mutated. Kept values are the SAME objects the input holds, so a MODEL-DERIVED
    value cannot drift between conditions."""
    filtered: Dict[str, Any] = {}
    removed: List[str] = []
    series_source = None

    for tool_name, value in (dynamic_context or {}).items():
        if tool_name == "series_analysis":
            series_source = value
            removed.append("series_analysis (whole block; replaced by model_series_analysis)")
            continue
        payload = _filter_tool_payload(tool_name, value, removed)
        if payload is not None:
            filtered[tool_name] = payload

    model_analysis = derive_model_series_analysis(series_source, removed)
    if model_analysis:
        filtered["model_series_analysis"] = model_analysis

    if "get_inference_history" in filtered:
        # Its model content is now carried, window-keyed, by
        # model_series_analysis; the key itself is dropped so the prompt's
        # window renderer -- which always prints an OBSERVED sub-block -- is
        # never reached in this condition.
        del filtered["get_inference_history"]
        removed.append("get_inference_history (whole block; model content lifted into "
                       "model_series_analysis)")

    return filtered, removed


def filter_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """FULL context -> PREDICTION-ONLY context. Returns a new dict; the input is
    never mutated. `document_context` (RAG), `question`, `route` and the
    top-level `warnings` are carried over unchanged -- only `dynamic_context`
    is filtered, and nothing is ever ADDED to it.

    The result additionally carries `prediction_only_removed_paths`, the full
    list of dropped fields."""
    filtered_dynamic, removed = filter_dynamic_context(context.get("dynamic_context") or {})

    out = dict(context)
    out["dynamic_context"] = filtered_dynamic
    out["prediction_only_removed_paths"] = removed
    out["condition"] = CONDITION_PREDICTION_ONLY
    return out


# ---------------------------------------------------------------------------
# independent verification
# ---------------------------------------------------------------------------

def find_observed_keys(value: Any, path: str = "") -> List[str]:
    """Every dotted path in `value` whose key name is in FORBIDDEN_KEYS, plus
    every path-scoped key (see `_PATH_SCOPED_KEYS`) found outside the one
    model-derived container that is allowed to carry it."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, sub_value in value.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_KEYS:
                found.append(sub_path)
            elif key in _PATH_SCOPED_KEYS and _PATH_SCOPED_ALLOWED_PARENT not in sub_path:
                found.append(sub_path)
            found.extend(find_observed_keys(sub_value, sub_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_observed_keys(item, f"{path}[{index}]"))
    return found


def find_observed_value_markers(value: Any, path: str = "") -> List[str]:
    """Every dotted path whose STRING value contains a provenance marker of the
    annotation class -- catches a leak that survived under a renamed key."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, sub_value in value.items():
            found.extend(find_observed_value_markers(sub_value, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_observed_value_markers(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        low = value.lower()
        for marker in FORBIDDEN_VALUE_MARKERS:
            if marker in low:
                found.append(f"{path} :: {marker}")
    return found


def find_rendered_markers(rendered_dynamic_section: str) -> List[str]:
    """Every FORBIDDEN_RENDERED_MARKERS substring present in the rendered
    dynamic-context text -- the last line of defence, run on the string that is
    actually built for the provider rather than on the structure."""
    return [m for m in FORBIDDEN_RENDERED_MARKERS if m.lower() in rendered_dynamic_section.lower()]


def find_ground_truth_fingerprints(filtered_context: Dict[str, Any],
                                   full_context: Dict[str, Any]) -> List[str]:
    """Checks that the annotation OBJECTS taken from the FULL context do not
    survive, by value, anywhere in the filtered structure. Catches a leak that
    passed both the key scan and the marker scan because the payload was moved
    under an innocuous name."""
    full_dynamic = full_context.get("dynamic_context") or {}
    fingerprints: List[Tuple[str, Any]] = []

    current = full_dynamic.get("get_current_inference") or {}
    if isinstance(current, dict) and current.get("ground_truth"):
        fingerprints.append(("get_current_inference.ground_truth", current["ground_truth"]))

    events = full_dynamic.get("get_transition_events") or {}
    if isinstance(events, dict) and events.get("transitions"):
        fingerprints.append(("get_transition_events.transitions", events["transitions"]))

    series = full_dynamic.get("series_analysis") or {}
    if isinstance(series, dict):
        sequences = series.get("sequences") or {}
        if sequences.get("observed_annotation_phase_sequence"):
            fingerprints.append(("series_analysis.sequences.observed_annotation_phase_sequence",
                                 sequences["observed_annotation_phase_sequence"]))
        timing = series.get("timing") or {}
        if timing.get("all_observed_phase_changes"):
            fingerprints.append(("series_analysis.timing.all_observed_phase_changes",
                                 timing["all_observed_phase_changes"]))

    trajectory = full_dynamic.get("get_trajectory") or {}
    if isinstance(trajectory, dict) and trajectory.get("transition_chain"):
        fingerprints.append(("get_trajectory.transition_chain", trajectory["transition_chain"]))

    found: List[str] = []
    for name, payload in fingerprints:
        if _contains_value(filtered_context.get("dynamic_context"), payload):
            found.append(name)
    return found


def _contains_value(haystack: Any, needle: Any) -> bool:
    if haystack == needle:
        return True
    if isinstance(haystack, dict):
        return any(_contains_value(v, needle) for v in haystack.values())
    if isinstance(haystack, list):
        return any(_contains_value(v, needle) for v in haystack)
    return False


def assert_no_observed(filtered_context: Dict[str, Any],
                       full_context: Optional[Dict[str, Any]] = None,
                       rendered_dynamic_section: Optional[str] = None) -> None:
    """Raises ObservedLeakError if ANY of the four independent checks finds
    annotation content. Called before a single token is generated."""
    dynamic = filtered_context.get("dynamic_context") or {}

    offenders = find_observed_keys(dynamic)
    if offenders:
        raise ObservedLeakError(f"OBSERVED key(s) survived the filter: {offenders}")

    markers = find_observed_value_markers(dynamic)
    if markers:
        raise ObservedLeakError(f"OBSERVED provenance marker(s) in a value: {markers}")

    if rendered_dynamic_section is not None:
        rendered = find_rendered_markers(rendered_dynamic_section)
        if rendered:
            raise ObservedLeakError(f"OBSERVED marker(s) in the rendered context: {rendered}")

    if full_context is not None:
        fingerprints = find_ground_truth_fingerprints(filtered_context, full_context)
        if fingerprints:
            raise ObservedLeakError(f"annotation payload(s) survived by value: {fingerprints}")


# ---------------------------------------------------------------------------
# evaluator-side ground truth (NEVER sent to the provider)
# ---------------------------------------------------------------------------

def evaluator_ground_truth(full_context: Dict[str, Any]) -> Dict[str, Any]:
    """The annotation, projected out of the FULL context for the EVALUATOR.

    This is what the later analysis compares the model's prediction against. It
    is stored in the artefact under its own explicitly-named block and is never
    part of any context passed to `build_user_prompt()` or to the provider."""
    full_dynamic = full_context.get("dynamic_context") or {}
    current = full_dynamic.get("get_current_inference") or {}
    events = full_dynamic.get("get_transition_events") or {}
    series = full_dynamic.get("series_analysis") or {}
    sequences = series.get("sequences") or {}
    timing = series.get("timing") or {}
    trajectory = full_dynamic.get("get_trajectory") or {}

    return {
        "NOTE": "EVALUATOR-ONLY. Never rendered into a prompt, never sent to the provider.",
        "ground_truth_at_center": current.get("ground_truth"),
        "observed_transitions": events.get("transitions"),
        "n_observed_transitions": events.get("n_transitions"),
        "observed_annotation_phase_sequence": sequences.get("observed_annotation_phase_sequence"),
        "all_observed_phase_changes": timing.get("all_observed_phase_changes"),
        "model_vs_observed_lag_windows": timing.get("model_vs_observed_lag_windows"),
        "observed_transition_chain": trajectory.get("transition_chain"),
    }


# ---------------------------------------------------------------------------
# computability
# ---------------------------------------------------------------------------

# Mirror of observed_only._COMPUTABILITY_BY_TYPE, inverted:
#   OBSERVED      -> NOT_COMPUTABLE   the answer IS a property of the annotation
#   HYBRID        -> PARTIALLY        the model/documentary half survives,
#                                      the annotation half does not
#   MODEL-DERIVED -> COMPUTABLE       the prediction carries the answer
#   DOCUMENTARY   -> COMPUTABLE       the corpus carries the answer
#   UNAVAILABLE   -> COMPUTABLE       the correct answer is an honest refusal
#                                      in EVERY condition, unchanged here
_COMPUTABILITY_BY_TYPE = {
    "OBSERVED": NOT_COMPUTABLE,
    "HYBRID": PARTIALLY_COMPUTABLE,
    "MODEL-DERIVED": COMPUTABLE,
    "DOCUMENTARY": COMPUTABLE,
    "UNAVAILABLE": COMPUTABLE,
}


def computability_of(question_type: str) -> str:
    """The pre-declared status of a frozen question under PREDICTION-ONLY.
    Deterministic, decided before any answer is produced, so it can never be
    fitted to what the LLM happened to say."""
    return _COMPUTABILITY_BY_TYPE.get(question_type, PARTIALLY_COMPUTABLE)

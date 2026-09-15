"""
OBSERVED-ONLY condition for the paired context-ablation experiment
(`run_observed_only_experiment.py`).

WHY THIS MODULE EXISTS
----------------------
Diagnostic question: do the MODEL-DERIVED parts of the context degrade the
LLM's answers? The comparison is between two conditions over the SAME frozen
Q1-Q15 benchmark, the SAME anchor (Patient_319 / val / window 156), the SAME
model and the SAME prompt:

  FULL           OBSERVED + MODEL-DERIVED + DERIVED + RAG   (production, unchanged)
  OBSERVED-ONLY  OBSERVED + DERIVED-from-OBSERVED-only + RAG

This module builds the second one, and ONLY by SUBTRACTION from the first:
`filter_context()` takes the real `context_builder.build_context()` output and
returns a copy with every model-derived field removed. Three properties are
guaranteed by construction and pinned by `Tests/orchestrator/test_observed_only.py`:

  1. Every OBSERVED value is byte-identical between the two conditions -- the
     filtered structure holds the SAME objects the full context holds, never a
     recomputed or re-fetched copy.
  2. No model-derived value survives anywhere in the returned structure, at any
     depth (`assert_no_model_derived()` re-verifies this independently of the
     filter, by scanning the result).
  3. A model-derived field is REMOVED, never overwritten with its observed
     counterpart. `model_phase` does not become the annotation; the key is gone
     and the annotation stays under its own `observed_*` name. Filling the same
     field with a different provenance is exactly the confusion the whole
     project is built to prevent.
  4. The filter only SUBTRACTS. It adds no marker, no note and no instruction to
     the OBSERVED-ONLY context, so the two arms differ by the ablation and by
     nothing else. The system prompt -- which already states the honest-refusal
     rule -- is byte-identical for both arms.

SCOPE DISCIPLINE
----------------
Nothing here modifies the dataset, the annotations, the scientific models, the
Reporting API, the RAG corpus/index, the router, the grounding checker, the
system prompt or the 15 frozen questions. It is a pure, additive function over
an already-built context dict, composed by a separate experiment runner. The
production path (`orchestrator.answer_question()`) never imports it.

WHITELIST, NOT BLACKLIST
------------------------
Each tool's payload is rebuilt from an explicit list of KEPT keys. Anything not
named is dropped. A future field added to a Reporting API schema is therefore
excluded by default rather than silently leaking into the OBSERVED-ONLY arm --
the conservative direction for this experiment (over-removal can only hurt the
OBSERVED-ONLY condition, never flatter it).

WHAT IS DROPPED, AND WHY (per tool)
-----------------------------------
get_current_inference : current_state (posterior, argmax phase, probability,
                        entropy), next_phase, duration (the fitted duration
                        model's expectation/quantiles), model (ModelInfo).
                        `warnings` is dropped too -- "duration not estimable
                        (structural censoring)" is a statement about the
                        model's duration model, so the list is removed wholesale
                        rather than string-filtered. Documented cost: the
                        honest non-model caveats in that list (incomplete time
                        metadata, repeated timestamps) are lost in this arm.
get_trajectory        : transition_chain is reduced to the two annotation-only
                        fields (`phase`, `observed_duration_windows`);
                        `duration_probability_at_observed` (the model's
                        likelihood of the observed duration) and
                        `next_phase_distribution` (the model's transition
                        matrix row) are dropped, as is `model`.
get_transition_events : kept ENTIRELY. `TransitionEvent` carries no
                        model-derived field by construction (schemas.py) --
                        every entry is `observed=True`,
                        `provenance="observed_annotation"`.
get_inference_history : dropped from the context ENTIRELY, after its
                        annotation half has been lifted into
                        `observed_series_analysis` (below). Keeping the key
                        with its `model_derived` blocks removed was tried and
                        rejected: `prompt._format_dynamic_context()` routes
                        that key to `temporal_context.render_inference_history`,
                        whose window block prints a MODEL-DERIVED section --
                        so the arm would still show `W151.phase_probability =
                        null`, i.e. the names of the model's fields. No value
                        leaked, but the condition is "no model-derived
                        information", not "model-derived fields set to null".
                        The annotation loses nothing: `observed_series_analysis.
                        per_window` carries every window's annotation, its
                        offset and its times, keyed BY THE WINDOW ID (not by an
                        array index), so the OBSERVED-ONLY arm keeps the
                        window-addressed readability the FULL arm gets from its
                        own renderer.
series_analysis       : the whole block is dropped and REPLACED by
                        `observed_series_analysis` (below), which is recomputed
                        from the OBSERVED annotation sequence alone. Nothing of
                        the model's gap/convergence/stability/entropy/lag
                        analysis survives.
get_model_metadata    : dropped entirely. (Never planned for any of Q1-Q15 by
                        `plan_tools`, so this is a defensive rule, not an
                        observed code path.)
document_context      : untouched -- the RAG half of the context is identical
                        in both conditions, which is the point of the design.

DERIVED-FROM-OBSERVED
---------------------
`derive_observed_series_analysis()` is the OBSERVED-ONLY counterpart of
`temporal_context.derive_series_analysis()`: same deterministic-arithmetic
idea, but it reads ONLY each entry's `observed` block. It reuses
`temporal_context._changes` rather than reimplementing consecutive-change
detection, so the two conditions cannot diverge on what "a phase change"
means. It computes nothing that would require a prediction: no gap, no
convergence, no entropy, no model-vs-annotation lag.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from orchestrator import temporal_context
from orchestrator.fixed_question_benchmark import (
    TYPE_DOCUMENTARY,
    TYPE_HYBRID,
    TYPE_MODEL_DERIVED,
    TYPE_OBSERVED,
    TYPE_UNAVAILABLE,
)

CONDITION_FULL = "FULL"
CONDITION_OBSERVED_ONLY = "OBSERVED_ONLY"

# The string the experiment records for a question whose answer intrinsically
# requires a model prediction. Pre-declared per question (see
# `computability_of()`), never inferred from what the LLM happened to say.
NOT_COMPUTABLE = "NON CALCULABLE DANS CETTE CONDITION"
COMPUTABLE = "CALCULABLE"
PARTIALLY_COMPUTABLE = "PARTIELLEMENT CALCULABLE"

PROVENANCE_OBSERVED_DERIVED = "derived_from_observed_only"


# ---------------------------------------------------------------------------
# per-tool whitelists
# ---------------------------------------------------------------------------

# Top-level keys kept from each tool's payload. A tool absent from this map is
# dropped entirely.
_KEPT_TOP_LEVEL_KEYS: Dict[str, Tuple[str, ...]] = {
    # InferenceRecord: identity + window timing + the annotation. Everything the
    # frozen Semi-HMM produced for this window is gone.
    "get_current_inference": ("sample", "window", "ground_truth"),
    # TrajectorySummary: identity + window list; transition_chain handled below.
    "get_trajectory": ("sample", "n_windows", "window_starts", "transition_chain"),
    # TransitionEvent list -- structurally observed-only, kept whole.
    "get_transition_events": ("video_id", "split", "window", "n_transitions", "transitions"),
    # InferenceHistory: identity + band geometry; entries handled below.
    "get_inference_history": ("sample", "center_window", "requested_before", "requested_after",
                              "n_windows_before", "n_windows_after", "n_windows", "window_starts",
                              "entries"),
}

# One `transition_chain` element: the two fields that come from the annotation's
# own segments (HMMModel._segments over ground truth), nothing else.
_KEPT_TRANSITION_CHAIN_KEYS = ("phase", "observed_duration_windows")

# One `entries` element of get_inference_history: identity, timing, annotation.
_KEPT_HISTORY_ENTRY_KEYS = ("window_start", "offset_from_center", "is_center",
                            "window_start_time", "window_end_time", "time_available", "observed")

# Key names that must never appear anywhere in an OBSERVED-ONLY context, at any
# depth. Used by `assert_no_model_derived()` as an INDEPENDENT check on the
# filter's output -- not by the filter itself, so a bug in one cannot hide a bug
# in the other.
FORBIDDEN_KEYS = frozenset({
    # CurrentState / InferenceHistoryEntry.model_derived
    "model_derived", "current_state", "current_phase", "current_phase_index",
    "phase_probability", "phase_probabilities", "entropy",
    "most_likely_next_phase", "next_phase", "next_phase_probability", "next_phase_distribution",
    # DurationInfo
    "duration", "duration_available", "expected_duration", "duration_quantiles",
    "duration_probability_at_observed",
    # ModelInfo
    "model", "model_name", "model_version", "model_configuration", "model_source",
    "inference_timestamp",
    # temporal_context.derive_series_analysis
    "model_phase", "model_phase_index", "second_phase", "second_phase_probability",
    "probability_gap", "convergence", "crossing_window", "convergence_start_window",
    "all_model_phase_changes", "stability", "phase_returns", "phase_skips", "phase_regressions",
    "is_stable_after_center", "center_model_phase", "model_derived_phase_sequence",
    "model_vs_observed_lag_windows", "matched_change_pair",
})


# ---------------------------------------------------------------------------
# DERIVED-from-OBSERVED
# ---------------------------------------------------------------------------

def _observed_view(entry: Dict) -> Dict:
    """One already-FILTERED history entry -> a flat view holding only
    annotation and window identity. Reads `observed` and nothing else, so it
    cannot see a model field even if one were still present."""
    observed = entry.get("observed", {}) or {}
    return {
        "window_start": entry.get("window_start"),
        "offset_from_center": entry.get("offset_from_center"),
        "is_center": bool(entry.get("is_center")),
        "window_start_time": entry.get("window_start_time"),
        "window_end_time": entry.get("window_end_time"),
        "observed_phase": observed.get("ground_truth_phase"),
        "consistency_flag": observed.get("consistency_flag"),
    }


def derive_observed_series_analysis(filtered_history: Optional[Dict]) -> Dict:
    """The OBSERVED-ONLY counterpart of
    `temporal_context.derive_series_analysis()`. Pure function of the FILTERED
    history (annotation fields only); returns {} for an empty band rather than
    inventing one.

    Computes only what the annotation alone supports: the observed phase
    sequence over the band, its consecutive changes, the observed transition
    flag series, and the annotation's phase at the center window. It computes
    NO probability gap, NO convergence window, NO entropy, NO stability count
    over a predicted sequence and NO model-vs-annotation lag -- those are
    properties of a prediction, and reconstructing a stand-in for them from the
    annotation is exactly what this experiment forbids."""
    entries = (filtered_history or {}).get("entries") or []
    if not entries:
        return {}

    views = [_observed_view(e) for e in entries]
    views.sort(key=lambda v: v["window_start"])
    center = next((v for v in views if v["is_center"]), None)

    # Reused, never reimplemented: the same consecutive-change definition the
    # FULL condition uses, applied here to the annotation series.
    observed_changes = temporal_context._changes(views, "observed_phase")

    return {
        "provenance": PROVENANCE_OBSERVED_DERIVED,
        "computed_from": "get_inference_history.entries[].observed (annotation only)",
        "computed_by": "orchestrator.observed_only.derive_observed_series_analysis",
        "is_model_output": False,
        "is_observed_annotation": False,
        "note": "Calcul deterministe Python portant UNIQUEMENT sur l'annotation du jeu de "
                "donnees. Aucune sortie de modele n'entre dans ces valeurs.",
        "video_id": ((filtered_history or {}).get("sample") or {}).get("video_name"),
        "split": ((filtered_history or {}).get("sample") or {}).get("split"),
        "center_window": (filtered_history or {}).get("center_window"),
        "n_windows": len(views),
        "window_starts": [v["window_start"] for v in views],
        "per_window": {
            str(v["window_start"]): {
                "observed_phase": v["observed_phase"],
                "consistency_flag": v["consistency_flag"],
                "offset_from_center": v["offset_from_center"],
                "window_start_time": v["window_start_time"],
                "window_end_time": v["window_end_time"],
            }
            for v in views
        },
        "observed_annotation_phase_sequence": {str(v["window_start"]): v["observed_phase"]
                                                for v in views},
        "all_observed_phase_changes": observed_changes,
        "n_observed_phase_changes": len(observed_changes),
        "observed_phase_at_center_window": center["observed_phase"] if center else None,
        "observed_consistency_flag_series": {str(v["window_start"]): v["consistency_flag"]
                                              for v in views},
        "definitions": {
            "observed_phase": "phase annotee par les embryologistes pour cette fenetre "
                               "(ground_truth_phase). Jamais une prediction.",
            "consistency_flag": "indicateur d'annotation 0/1 : 1 = la phase change entre le debut "
                                 "et la fin de la fenetre annotee.",
            "all_observed_phase_changes": "changements de phase consecutifs DANS L'ANNOTATION, "
                                           "chacun avec ses deux identifiants de fenetre.",
            "provenance_of_this_whole_block": "DERIVED-from-OBSERVED. Aucune valeur ici ne depend "
                                               "d'une prediction du modele.",
        },
    }


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------

def _filter_tool_payload(tool_name: str, value: Any, removed: List[str]) -> Optional[Any]:
    """Returns the whitelisted copy of one tool's payload, or None when the
    whole tool is dropped. Appends a dotted path to `removed` for every key it
    drops -- that list IS the experiment's removal evidence."""
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

    if tool_name == "get_trajectory" and isinstance(out.get("transition_chain"), list):
        chain = []
        for index, segment in enumerate(out["transition_chain"]):
            if not isinstance(segment, dict):
                removed.append(f"{tool_name}.transition_chain[{index}]")
                continue
            kept = {}
            for key, sub_value in segment.items():
                if key in _KEPT_TRANSITION_CHAIN_KEYS:
                    kept[key] = sub_value
                else:
                    removed.append(f"{tool_name}.transition_chain[{index}].{key}")
            chain.append(kept)
        out["transition_chain"] = chain

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


def filter_dynamic_context(dynamic_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """(filtered_dynamic_context, removed_paths). Pure: the input is never
    mutated. Kept values are the SAME objects the input holds, so an OBSERVED
    value cannot drift between the two conditions."""
    filtered: Dict[str, Any] = {}
    removed: List[str] = []

    for tool_name, value in (dynamic_context or {}).items():
        if tool_name == "series_analysis":
            # Dropped wholesale and replaced below by the observed-only
            # recomputation -- never partially reused.
            removed.append("series_analysis (whole block; replaced by observed_series_analysis)")
            continue
        payload = _filter_tool_payload(tool_name, value, removed)
        if payload is not None:
            filtered[tool_name] = payload

    observed_analysis = derive_observed_series_analysis(filtered.get("get_inference_history"))
    if observed_analysis:
        filtered["observed_series_analysis"] = observed_analysis
    if "get_inference_history" in filtered:
        # Its annotation is now carried, window-keyed, by observed_series_analysis;
        # the key itself is dropped so the prompt's window renderer -- which always
        # prints a MODEL-DERIVED section -- is never reached in this condition.
        del filtered["get_inference_history"]
        removed.append("get_inference_history (whole block; annotation lifted into "
                       "observed_series_analysis)")

    return filtered, removed


# NO CONDITION MARKER.
#
# An earlier draft inserted an `experimental_condition` block at the top of the
# OBSERVED-ONLY dynamic context, stating that no model output was present and
# telling the model to say so rather than reconstruct a prediction from the
# annotation. It was REMOVED, with no equivalent text put in its place, and it
# must not come back: its last sentence was an instruction, and an instruction
# given to arm B and not to arm A makes the two arms differ by more than the
# ablation. This experiment measures the effect of removing MODEL-DERIVED
# CONTENT, not the effect of an extra instruction.
#
# Nothing is lost by removing it: the shared system prompt already carries the
# grounding rules and the honest-refusal rule (A.6 "une valeur explicitement
# null signifie indisponible / non calculable", A.10, C.2), and it is identical
# for both arms. `Tests/orchestrator/test_observed_only.py` pins the absence of
# any B-only instruction.


def filter_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """FULL context -> OBSERVED-ONLY context. Returns a new dict; the input is
    never mutated. `document_context` (RAG), `question`, `route` and the
    top-level `warnings` are carried over unchanged -- only `dynamic_context`
    is filtered, and nothing is ever ADDED to it (see the note above).

    The result additionally carries `observed_only_removed_paths`, the full
    list of dropped fields, so the artefact can show exactly what was taken
    away without re-deriving it."""
    filtered_dynamic, removed = filter_dynamic_context(context.get("dynamic_context") or {})

    out = dict(context)
    out["dynamic_context"] = filtered_dynamic
    out["model_metadata"] = None
    out["observed_only_removed_paths"] = removed
    out["condition"] = CONDITION_OBSERVED_ONLY
    return out


# ---------------------------------------------------------------------------
# independent verification
# ---------------------------------------------------------------------------

def find_model_derived_keys(value: Any, path: str = "") -> List[str]:
    """Every dotted path in `value` whose key name is in FORBIDDEN_KEYS.
    Walks dicts and lists to any depth. Independent of the filter: it re-reads
    the produced structure rather than trusting the whitelist that built it."""
    found: List[str] = []
    if isinstance(value, dict):
        for key, sub_value in value.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_KEYS:
                found.append(sub_path)
            found.extend(find_model_derived_keys(sub_value, sub_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_model_derived_keys(item, f"{path}[{index}]"))
    return found


def assert_no_model_derived(context: Dict[str, Any]) -> None:
    """Raises AssertionError naming every offending path. Called by the runner
    BEFORE any LLM call, so a leak stops the experiment instead of quietly
    contaminating a condition."""
    offenders = find_model_derived_keys(context.get("dynamic_context") or {})
    if offenders:
        raise AssertionError(
            "OBSERVED-ONLY context still carries model-derived field(s): " + ", ".join(offenders)
        )


def observed_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    """The OBSERVED-only projection of ANY context (full or filtered), used to
    prove the two conditions carry the same annotation. Deep-copied so a caller
    comparing two projections cannot mutate either context."""
    dynamic = context.get("dynamic_context") or {}
    out: Dict[str, Any] = {}

    events = dynamic.get("get_transition_events")
    if events is not None:
        out["get_transition_events"] = copy.deepcopy(events)

    inference = dynamic.get("get_current_inference")
    if isinstance(inference, dict) and "ground_truth" in inference:
        out["get_current_inference.ground_truth"] = copy.deepcopy(inference["ground_truth"])

    trajectory = dynamic.get("get_trajectory")
    if isinstance(trajectory, dict) and isinstance(trajectory.get("transition_chain"), list):
        out["get_trajectory.transition_chain.observed"] = [
            {k: v for k, v in segment.items() if k in _KEPT_TRANSITION_CHAIN_KEYS}
            for segment in trajectory["transition_chain"] if isinstance(segment, dict)
        ]

    # The per-window annotation, normalized to ONE canonical shape so the two
    # conditions can be compared even though they carry it under different keys
    # (FULL: get_inference_history.entries[].observed; OBSERVED-ONLY:
    # observed_series_analysis.per_window). The values compared are the same
    # annotation either way -- that is exactly what this projection must prove.
    per_window: Dict[str, Dict[str, Any]] = {}
    history = dynamic.get("get_inference_history")
    if isinstance(history, dict) and isinstance(history.get("entries"), list):
        for entry in history["entries"]:
            if not isinstance(entry, dict):
                continue
            observed = entry.get("observed") or {}
            per_window[str(entry.get("window_start"))] = {
                "ground_truth_phase": observed.get("ground_truth_phase"),
                "consistency_flag": observed.get("consistency_flag"),
            }
    analysis = dynamic.get("observed_series_analysis")
    if not per_window and isinstance(analysis, dict):
        for window, values in (analysis.get("per_window") or {}).items():
            per_window[str(window)] = {
                "ground_truth_phase": values.get("observed_phase"),
                "consistency_flag": values.get("consistency_flag"),
            }
    if per_window:
        out["per_window_annotation"] = per_window

    return out


# ---------------------------------------------------------------------------
# per-question computability in this condition
# ---------------------------------------------------------------------------

# Mechanically derived from each frozen question's own `type` field -- this
# module never re-types a question and never edits the benchmark.
#
#   MODEL-DERIVED -> NOT_COMPUTABLE     the answer IS a property of a prediction
#   HYBRID        -> PARTIALLY          the observed/documentary half survives,
#                                        the model half does not
#   OBSERVED      -> COMPUTABLE         the annotation carries the answer
#   DOCUMENTARY   -> COMPUTABLE         the corpus carries the answer
#   UNAVAILABLE   -> COMPUTABLE         the correct answer is an honest refusal
#                                        in BOTH conditions, unchanged here
_COMPUTABILITY_BY_TYPE = {
    TYPE_MODEL_DERIVED: NOT_COMPUTABLE,
    TYPE_HYBRID: PARTIALLY_COMPUTABLE,
    TYPE_OBSERVED: COMPUTABLE,
    TYPE_DOCUMENTARY: COMPUTABLE,
    TYPE_UNAVAILABLE: COMPUTABLE,
}


def computability_of(question_type: str) -> str:
    """The pre-declared status of a frozen question under OBSERVED-ONLY.
    Deterministic, decided before any answer is produced, so it can never be
    fitted to what the LLM happened to say."""
    return _COMPUTABILITY_BY_TYPE.get(question_type, PARTIALLY_COMPUTABLE)

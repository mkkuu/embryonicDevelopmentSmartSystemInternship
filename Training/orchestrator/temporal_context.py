"""Structured multi-window temporal context (ETAPE 1-4 of the
post-benchmark context-restructuring session, 2026-09-02).

WHY THIS MODULE EXISTS
----------------------
The comparative LLM benchmark (llama3.2 1.0/15 vs mistral-nemo:12b
5.5/15, single-variable model swap, docs/RAG_LLM_MODEL_AB_TEST.md)
showed that the remaining mistral-nemo failures are NOT purely a model-
capacity problem -- they are a CONTEXT REPRESENTATION problem. The audit
of `prompt._format_dynamic_context()`'s output for Q1/Q4/Q5-Q10 on the
real Patient_319 / val / window 156 anchor found five concrete defects:

  A. ARRAY-INDEX ADDRESSING. Every multi-window value is addressed
     `get_inference_history.entries[3].model_derived.phase_probabilities.t6`
     -- the actual window identifier (154) appears on exactly ONE line of
     that entry's ~16. Reading "the probability of t6 at window 154"
     requires the reader to hold an index->window map across ~300 lines.
     This is the direct mechanical cause of Q1's "mélange d'informations
     appartenant à deux transitions différentes" and of the partial /
     mis-attributed series reads in Q5/Q6/Q8/Q9.

  B. TWO COEXISTING REPRESENTATIONS OF THE SAME WINDOW. For every
     multi-window question the plan calls BOTH get_current_inference
     (window 156, full 15-entry distributions) AND get_inference_history
     (window 156 again, top-4 truncated). The same quantity therefore
     appears twice, ~200 lines apart, with two different truncations.

  C. DEFERRAL SPLITS A WINDOW'S DATA. The priority-reordering fix
     (docs/RAG_PROMPT_REORDERING_AB_TEST.md) relocates bulky
     distributions to the end of the prompt. Correct for the single-
     window case it was designed for; for a series it separates a
     window's header from its own numbers.

  D. FOUR SIMILARLY-NAMED PROBABILITY FIELDS ON ADJACENT LINES
     (`phase_probability`, `phase_probabilities.<p>`,
     `next_phase_probability`, `next_phase_distribution.<p>`) with no
     semantic grouping -- Q4's observed field confusion.

  E. NOTHING IS PRE-COMPUTED. Q5 (when do the two probabilities
     converge), Q6 (how do they evolve), Q7 (model/annotation lag),
     Q8 (post-transition stability / returns), Q9 (skip / regression),
     Q10 (is uncertainty higher here than in stable periods) each
     require subtracting, comparing or arg-maxing across 11 windows
     whose numbers are scattered over hundreds of lines. Every one of
     those operations is exactly computable in Python.

WHAT THIS MODULE DOES
---------------------
Two pure, deterministic, dependency-free functions over ONE tool output
(`tools.get_inference_history()`'s dict -- Training/reporting/schemas.py's
`InferenceHistory.to_dict()`):

  `derive_series_analysis(history)` -- computes the series properties
      listed above. NEVER refits, never re-runs the model, never invents:
      every number is an arithmetic function of values already present in
      `history`. Emitted as a plain dict that the orchestrator puts into
      `context["dynamic_context"]["series_analysis"]`, so
      `grounding_check` sees these derived numbers as KNOWN values (a
      derived number the LLM restates must be verifiable, exactly like a
      tool-returned one).

  `render_inference_history(history)` / `render_series_analysis(analysis)`
      -- the WINDOW-blocked text form. Each window is one block headed by
      its real identifier (`WINDOW 156`), every value line inside it is
      additionally prefixed `W156.`, so no value can ever appear without
      its temporal identifier. Within a block, three PHYSICALLY SEPARATE,
      explicitly labelled provenance classes:

        MODEL-DERIVED : model_phase, phase_probability, second_phase,
                        second_phase_probability, entropy, next_phase,
                        next_phase_probability, the top-k distributions
        OBSERVED      : observed_phase, consistency_flag, annotation time
        DERIVED       : probability_gap (Python arithmetic, neither a
                        model output nor an annotation)

This mirrors, at the PROMPT layer, the separation `schemas.py` already
enforces at the DATA layer (`model_derived` / `observed` keys) -- the
benchmark showed that the data-layer separation alone did not survive
generic path flattening.

SCOPE DISCIPLINE
----------------
Nothing here touches the dataset, annotations, embeddings, the scientific
models, the Reporting API, the router, the 15 benchmark questions or the
scoring criteria. It is a pure presentation + deterministic-arithmetic
layer over an existing tool's existing output, added by composition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Provenance tag carried by derive_series_analysis()'s output. Deliberately
# NOT "observed_annotation" (transition_events') and NOT a model field --
# a third, distinct class, so an answer can never present a Python-computed
# quantity as either a measurement or a model prediction.
PROVENANCE_DERIVED = "derived_deterministic"

_TRUNCATION_TOLERANCE = 1e-3  # a top-k distribution whose entries sum to less
# than 1 - this is reported as truncated, with the omitted mass quantified


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """Same rendering as prompt._format_scalar -- a value restated verbatim
    from a line this module emits must still be findable by grounding_check
    (which stringifies leaves with plain str())."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ranked(distribution: Dict[str, float]) -> List[Tuple[str, float]]:
    """Descending by probability, ties broken by phase name so the order is
    deterministic across runs and platforms (never dict insertion order)."""
    return sorted((distribution or {}).items(), key=lambda kv: (-kv[1], kv[0]))


def _entry_view(entry: Dict) -> Dict:
    """One InferenceHistoryEntry.to_dict() -> a flat, window-keyed view.
    Reads only; every value is taken verbatim from the entry."""
    model = entry.get("model_derived", {}) or {}
    observed = entry.get("observed", {}) or {}
    ranked = _ranked(model.get("phase_probabilities", {}))
    second_phase = ranked[1][0] if len(ranked) > 1 else None
    second_probability = ranked[1][1] if len(ranked) > 1 else None
    top_probability = model.get("phase_probability")
    gap = None
    if top_probability is not None and second_probability is not None:
        gap = round(top_probability - second_probability, 6)
    mass = sum(p for _, p in ranked)
    return {
        "window_start": entry.get("window_start"),
        "offset_from_center": entry.get("offset_from_center"),
        "is_center": bool(entry.get("is_center")),
        "window_start_time": entry.get("window_start_time"),
        "window_end_time": entry.get("window_end_time"),
        "time_available": entry.get("time_available"),
        # --- MODEL-DERIVED ---
        "model_phase": model.get("current_phase"),
        "model_phase_index": model.get("current_phase_index"),
        "phase_probability": top_probability,
        "second_phase": second_phase,
        "second_phase_probability": second_probability,
        "entropy": model.get("entropy"),
        "next_phase": model.get("most_likely_next_phase"),
        "next_phase_probability": model.get("next_phase_probability"),
        "phase_probabilities": dict(ranked),
        "next_phase_distribution": dict(_ranked(model.get("next_phase_distribution", {}))),
        "phase_probabilities_truncated": bool(mass < 1.0 - _TRUNCATION_TOLERANCE),
        "phase_probabilities_omitted_mass": round(max(0.0, 1.0 - mass), 6),
        # --- OBSERVED (annotation) ---
        "observed_phase": observed.get("ground_truth_phase"),
        "consistency_flag": observed.get("consistency_flag"),
        # --- DERIVED (Python arithmetic) ---
        "probability_gap": gap,
    }


def _phase_index_map(views: List[Dict]) -> Dict[str, int]:
    """phase name -> canonical index, learned ONLY from the (current_phase,
    current_phase_index) pairs the Reporting API itself returned. Never a
    hardcoded phase list here -- the canonical order lives in the scientific
    layer, and duplicating it in the prompt layer is exactly the kind of
    second source of truth this project forbids."""
    return {v["model_phase"]: v["model_phase_index"] for v in views
            if v.get("model_phase") is not None and v.get("model_phase_index") is not None}


def _changes(views: List[Dict], key: str) -> List[Dict]:
    """Consecutive-window changes of `key`, each tagged with BOTH window
    identifiers -- never a bare 'it changed'."""
    out: List[Dict] = []
    for previous, current in zip(views, views[1:]):
        before, after = previous.get(key), current.get(key)
        if before is None or after is None or before == after:
            continue
        out.append({
            "from_window": previous["window_start"], "to_window": current["window_start"],
            "from_phase": before, "to_phase": after,
        })
    return out


def _mean(values: List[float]) -> Optional[float]:
    numbers = [v for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 6) if numbers else None


# ---------------------------------------------------------------------------
# ETAPE 3: the deterministic series analysis
# ---------------------------------------------------------------------------

def derive_series_analysis(history: Dict) -> Dict:
    """Every series property Q5/Q6/Q7/Q8/Q9/Q10 need that is exactly
    computable, computed here instead of asked of the LLM. Pure function of
    `history` (tools.get_inference_history()'s dict); no I/O, no randomness,
    no model call. Returns {} for an empty/absent band rather than
    fabricating one."""
    entries = (history or {}).get("entries") or []
    if not entries:
        return {}

    views = [_entry_view(e) for e in entries]
    views.sort(key=lambda v: v["window_start"])  # defensive: the tool already
    # returns ascending order; sorting here means a future change upstream can
    # never silently produce an out-of-order narrative.
    windows = [v["window_start"] for v in views]
    center_window = (history or {}).get("center_window")
    center = next((v for v in views if v["is_center"]), None)
    warnings: List[str] = []

    # --- per-window table (window id -> the values that window owns) ---
    per_window = {
        str(v["window_start"]): {
            "model_phase": v["model_phase"],
            "phase_probability": v["phase_probability"],
            "second_phase": v["second_phase"],
            "second_phase_probability": v["second_phase_probability"],
            "probability_gap": v["probability_gap"],
            "entropy": v["entropy"],
            "next_phase": v["next_phase"],
            "next_phase_probability": v["next_phase_probability"],
            "observed_phase": v["observed_phase"],
        }
        for v in views
    }

    if any(v["phase_probabilities_truncated"] for v in views):
        worst = max(v["phase_probabilities_omitted_mass"] for v in views)
        warnings.append(
            f"phase_probabilities are top-k truncated in this band (largest omitted "
            f"probability mass in any single window: {worst}); second_phase is the "
            f"runner-up AMONG THE RETAINED entries. entropy / phase_probability were "
            f"computed upstream from the FULL distribution and are unaffected."
        )

    # --- the two phase SEQUENCES, kept explicitly apart (Q9) ---
    #
    # Q9 asks about a skip/regression in the PREDICTED sequence. The same
    # windows also carry an ANNOTATION sequence, and get_transition_events
    # separately reports the annotation's own skip. Rendering the two
    # sequences side by side, each labelled by provenance, is what stops the
    # annotated skip from being read as the model's
    # (docs/TEMPORAL_CONTEXT_RESTRUCTURING.md sec 6). Both are already in
    # `per_window`; this block exists so neither can be quoted without the
    # other being visible.
    sequences = {
        "model_derived_phase_sequence": {str(v["window_start"]): v["model_phase"] for v in views},
        "observed_annotation_phase_sequence": {str(v["window_start"]): v["observed_phase"]
                                                for v in views},
        "note": "TWO DIFFERENT sequences over the SAME windows. The first is what the model "
                 "predicts; the second is the dataset annotation. A question about the PREDICTED "
                 "sequence (skip, regression, stability) must be answered from "
                 "model_derived_phase_sequence ONLY -- never from the annotation, and never from "
                 "get_transition_events (which reports OBSERVED annotation transitions only).",
    }

    # --- probability gap (Q5, Q6) ---
    gaps = [(v["window_start"], v["probability_gap"]) for v in views if v["probability_gap"] is not None]
    gap_block: Dict[str, Any] = {
        "definition": "probability_gap = phase_probability - second_phase_probability, per window",
        "series": {str(w): g for w, g in gaps},
    }
    if gaps:
        min_window, min_gap = min(gaps, key=lambda wg: (wg[1], wg[0]))
        max_window, max_gap = max(gaps, key=lambda wg: (wg[1], -wg[0]))
        gap_block.update({
            "minimum": min_gap, "minimum_window": min_window,
            "maximum": max_gap, "maximum_window": max_window,
            "at_center_window": center["probability_gap"] if center else None,
        })

    # --- convergence / crossing (Q5) ---
    leader_changes = _changes(views, "model_phase")
    crossing_window = leader_changes[0]["to_window"] if leader_changes else None
    convergence_start_window = None
    if crossing_window is not None:
        # The maximal run of consecutive windows ending just before the crossing
        # over which probability_gap strictly decreases -- an explicit, checkable
        # rule, not a judgement call.
        index = windows.index(crossing_window)
        start = index - 1
        while start - 1 >= 0:
            previous_gap = views[start - 1]["probability_gap"]
            current_gap = views[start]["probability_gap"]
            if previous_gap is None or current_gap is None or not current_gap < previous_gap:
                break
            start -= 1
        convergence_start_window = views[start]["window_start"] if index > 0 else None
    convergence = {
        "crossing_window": crossing_window,
        "crossing_definition": "first window at which the model's most-probable phase "
                                "differs from the previous window's",
        "convergence_start_window": convergence_start_window,
        "convergence_definition": "first window of the maximal run of consecutive windows, "
                                   "ending immediately before crossing_window, over which "
                                   "probability_gap strictly decreases",
        "all_model_phase_changes": leader_changes,
    }
    if crossing_window is None:
        warnings.append("no change of the model's most-probable phase occurs anywhere in this "
                        "band -- crossing_window is null, and must be reported as such, never guessed.")

    # --- model vs observed timing (Q7) ---
    observed_changes = _changes(views, "observed_phase")
    lag: Optional[int] = None
    lag_note = ("no model phase change and observed phase change with the SAME (from_phase, "
                "to_phase) pair occur inside this band -- the lag is not computable here and "
                "must be reported as unavailable, never estimated.")
    matched: Optional[Dict] = None
    for observed_change in observed_changes:
        for model_change in leader_changes:
            if (model_change["from_phase"], model_change["to_phase"]) == \
                    (observed_change["from_phase"], observed_change["to_phase"]):
                lag = model_change["to_window"] - observed_change["to_window"]
                matched = {"model_change": model_change, "observed_change": observed_change}
                lag_note = ("lag = (window at which the MODEL first shows the new phase) - (window "
                            "at which the ANNOTATION first shows it). Negative = the model changes "
                            "EARLIER than the annotation; positive = LATER; 0 = same window. This is "
                            "a phase-flip proxy for detection timing, not a calibrated detector "
                            "(documented limitation).")
                break
        if lag is not None:
            break
    timing = {
        "all_observed_phase_changes": observed_changes,
        "matched_change_pair": matched,
        "model_vs_observed_lag_windows": lag,
        "lag_definition": lag_note,
    }

    # --- stability / returns / skips (Q8, Q9) ---
    returns: List[Dict] = []
    for position, change in enumerate(leader_changes):
        earlier = {c["from_phase"] for c in leader_changes[:position]}
        if change["to_phase"] in earlier:
            returns.append(change)
    phase_index = _phase_index_map(views)
    skips = [
        {**change,
         "phase_distance": abs(phase_index[change["to_phase"]] - phase_index[change["from_phase"]])}
        for change in leader_changes
        if change["from_phase"] in phase_index and change["to_phase"] in phase_index
        and phase_index[change["to_phase"]] - phase_index[change["from_phase"]] >= 2
    ]
    regressions = [
        {**change,
         "phase_distance": phase_index[change["to_phase"]] - phase_index[change["from_phase"]]}
        for change in leader_changes
        if change["from_phase"] in phase_index and change["to_phase"] in phase_index
        and phase_index[change["to_phase"]] < phase_index[change["from_phase"]]
    ]
    after_center = [v for v in views if (v["offset_from_center"] or 0) > 0]
    center_phase = center["model_phase"] if center else None
    differing_after = [v["window_start"] for v in after_center if v["model_phase"] != center_phase]
    stability = {
        "windows_after_center": [v["window_start"] for v in after_center],
        "center_model_phase": center_phase,
        "windows_after_center_differing_from_center_phase": differing_after,
        "n_windows_after_center": len(after_center),
        "n_windows_after_center_differing": len(differing_after),
        "is_stable_after_center": bool(after_center) and not differing_after,
        "phase_returns": returns,
        "n_phase_returns": len(returns),
        "phase_skips": skips,
        "n_phase_skips": len(skips),
        "phase_regressions": regressions,
        "n_phase_regressions": len(regressions),
        "definitions": {
            "phase_return": "a model phase change back to a phase the model had already LEFT "
                             "earlier in this band",
            "phase_skip": "a FORWARD model phase change whose phase index difference is >= 2, "
                           "i.e. at least one intermediate phase was never the most probable "
                           "(indices as returned by the Reporting API itself). A BACKWARD change "
                           "is never counted here -- it is a phase_regression, listed separately.",
            "phase_regression": "a model phase change to a LOWER phase index than the previous window's",
            "is_stable_after_center": "true only if EVERY window after the center keeps the "
                                       "center's most-probable phase",
            "provenance_of_this_whole_block": "MODEL-DERIVED. Every count here is computed over "
                                               "the model's predicted phase sequence. It says "
                                               "NOTHING about the annotation, whose own "
                                               "transitions come from get_transition_events.",
        },
    }

    # --- entropy (Q10) ---
    entropies = [(v["window_start"], v["entropy"]) for v in views if v["entropy"] is not None]
    entropy_block: Dict[str, Any] = {
        "series": {str(w): e for w, e in entropies},
        "note": "entropy is computed upstream from the FULL 15-phase posterior; it is NOT "
                 "affected by the top-k truncation of the rendered distributions.",
    }
    if entropies:
        max_window, max_entropy = max(entropies, key=lambda we: (we[1], -we[0]))
        min_window, min_entropy = min(entropies, key=lambda we: (we[1], we[0]))
        near = [v["entropy"] for v in views if abs(v["offset_from_center"] or 0) <= 1]
        far = [v["entropy"] for v in views if abs(v["offset_from_center"] or 0) >= 3]
        near_mean, far_mean = _mean(near), _mean(far)
        entropy_block.update({
            "maximum": max_entropy, "maximum_window": max_window,
            "minimum": min_entropy, "minimum_window": min_window,
            "at_center_window": center["entropy"] if center else None,
            "mean_near_center_offset_le_1": near_mean,
            "mean_far_from_center_offset_ge_3": far_mean,
            "near_minus_far": round(near_mean - far_mean, 6)
                               if near_mean is not None and far_mean is not None else None,
            "comparison_definition": "near = windows with |offset_from_center| <= 1; far = windows "
                                      "with |offset_from_center| >= 3. This is a LOCAL band "
                                      "comparison only, never the whole-trajectory aggregate "
                                      "(documented limitation).",
        })
        if far_mean is None:
            warnings.append("this band contains no window with |offset_from_center| >= 3, so no "
                            "'stable period' reference is available for the entropy comparison.")

    return {
        "provenance": PROVENANCE_DERIVED,
        "computed_from": "get_inference_history",
        "computed_by": "orchestrator.temporal_context.derive_series_analysis",
        "is_model_output": False,
        "is_observed_annotation": False,
        "video_id": ((history or {}).get("sample") or {}).get("video_name"),
        "split": ((history or {}).get("sample") or {}).get("split"),
        "center_window": center_window,
        "n_windows": len(views),
        "window_starts": windows,
        "per_window": per_window,
        "sequences": sequences,
        "probability_gap": gap_block,
        "convergence": convergence,
        "timing": timing,
        "stability": stability,
        "entropy": entropy_block,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# ETAPE 2: the window-blocked rendering
# ---------------------------------------------------------------------------

def _render_distribution(window: int, name: str, distribution: Dict[str, float]) -> str:
    body = " | ".join(f"{phase}={_fmt(value)}" for phase, value in distribution.items())
    return f"    W{window}.{name} : {body}" if body else f"    W{window}.{name} : (aucune)"


def _render_window_block(view: Dict) -> List[str]:
    window = view["window_start"]
    header = f"WINDOW {window}"
    if view["is_center"]:
        header += "  [FENETRE CENTRALE / CENTER]"
    header += f"  offset_from_center={_fmt(view['offset_from_center'])}"
    if view["time_available"]:
        header += (f"  window_start_time={_fmt(view['window_start_time'])}"
                   f"  window_end_time={_fmt(view['window_end_time'])}")
    else:
        header += "  (aucune donnee temporelle pour cette fenetre)"

    lines = [header, "  MODEL-DERIVED (estimation du modele Semi-HMM pour CETTE fenetre) :"]
    for field in ("model_phase", "phase_probability", "second_phase",
                  "second_phase_probability", "entropy", "next_phase", "next_phase_probability"):
        lines.append(f"    W{window}.{field} = {_fmt(view[field])}")
    lines.append(_render_distribution(window, "phase_probabilities", view["phase_probabilities"]))
    lines.append(_render_distribution(window, "next_phase_distribution", view["next_phase_distribution"]))

    lines.append("  OBSERVED (annotation du jeu de donnees pour CETTE fenetre — jamais une prediction) :")
    for field in ("observed_phase", "consistency_flag"):
        lines.append(f"    W{window}.{field} = {_fmt(view[field])}")

    lines.append("  DERIVED (calcul deterministe Python — ni sortie de modele, ni annotation) :")
    lines.append(f"    W{window}.probability_gap = {_fmt(view['probability_gap'])}"
                 f"   (= W{window}.phase_probability - W{window}.second_phase_probability)")
    return lines


def render_inference_history(history: Dict) -> str:
    """The window-blocked text form of one get_inference_history() output.
    Every value line carries its own `W<window>.` prefix, so no value can be
    read without its temporal identifier, and the three provenance classes
    are separate labelled sub-blocks inside each window."""
    entries = (history or {}).get("entries") or []
    if not entries:
        return "SERIE TEMPORELLE MULTI-FENETRES : (aucune fenetre retournee)"

    views = sorted((_entry_view(e) for e in entries), key=lambda v: v["window_start"])
    sample = (history or {}).get("sample") or {}
    model = (history or {}).get("model") or {}
    windows = [v["window_start"] for v in views]
    contiguous = windows == list(range(windows[0], windows[0] + len(windows)))

    lines = [
        "=== SERIE TEMPORELLE MULTI-FENETRES (source : Reporting API — get_inference_history) ===",
        f"video={_fmt(sample.get('video_name'))}  split={_fmt(sample.get('split'))}  "
        f"fenetre_centrale={_fmt((history or {}).get('center_window'))}  "
        f"n_fenetres={_fmt((history or {}).get('n_windows'))}",
        f"fenetres presentes (ordre chronologique croissant, aucune omise) : "
        f"{', '.join(str(w) for w in windows)}"
        + ("" if contiguous else "  (ATTENTION : suite non contigue — lire les identifiants)"),
        f"model_version={_fmt(model.get('model_version'))}",
        "Chaque valeur ci-dessous appartient a UNE SEULE fenetre, identifiee par le prefixe W<numero>. "
        "Ne jamais combiner deux prefixes differents dans une meme affirmation.",
        "",
    ]
    for view in views:
        lines.extend(_render_window_block(view))
        lines.append("")
    for warning in (history or {}).get("warnings") or []:
        lines.append(f"AVERTISSEMENT (serie) : {warning}")
    return "\n".join(lines).rstrip()


def _render_change_list(changes: List[Dict]) -> str:
    if not changes:
        return "(aucun)"
    return " | ".join(
        f"W{c['from_window']}->W{c['to_window']} : {c['from_phase']}->{c['to_phase']}"
        + (f" (distance={c['phase_distance']})" if "phase_distance" in c else "")
        for c in changes
    )


def _render_window_keyed_series(name: str, series: Dict[str, Any]) -> str:
    body = " | ".join(f"W{window}={_fmt(value)}" for window, value in series.items())
    return f"{name} : {body}" if body else f"{name} : (aucune)"


def render_series_analysis(analysis: Dict) -> str:
    """The deterministic analysis block. Every quantity here was computed in
    Python from the series above -- explicitly labelled as such so it can
    never be restated as a model prediction or as an annotation."""
    if not analysis:
        return "=== ANALYSE DETERMINISTE DE LA SERIE === (aucune serie disponible)"

    gap = analysis.get("probability_gap", {}) or {}
    convergence = analysis.get("convergence", {}) or {}
    timing = analysis.get("timing", {}) or {}
    stability = analysis.get("stability", {}) or {}
    entropy = analysis.get("entropy", {}) or {}

    lines = [
        "=== ANALYSE DETERMINISTE DE LA SERIE (calculee en Python a partir de "
        "get_inference_history — provenance=derived_deterministic) ===",
        "Ces valeurs ne sont NI une sortie du modele NI une annotation : ce sont des calculs "
        "exacts sur la serie ci-dessus. Elles sont fiables telles quelles ; ne pas les recalculer.",
        "",
        "-- Ecart entre les deux phases les plus probables (Q5 / Q6) --",
        f"definition : {gap.get('definition')}",
        _render_window_keyed_series("probability_gap", gap.get("series", {})),
        f"probability_gap_minimum = {_fmt(gap.get('minimum'))} a la fenetre "
        f"W{_fmt(gap.get('minimum_window'))}",
        f"probability_gap_maximum = {_fmt(gap.get('maximum'))} a la fenetre "
        f"W{_fmt(gap.get('maximum_window'))}",
        f"probability_gap_at_center_window = {_fmt(gap.get('at_center_window'))}",
        f"crossing_window = {_fmt(convergence.get('crossing_window'))}  "
        f"({convergence.get('crossing_definition')})",
        f"convergence_start_window = {_fmt(convergence.get('convergence_start_window'))}  "
        f"({convergence.get('convergence_definition')})",
        f"model_phase_changes : {_render_change_list(convergence.get('all_model_phase_changes', []))}",
        "",
        "-- Modele vs annotation (Q7) --",
        f"observed_phase_changes : {_render_change_list(timing.get('all_observed_phase_changes', []))}",
        f"model_vs_observed_lag_windows = {_fmt(timing.get('model_vs_observed_lag_windows'))}",
        f"definition : {timing.get('lag_definition')}",
        "",
        "-- Sequence predite vs sequence annotee (Q9) --",
        _render_window_keyed_series(
            "MODEL-DERIVED  model_phase_sequence (ce que le MODELE predit)",
            (analysis.get("sequences", {}) or {}).get("model_derived_phase_sequence", {})),
        _render_window_keyed_series(
            "OBSERVED       observed_phase_sequence (annotation du jeu de donnees)",
            (analysis.get("sequences", {}) or {}).get("observed_annotation_phase_sequence", {})),
        f"note : {(analysis.get('sequences', {}) or {}).get('note')}",
        "",
        "-- Stabilite, retours, sauts, regressions de la SEQUENCE PREDITE (Q8 / Q9) --",
        "provenance : MODEL-DERIVED uniquement. Ces comptes portent sur la sequence predite par "
        "le modele. Ils ne disent RIEN de l'annotation, dont les transitions observees "
        "proviennent de get_transition_events (outil distinct).",
        f"center_model_phase = {_fmt(stability.get('center_model_phase'))}",
        f"windows_after_center = {_fmt(stability.get('windows_after_center'))}",
        f"windows_after_center_differing_from_center_phase = "
        f"{_fmt(stability.get('windows_after_center_differing_from_center_phase'))}",
        f"is_stable_after_center = {_fmt(stability.get('is_stable_after_center'))}",
        f"n_phase_returns = {_fmt(stability.get('n_phase_returns'))} : "
        f"{_render_change_list(stability.get('phase_returns', []))}",
        f"n_phase_skips_in_predicted_sequence = {_fmt(stability.get('n_phase_skips'))} : "
        f"{_render_change_list(stability.get('phase_skips', []))}",
        f"n_phase_regressions_in_predicted_sequence = {_fmt(stability.get('n_phase_regressions'))} : "
        f"{_render_change_list(stability.get('phase_regressions', []))}",
        "",
        "-- Incertitude / entropie (Q10) --",
        _render_window_keyed_series("entropy", entropy.get("series", {})),
        f"entropy_maximum = {_fmt(entropy.get('maximum'))} a la fenetre W{_fmt(entropy.get('maximum_window'))}",
        f"entropy_minimum = {_fmt(entropy.get('minimum'))} a la fenetre W{_fmt(entropy.get('minimum_window'))}",
        f"entropy_at_center_window = {_fmt(entropy.get('at_center_window'))}",
        f"entropy_mean_near_center_offset_le_1 = {_fmt(entropy.get('mean_near_center_offset_le_1'))}",
        f"entropy_mean_far_from_center_offset_ge_3 = {_fmt(entropy.get('mean_far_from_center_offset_ge_3'))}",
        f"entropy_near_minus_far = {_fmt(entropy.get('near_minus_far'))}",
        f"definition : {entropy.get('comparison_definition')}",
        f"note : {entropy.get('note')}",
    ]
    for warning in analysis.get("warnings") or []:
        lines.append(f"AVERTISSEMENT (analyse) : {warning}")
    return "\n".join(lines)

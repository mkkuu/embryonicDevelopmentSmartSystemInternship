"""
COMPACT_CONTEXT_V2 renderer -- one prompt organised BY PROVENANCE CLASS.

    QUESTION
    [IDENTIFIANTS]      sample / window / model identifiers (neither observed nor predicted)
    [OBSERVED]          annotation of the dataset, only
    [MODEL-DERIVED]     Semi-HMM outputs, only
    [SERIE TEMPORELLE]  per-window blocks, each with its own local sub-labels (unchanged renderer)
    [DERIVED]           exact Python arithmetic on the data above
    [SCIENTIFIC]        Istanbul Consensus blocks + the limitations that travel with them
    [DOCUMENTATION PROJET]  filtered, curated project chunks (stubs for narrative ones)
    [LIMITATIONS]       context fields that bound the answer, cited by path
    [AVERTISSEMENTS]    tool / context warnings

What is different from compact_v1, and why (graded artefact compact_v1,
docs analysis 2026-09-12):
  * NO global "PROVENANCE DES BLOCS" legend. Q3 regressed under compact_v1
    (3/3 -> 1/3) and one answer recopied the legend inverted. Provenance is now
    a LOCAL label printed directly above the lines it qualifies, and nothing
    else.
  * Every value line keeps the exact `path = value` form of the historical
    renderer (`prompt._flatten_dynamic_context`), so grounding_check finds
    each restated value unchanged, and a reader can chase any line back into
    context["dynamic_context"]. Only the GROUPING of lines changes.
  * The Istanbul Consensus is a separate class ([SCIENTIFIC]); project
    documentation is another ([DOCUMENTATION PROJET]); neither is ever
    described as a measurement on this embryo.
  * Long scalar lists (get_trajectory.window_starts, hundreds of integers) are
    summarised as a contiguous range when they ARE contiguous -- a lossless
    description -- and printed in full otherwise.

Nothing here is invented: every rendered value is a context value, every
label is a fixed class name, and no expected answer, phase name or number is
added by the renderer (pinned by Tests/orchestrator/test_compact_v2_context.py).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .compact_context import (
    MARKER_TEMPLATES,
    compact_document_context_v2,
)
from .temporal_context import render_inference_history, render_series_analysis

CONTEXT_FORMAT_COMPACT_V2 = "compact_v2"

# Local labels -- the ONLY prose this renderer adds around the data.
LABEL_REFERENCE = "[FENETRE DE REFERENCE — echantillon et fenetre courante ; ni observation ni prediction]"
LABEL_IDENTIFIERS = "[MODELE CHARGE ET METADONNEES — identite du modele pour citation ; ni observation ni prediction]"
LABEL_OBSERVED = "[OBSERVED — annotation du jeu de donnees ; jamais une prediction du modele]"
LABEL_MODEL = "[MODEL-DERIVED — prediction / estimation du Semi-HMM ; jamais une observation biologique]"
LABEL_SERIES = "[SERIE TEMPORELLE MULTI-FENETRES — chaque fenetre porte ses sous-blocs MODEL-DERIVED / OBSERVED / DERIVED]"
LABEL_DERIVED = "[DERIVED — calcul deterministe Python sur les donnees ci-dessus ; ni sortie du modele ni annotation]"
LABEL_SCIENTIFIC = "[SCIENTIFIC — Istanbul Consensus 2025 ; connaissance externe, jamais une mesure sur cet embryon]"
LABEL_PROJECT_DOCS = "[DOCUMENTATION PROJET — documentation interne filtree ; jamais une mesure sur cet embryon]"
LABEL_LIMITATIONS = "[LIMITATIONS — champs du contexte qui bornent la reponse]"
LABEL_WARNINGS = "[AVERTISSEMENTS — a signaler dans la reponse, jamais a omettre]"

# Path prefixes -> class. Anything from a Reporting-API tool that matches none
# of these is an identifier (sample, window, model, warnings, n_windows...).
_OBSERVED_PREFIXES = (
    "get_current_inference.ground_truth.",
    "get_transition_events.",
)
_MODEL_PREFIXES = (
    "get_current_inference.current_state.",
    "get_current_inference.next_phase.",
    "get_current_inference.duration.",
    "get_model_metadata.",
)
# get_trajectory.transition_chain mixes both: the segment phase and its
# observed duration come from the annotation, the probabilities from the model.
_TRAJECTORY_OBSERVED_SUFFIXES = (".phase =", ".observed_duration_windows =")
_DERIVED_PREFIXES = ("transition_context.",)

_MAX_INLINE_LIST = 12


_REFERENCE_PREFIXES = ("get_current_inference.sample.", "get_current_inference.window.",
                       "get_trajectory.sample.", "get_inference_history.sample.")


def _bucket(line: str) -> str:
    path = line.split(" = ", 1)[0]
    if path.startswith(_DERIVED_PREFIXES):
        return "derived"
    if path.startswith(_REFERENCE_PREFIXES):
        return "reference"
    if path.startswith(_OBSERVED_PREFIXES):
        return "observed"
    if path.startswith(_MODEL_PREFIXES):
        return "model"
    if path.startswith("get_trajectory.transition_chain"):
        return "observed" if any(s in line for s in _TRAJECTORY_OBSERVED_SUFFIXES) else "model"
    return "identifiers"


def _summarise_scalar_list(path: str, values: List[Any]) -> Optional[str]:
    """Lossless range description for a long contiguous integer list."""
    if len(values) <= _MAX_INLINE_LIST or not all(isinstance(v, int) and not isinstance(v, bool)
                                                   for v in values):
        return None
    if values == list(range(values[0], values[0] + len(values))):
        return (f"{path} = {len(values)} valeurs entieres contigues de {values[0]} a {values[-1]} "
                f"(liste complete non imprimee : aucune valeur omise)")
    return None


def _flatten(value: Any, path: str, lines: List[str], deferred: List[str]) -> None:
    """Same output as prompt._flatten_dynamic_context (imported lazily to avoid
    an import cycle), plus the long-list summary."""
    from .prompt import _flatten_dynamic_context, _is_scalar  # noqa: WPS433

    if isinstance(value, list) and value and all(_is_scalar(v) for v in value):
        summary = _summarise_scalar_list(path, value)
        if summary is not None:
            lines.append(summary)
            return
    if isinstance(value, dict):
        # descend one level ourselves so nested long lists get the summary too
        if not value:
            lines.append(f"{path} = {{}}")
            return
        from .prompt import _is_homogeneous_scalar_distribution, _sub_path, _format_scalar
        if _is_homogeneous_scalar_distribution(value):
            for key, sub in value.items():
                deferred.append(f"{_sub_path(path, key)} = {_format_scalar(sub)}")
            return
        for key, sub in value.items():
            _flatten(sub, _sub_path(path, key), lines, deferred)
        return
    _flatten_dynamic_context(value, path, lines, deferred)


def derive_transition_context(dynamic_context: Dict[str, Any]) -> Dict[str, Any]:
    """Exact arithmetic over get_transition_events + the reference window:
    which annotated transition is the LAST at/before the window and which is
    the FIRST after it. Motivation: with the full 12-transition history in
    the context, 6/6 graded answers to Q1 picked the video's last transition
    (w176) instead of the last one at/before window 156 -- the reading rule was
    stated in the prompt but never materialised as a field. This is the same
    principle as `temporal_context.derive_series_analysis`: a quantity that is
    exactly computable is computed in Python, labelled DERIVED, and never left
    to the LLM. Returns {} when either input is missing."""
    events = (dynamic_context or {}).get("get_transition_events")
    current = (dynamic_context or {}).get("get_current_inference")
    if not isinstance(events, dict):
        return {}
    window = None
    if isinstance(current, dict):
        window = (current.get("window") or {}).get("window_start")
    if window is None:
        window = events.get("window")
    transitions = [t for t in (events.get("transitions") or [])
                   if isinstance(t, dict) and t.get("window_start") is not None]
    if window is None or len(transitions) < 2:
        # With a single (bracketing) transition there is nothing to select:
        # the block would only repeat transitions[0]. Empty, not redundant.
        return {}
    ordered = sorted(enumerate(transitions), key=lambda it: it[1]["window_start"])
    before = [(i, t) for i, t in ordered if t["window_start"] <= window]
    after = [(i, t) for i, t in ordered if t["window_start"] > window]

    def _view(item: Optional[Tuple[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        index, t = item
        return {
            "transitions_index": index,
            "from_phase": t.get("from_phase"), "to_phase": t.get("to_phase"),
            "window_start": t.get("window_start"), "window_end": t.get("window_end"),
            "window_start_time": t.get("window_start_time"),
            "window_end_time": t.get("window_end_time"),
            "phase_distance": t.get("phase_distance"), "is_skip": t.get("is_skip"),
        }

    return {
        "provenance": "derived_deterministic",
        "computed_from": "get_transition_events.transitions + get_current_inference.window.window_start",
        "reference_window": window,
        "definition": ("last_annotated_transition_at_or_before_reference_window = l'entree de "
                       "get_transition_events.transitions dont window_start est le plus grand "
                       "sans depasser reference_window ; next_annotated_transition_after_reference_window "
                       "= la premiere entree dont window_start depasse reference_window. Les deux "
                       "sont des transitions ANNOTEES (OBSERVED), selectionnees par arithmetique."),
        "n_annotated_transitions_at_or_before_reference_window": len(before),
        "n_annotated_transitions_after_reference_window": len(after),
        "last_annotated_transition_at_or_before_reference_window": _view(before[-1] if before else None),
        "next_annotated_transition_after_reference_window": _view(after[0] if after else None),
    }


def _render_scientific(scientific_context: Optional[Dict[str, Any]]) -> List[str]:
    if not scientific_context:
        return []
    lines = [LABEL_SCIENTIFIC]
    blocks = scientific_context.get("blocks") or []
    limitations = scientific_context.get("limitations") or []
    if not blocks and not limitations:
        for w in scientific_context.get("warnings") or []:
            lines.append(f"(aucune reference scientifique : {w})")
        return lines + [""]
    for entry in blocks:
        lines.extend(_render_science_block(entry, "REFERENCE"))
    for entry in limitations:
        lines.extend(_render_science_block(entry, "LIMITATION LIEE"))
    for w in scientific_context.get("warnings") or []:
        lines.append(f"(note : {w})")
    lines.append("")
    return lines


def _render_science_block(entry: Dict[str, Any], role: str) -> List[str]:
    head = f"[{role} {entry.get('block_id')}] {entry.get('subject') or ''}".rstrip()
    meta = []
    for key, label in (("evidence_type", "type"), ("evidence_grade", "niveau de preuve"),
                       ("page", "page"), ("table", "table"), ("section", "section")):
        if entry.get(key):
            meta.append(f"{label}: {entry[key]}")
    if meta:
        head += "  (" + " ; ".join(meta) + ")"
    text = entry.get("text") or ""
    # The corpus text of a shape-A block starts with its own heading line
    # ("### <id> — <subject> - **type:** ... - **source:** ..."), which the
    # head above already prints; that exact prefix is removed and nothing else.
    prefix = f"### {entry.get('block_id')} — {entry.get('subject') or ''}".rstrip()
    if prefix and text.startswith(prefix):
        text = text[len(prefix):].lstrip(" -")
    return [head, f"  {text}".rstrip()]


def _render_project_docs(document_context: List[Dict[str, Any]],
                         retrieval_audit: Optional[List[Dict[str, Any]]]) -> List[str]:
    lines = [LABEL_PROJECT_DOCS]
    if not document_context:
        lines.append("(aucun document projet retenu)")
    for i, chunk in enumerate(compact_document_context_v2(document_context), start=1):
        head = (f"[RAG DOCUMENT {i}] source: {chunk.get('source')} | section: {chunk.get('section')} "
                f"| status: {chunk.get('status')}")
        if chunk.get("compact_dropped"):
            lines.append(head)
            lines.append("  " + MARKER_TEMPLATES["dropped_chunk"].format(
                reason=chunk.get("compact_drop_reason"), rule=chunk.get("compact_drop_rule")))
            continue
        lines.append(head)
        lines.append(chunk.get("compact_content") or "")
        if chunk.get("compact_dropped_lines"):
            lines.append("  " + MARKER_TEMPLATES["dropped_lines"].format(
                n=chunk["compact_dropped_lines"],
                rules=",".join(chunk.get("compact_rules_applied") or [])))
    if retrieval_audit:
        skipped = [a for a in retrieval_audit if a.get("decision") == "skipped"]
        if skipped:
            rules = sorted({a.get("rule") for a in skipped if a.get("rule")})
            lines.append(f"(selection : {len(skipped)} candidat(s) de documentation narrative "
                         f"ecarte(s) avant rendu -- regles {','.join(rules)})")
    lines.append("")
    return lines


def _limitations(dynamic_context: Dict[str, Any], scientific_context: Optional[Dict[str, Any]],
                 model_lines: List[str], identifier_lines: List[str]) -> List[str]:
    """Context fields with limitation semantics, each cited by its own path so
    the reader (and grounding) can find it above. Fixed labels, no values
    invented."""
    out: List[str] = []
    for line in identifier_lines:
        if ".time_unit = " in line:
            out.append(f"- {line}  -> unite temporelle non verifiee : ne nommer aucune unite")
    for line in model_lines:
        if ".duration_unit = " in line:
            out.append(f"- {line}  -> duree ATTENDUE par le modele, en fenetres ; ce n'est pas une "
                       f"duree ecoulee")
        if ".duration_available = false" in line:
            out.append(f"- {line}  -> aucune duree attendue disponible")
    series = (dynamic_context or {}).get("series_analysis") or {}
    timing = series.get("timing") or {}
    if "model_vs_observed_lag_windows" in timing and timing.get("model_vs_observed_lag_windows") is None:
        out.append("- model_vs_observed_lag_windows = null  -> decalage modele/annotation NON "
                   "calculable ici ; un offset_from_center n'est pas un decalage")
    for w in series.get("warnings") or []:
        out.append(f"- (serie) {w}")
    history = (dynamic_context or {}).get("get_inference_history") or {}
    for w in history.get("warnings") or []:
        out.append(f"- (serie) {w}")
    if scientific_context:
        ids = [e.get("block_id") for e in scientific_context.get("limitations") or []]
        if ids:
            out.append(f"- limitations du Consensus a conserver avec toute reference : {', '.join(ids)}")
    return out


def render_compact_v2(context: Dict[str, Any]) -> str:
    dynamic = context.get("dynamic_context") or {}
    documents = context.get("document_context") or []
    scientific = context.get("scientific_context")
    audit = context.get("retrieval_audit")
    warnings = context.get("warnings") or []

    lines: List[str] = []
    deferred: List[str] = []
    history_text: Optional[str] = None
    series_text: Optional[str] = None
    for tool, value in dynamic.items():
        if tool == "get_inference_history" and isinstance(value, dict):
            history_text = render_inference_history(value)
        elif tool == "series_analysis" and isinstance(value, dict):
            series_text = render_series_analysis(value)
        else:
            _flatten(value, tool, lines, deferred)

    buckets: Dict[str, List[str]] = {"reference": [], "identifiers": [], "observed": [],
                                     "model": [], "derived": []}
    for line in lines + deferred:
        buckets[_bucket(line)].append(line)

    out: List[str] = ["QUESTION:", context.get("question", "") or "", ""]

    def _block(label: str, body: List[str]) -> None:
        if body:
            out.append(label)
            out.extend(body)
            out.append("")

    _block(LABEL_REFERENCE, buckets["reference"])
    _block(LABEL_OBSERVED, buckets["observed"])
    _block(LABEL_MODEL, buckets["model"])
    if history_text:
        from .prompt import _center_window_reconciliation  # noqa: WPS433
        body = [history_text] + _center_window_reconciliation(dynamic)
        _block(LABEL_SERIES, body)
    derived_body = list(buckets["derived"])
    if series_text:
        if derived_body:
            derived_body.append("")
        derived_body.append(series_text)
    _block(LABEL_DERIVED, derived_body)
    _block(LABEL_IDENTIFIERS, buckets["identifiers"])
    if not dynamic:
        out.extend(["(aucune donnee dynamique recuperee)", ""])

    out.extend(_render_scientific(scientific))
    out.extend(_render_project_docs(documents, audit))

    limitations = _limitations(dynamic, scientific, buckets["model"], buckets["reference"])
    _block(LABEL_LIMITATIONS, limitations if limitations else ["(aucune limitation structurelle detectee)"])

    out.append(LABEL_WARNINGS)
    if warnings:
        out.extend(f"- {w}" for w in warnings)
    else:
        out.append("(aucun)")
    out.append("")
    return "\n".join(out)

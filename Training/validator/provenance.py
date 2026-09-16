"""
Provenance resolver: what the CONTEXT licenses, and how the CLAIM is framed.

This is the module that answers the question a presence check cannot:

    the claim says "la phase dominante etait t7"
    the context contains t7 -- but ONLY under `current_state.current_phase`
    => the claim is framed as an OBSERVATION and supported only by a PREDICTION

Two independent halves, deliberately kept apart so a bug in one cannot hide a
bug in the other:

  1. `phase_tokens_by_provenance()` / `transitions_by_provenance()` walk the
     REAL context and record, per provenance class, which phase names and
     which transition pairs that class actually carries. Class membership comes
     from the STRUCTURE (which tool, which key), never from a token match.
  2. `detect_framing()` reads only the claim TEXT and reports which provenance
     the sentence attributes itself to.

A mismatch between the two is the violation. Nothing here reads the annotation
when the annotation is absent from the context -- the resolver sees exactly
what the LLM saw, and nothing else.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .schema import (COMBINED, DERIVED_FROM_MODEL, DERIVED_FROM_OBSERVED,
                     MODEL_DERIVED, OBSERVED, SCIENTIFIC, SYSTEM)

PHASE_TOKEN = re.compile(r"\bt(?:PB2|PNa|PNf|M|SB|EB|B|\d\+?)\b", re.IGNORECASE)

# --- framing markers ------------------------------------------------------
# Kept explicit and narrow, same discipline as router.py: a keyword table that
# a reader can audit, never an NLP model.
_MODEL_MARKERS = (
    "le modele predit", "le modele prevoit", "le modele estime", "selon le modele",
    "prediction du modele", "phase predite", "sequence predite", "prediction",
    "predit", "predite", "model-derived", "model derived", "semi-hmm", "semi_hmm",
    "current_phase", "current_state", "next_phase", "phase_probability",
    "d'apres le modele", "estimation du modele",
    # v1.1 (2026-09-12) -- replay of the compact_v1 artefact (90 answers) showed
    # R2 firing on "la deuxieme phase la plus probable est t8" (Q4, 6/6), on
    # "t7 avec une probabilite de 0.76" (Q6) and on "W156.model_phase = t7"
    # (Q9). In this project every probability, every `model_phase` and every
    # `second_phase` is a Semi-HMM quantity -- the annotation carries none --
    # so this vocabulary attributes the sentence to the model. "probable" alone
    # is NOT listed: "probablement t7" is a hedge, not an attribution.
    "plus probable", "probabilit", "model_phase", "second_phase",
    "phase_probabilities", "most_likely",
)
_OBSERVED_MARKERS = (
    "annotation", "annote", "annotee", "annotees", "observe", "observee", "observees",
    "observed", "ground_truth", "ground truth", "jeu de donnees", "embryologiste",
    "get_transition_events", "verite terrain",
)
_SCIENTIFIC_MARKERS = (
    "consensus", "istanbul", "eshre", "alpha", "referentiel", "corpus",
    "documentation", "litterature", "[doc ", "doc 1", "doc 2", "doc 3",
    "recommandation", "publication",
)
_DERIVED_MARKERS = (
    "analyse deterministe", "derived", "calcul deterministe", "derive_", "calcule en python",
)
_SYSTEM_MARKERS = (
    "non calculable", "indisponible", "n'est pas disponible", "pas suffisamment d'informations",
    "le contexte ne contient pas", "quantite demandee",
)


def _norm(text: str) -> str:
    """Accent-insensitive lowercase, same normalisation family the Router uses."""
    out = (text or "").lower()
    for a, b in (("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"), ("à", "a"),
                 ("â", "a"), ("î", "i"), ("ï", "i"), ("ô", "o"), ("û", "u"),
                 ("ù", "u"), ("ç", "c"), ("–", "-"), ("—", "-"), ("’", "'")):
        out = out.replace(a, b)
    return out


def detect_framing(text: str) -> Dict[str, List[str]]:
    """Which provenance classes the SENTENCE attributes itself to, with the
    exact markers found. Empty lists everywhere = a bare assertion."""
    low = _norm(text)
    return {
        MODEL_DERIVED: [m for m in _MODEL_MARKERS if m in low],
        OBSERVED: [m for m in _OBSERVED_MARKERS if m in low],
        SCIENTIFIC: [m for m in _SCIENTIFIC_MARKERS if m in low],
        DERIVED_FROM_MODEL: [m for m in _DERIVED_MARKERS if m in low],
        SYSTEM: [m for m in _SYSTEM_MARKERS if m in low],
    }


def framing_classes(text: str) -> Set[str]:
    return {k for k, v in detect_framing(text).items() if v}


# --- what the context actually carries ------------------------------------

def _walk_strings(value: Any, out: Set[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _walk_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_strings(v, out)
    elif isinstance(value, str):
        out.add(value)


def _phases_in(value: Any) -> Set[str]:
    strings: Set[str] = set()
    _walk_strings(value, strings)
    found: Set[str] = set()
    for s in strings:
        for m in PHASE_TOKEN.finditer(s):
            found.add(m.group(0))
    return found


def _dict_keys_that_are_phases(value: Any, out: Set[str]) -> None:
    """Phase names reachable only as dict KEYS (e.g. `phase_probabilities`)."""
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and PHASE_TOKEN.fullmatch(k):
                out.add(k)
            _dict_keys_that_are_phases(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _dict_keys_that_are_phases(v, out)


def phase_tokens_by_provenance(context: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Per provenance class, the phase names that class actually carries in
    THIS context. Membership is structural: which tool, which key."""
    dyn = (context or {}).get("dynamic_context") or {}
    out: Dict[str, Set[str]] = {OBSERVED: set(), MODEL_DERIVED: set(),
                                DERIVED_FROM_OBSERVED: set(), DERIVED_FROM_MODEL: set(),
                                SCIENTIFIC: set()}

    current = dyn.get("get_current_inference") or {}
    if isinstance(current, dict):
        gt = current.get("ground_truth")
        if gt:
            out[OBSERVED] |= _phases_in(gt)
        for key in ("current_state", "next_phase", "duration"):
            block = current.get(key)
            if block:
                out[MODEL_DERIVED] |= _phases_in(block)
                _dict_keys_that_are_phases(block, out[MODEL_DERIVED])

    events = dyn.get("get_transition_events")
    if events:
        out[OBSERVED] |= _phases_in(events)

    history = dyn.get("get_inference_history") or {}
    for entry in (history.get("entries") or []) if isinstance(history, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("observed"):
            out[OBSERVED] |= _phases_in(entry["observed"])
        if entry.get("model_derived"):
            out[MODEL_DERIVED] |= _phases_in(entry["model_derived"])
            _dict_keys_that_are_phases(entry["model_derived"], out[MODEL_DERIVED])

    obs_series = dyn.get("observed_series_analysis")
    if obs_series:
        out[DERIVED_FROM_OBSERVED] |= _phases_in(obs_series)

    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if not isinstance(series, dict):
            continue
        for sub in ("per_window", "sequences", "convergence", "stability"):
            if series.get(sub):
                target = DERIVED_FROM_MODEL
                out[target] |= _phases_in(series[sub])
        # `series_analysis` (FULL) mixes both: its observed_phase / observed
        # sequence belong to the OBSERVED-derived side, never to the model's.
        per_window = series.get("per_window") or {}
        if isinstance(per_window, dict):
            for entry in per_window.values():
                if isinstance(entry, dict) and entry.get("observed_phase"):
                    out[DERIVED_FROM_OBSERVED].add(entry["observed_phase"])
                    out[DERIVED_FROM_MODEL].discard(entry["observed_phase"])
        sequences = series.get("sequences") or {}
        if isinstance(sequences, dict) and sequences.get("observed_annotation_phase_sequence"):
            out[DERIVED_FROM_OBSERVED] |= _phases_in(
                sequences["observed_annotation_phase_sequence"])
        timing = series.get("timing") or {}
        if isinstance(timing, dict) and timing.get("all_observed_phase_changes"):
            out[DERIVED_FROM_OBSERVED] |= _phases_in(timing["all_observed_phase_changes"])

    for doc in (context or {}).get("document_context") or []:
        out[SCIENTIFIC] |= _phases_in(doc)
    # compact_v2 (2026-09-12): the Istanbul Consensus blocks placed in the
    # context are external scientific knowledge -- SCIENTIFIC, never a value
    # measured on this embryo.
    scientific = (context or {}).get("scientific_context")
    if scientific:
        out[SCIENTIFIC] |= _phases_in(scientific)
    return out


def transitions_by_provenance(context: Dict[str, Any]) -> Dict[str, Set[Tuple[str, str]]]:
    """Per provenance class, the (from_phase, to_phase) pairs it carries."""
    dyn = (context or {}).get("dynamic_context") or {}
    out: Dict[str, Set[Tuple[str, str]]] = {OBSERVED: set(), MODEL_DERIVED: set()}

    events = dyn.get("get_transition_events") or {}
    for t in (events.get("transitions") or []) if isinstance(events, dict) else []:
        if isinstance(t, dict) and t.get("from_phase") and t.get("to_phase"):
            out[OBSERVED].add((t["from_phase"], t["to_phase"]))

    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if not isinstance(series, dict):
            continue
        for c in ((series.get("convergence") or {}).get("all_model_phase_changes") or []):
            if isinstance(c, dict) and c.get("from_phase") and c.get("to_phase"):
                out[MODEL_DERIVED].add((c["from_phase"], c["to_phase"]))
        for c in ((series.get("timing") or {}).get("all_observed_phase_changes") or []):
            if isinstance(c, dict) and c.get("from_phase") and c.get("to_phase"):
                out[OBSERVED].add((c["from_phase"], c["to_phase"]))

    obs_series = dyn.get("observed_series_analysis")
    if isinstance(obs_series, dict):
        for c in (obs_series.get("observed_phase_changes") or []):
            if isinstance(c, dict) and c.get("from_phase") and c.get("to_phase"):
                out[OBSERVED].add((c["from_phase"], c["to_phase"]))
    return out


def available_provenance(context: Dict[str, Any]) -> Set[str]:
    """Which provenance classes this context makes available at all."""
    phases = phase_tokens_by_provenance(context)
    transitions = transitions_by_provenance(context)
    out = {k for k, v in phases.items() if v}
    if transitions[OBSERVED]:
        out.add(OBSERVED)
    if transitions[MODEL_DERIVED]:
        out.add(MODEL_DERIVED)
    dyn = (context or {}).get("dynamic_context") or {}
    current = dyn.get("get_current_inference") or {}
    if isinstance(current, dict) and current.get("ground_truth"):
        out.add(OBSERVED)
    if isinstance(current, dict) and current.get("current_state"):
        out.add(MODEL_DERIVED)
    if (context or {}).get("document_context") or (context or {}).get("scientific_context"):
        out.add(SCIENTIFIC)
    return out


def has_observed_annotation(context: Dict[str, Any]) -> bool:
    """v1.3: STRUCTURAL proof that an annotation is present in this context
    (ground_truth block, get_transition_events, an observed entry of the
    history, or an observed series). A qualification may only say "aucune
    annotation n'est presente" when this returns False -- the benchmark's R2
    text asserted it while `ground_truth_phase` sat in the very same context."""
    dyn = (context or {}).get("dynamic_context") or {}
    current = dyn.get("get_current_inference") or {}
    if isinstance(current, dict) and current.get("ground_truth"):
        return True
    events = dyn.get("get_transition_events")
    if isinstance(events, dict) and events.get("transitions"):
        return True
    history = dyn.get("get_inference_history") or {}
    if isinstance(history, dict) and any(
            isinstance(e, dict) and e.get("observed") for e in (history.get("entries") or [])):
        return True
    if dyn.get("observed_series_analysis"):
        return True
    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict):
            if (series.get("sequences") or {}).get("observed_annotation_phase_sequence"):
                return True
            if (series.get("timing") or {}).get("all_observed_phase_changes"):
                return True
    return False


# --- v1.3: verbatim restatement of a printed field -------------------------
# A sentence such as "* t8 : 0.23679880956344662" or "W156=t7 | W157=t8" does
# not assert that the embryo IS in t8 -- it copies a (key, value) pair the
# context printed. Such a copy carries the provenance of the field it copies,
# whatever the surrounding vocabulary. Resolution is STRUCTURAL: the pair must
# actually exist, with that value, in a block of the context.

_PHASE_VALUE = re.compile(
    r"(?P<phase>\bt(?:PB2|PNa|PNf|M|SB|EB|B|\d\+?))\s*[:=]\s*(?P<num>[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE)
_WINDOW_PHASE = re.compile(
    r"\bW(?P<win>\d+)\s*(?:\.\s*(?P<field>model_phase|second_phase|next_phase|observed_phase))?"
    r"\s*[:=]\s*(?P<phase>t(?:PB2|PNa|PNf|M|SB|EB|B|\d\+?))\b", re.IGNORECASE)


def _num_matches(claimed: str, value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    text = claimed.replace(",", ".")
    try:
        num = float(text)
    except ValueError:
        return False
    if str(value).startswith(text) or repr(float(value)).startswith(text):
        return True
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text.lower() else 0
    return abs(num - float(value)) <= 0.5 * 10 ** (-decimals) if decimals else num == float(value)


def _phase_value_in(block: Any, phase: str, num: str) -> bool:
    """True when `block` (walked recursively) holds `phase` as a KEY whose value
    is `num`, or a dict whose phase-valued field is `phase` and one of whose
    numeric fields is `num`."""
    low = phase.lower()
    if isinstance(block, dict):
        for k, v in block.items():
            if isinstance(k, str) and k.lower() == low and _num_matches(num, v):
                return True
        phase_fields = [v for v in block.values() if isinstance(v, str) and v.lower() == low]
        if phase_fields and any(_num_matches(num, v) for v in block.values()):
            return True
        return any(_phase_value_in(v, phase, num) for v in block.values())
    if isinstance(block, (list, tuple)):
        return any(_phase_value_in(v, phase, num) for v in block)
    return False


def _model_blocks(dyn: Dict[str, Any]) -> List[Any]:
    out: List[Any] = []
    current = dyn.get("get_current_inference") or {}
    if isinstance(current, dict):
        out += [current.get(k) for k in ("current_state", "next_phase", "duration") if current.get(k)]
    history = dyn.get("get_inference_history") or {}
    if isinstance(history, dict):
        out += [e.get("model_derived") for e in (history.get("entries") or [])
                if isinstance(e, dict) and e.get("model_derived")]
    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict):
            out += [series.get(k) for k in ("per_window", "probability_gap", "entropy") if series.get(k)]
    trajectory = dyn.get("get_trajectory") or {}
    if isinstance(trajectory, dict) and trajectory.get("transition_chain"):
        out.append(trajectory["transition_chain"])
    return out


def _observed_blocks(dyn: Dict[str, Any]) -> List[Any]:
    out: List[Any] = []
    current = dyn.get("get_current_inference") or {}
    if isinstance(current, dict) and current.get("ground_truth"):
        out.append(current["ground_truth"])
    history = dyn.get("get_inference_history") or {}
    if isinstance(history, dict):
        out += [e.get("observed") for e in (history.get("entries") or [])
                if isinstance(e, dict) and e.get("observed")]
    if dyn.get("get_transition_events"):
        out.append(dyn["get_transition_events"])
    return out


def restated_field_provenance(context: Dict[str, Any], text: str) -> Dict[str, Set[str]]:
    """Per phase token of `text` that is a verbatim copy of a printed field,
    the provenance classes of the block(s) holding that exact field. A phase
    absent from the result is NOT a restatement and is judged as an assertion.
    Empty dict when the sentence copies nothing."""
    dyn = (context or {}).get("dynamic_context") or {}
    out: Dict[str, Set[str]] = {}
    for m in _PHASE_VALUE.finditer(text or ""):
        phase, num = m.group("phase"), m.group("num")
        if any(_phase_value_in(b, phase, num) for b in _model_blocks(dyn)):
            out.setdefault(phase.lower(), set()).add(MODEL_DERIVED)
        if any(_phase_value_in(b, phase, num) for b in _observed_blocks(dyn)):
            out.setdefault(phase.lower(), set()).add(OBSERVED)
    per_window: Dict[str, Any] = {}
    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict) and isinstance(series.get("per_window"), dict):
            per_window.update(series["per_window"])
    sequences: Dict[str, Any] = {}
    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict) and isinstance(series.get("sequences"), dict):
            sequences = series["sequences"]
    for m in _WINDOW_PHASE.finditer(text or ""):
        win, field, phase = m.group("win"), (m.group("field") or "").lower(), m.group("phase")
        entry = per_window.get(win) or per_window.get(str(int(win))) or {}
        low = phase.lower()
        model_fields = ("model_phase", "second_phase", "next_phase")
        if field == "observed_phase":
            candidates = {DERIVED_FROM_OBSERVED: [entry.get("observed_phase")]}
        elif field in model_fields:
            candidates = {DERIVED_FROM_MODEL: [entry.get(field)]}
        else:
            candidates = {DERIVED_FROM_MODEL: [entry.get(f) for f in model_fields]
                          + [(sequences.get("model_derived_phase_sequence") or {}).get(win)],
                          DERIVED_FROM_OBSERVED: [entry.get("observed_phase"),
                                                  (sequences.get("observed_annotation_phase_sequence") or {}).get(win)]}
        for cls, values in candidates.items():
            if any(isinstance(v, str) and v.lower() == low for v in values):
                out.setdefault(low, set()).add(cls)
    return out


def sources_for_phase(context: Dict[str, Any], phase: str) -> List[str]:
    """Every provenance class of THIS context that carries `phase`.
    An empty list means the token is not in the context at all."""
    table = phase_tokens_by_provenance(context)
    low = phase.lower()
    return sorted({cls for cls, toks in table.items()
                   if any(t.lower() == low for t in toks)})


def sources_for_transition(context: Dict[str, Any],
                           pair: Tuple[str, str]) -> List[str]:
    table = transitions_by_provenance(context)
    lowered = (pair[0].lower(), pair[1].lower())
    return sorted({cls for cls, pairs in table.items()
                   if any((a.lower(), b.lower()) == lowered for a, b in pairs)})

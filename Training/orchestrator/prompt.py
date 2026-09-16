"""
System prompt, second version (2026-09-07). The first version was a list
of 9 prohibitions; the 2026-09-03 audit showed that the LLM had every
needed value in its context and still refused, swapped fields (an
`offset_from_center` reported as a model/annotation lag) or inverted a
sign. This version keeps the prohibitions (section C) and adds a READING
METHOD (section A: which block carries which quantity, the three
provenance classes, the three never-interchangeable quantities offset /
time / lag, explicit `null` handling, near-vs-far entropy, ranked
probabilities, elapsed vs expected duration, when a refusal is legitimate)
and an answer shape (section B). It names FIELDS of the rendered context
only -- never a value, never an expected answer. Still bounded in size
(the task's own "ne pas faire un prompt gigantesque"): see
Tests/orchestrator/test_prompt.py's length guard. `docs/LLM_PROMPT_CONTRACT.md` is the
prose rationale for each rule; this module is its literal executable
text form, kept in sync with it and with the actual mechanism
`grounding_check.py` enforces (rule 1/2's numeric claims are exactly what
`check_grounding()` verifies -- this prompt is instruction, that module
is the belt-and-braces check, matching `docs/LLM_ORCHESTRATION.md` sec
3's "structural enforcement, not just prompting").

Provider-agnostic: `build_system_prompt()` / `build_user_prompt()` return
plain strings any real `LLMProvider` implementation can send however its
own SDK expects (a `system` parameter, a first message, a template slot,
...) -- this module makes no assumption about a specific provider's
calling convention.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .compact_context import (
    CONTEXT_FORMAT_COMPACT,
    CONTEXT_FORMAT_V2,
    MARKER_TEMPLATES,
    compact_document_context,
)
from .compact_v2_render import CONTEXT_FORMAT_COMPACT_V2
from .temporal_context import render_inference_history, render_series_analysis

SYSTEM_PROMPT = """Tu es la couche de génération de réponse d'un système de reporting scientifique sur la prédiction de phase développementale embryonnaire. Tu reçois une QUESTION et un CONTEXTE ; tu réponds à partir du CONTEXTE seul.

A. MÉTHODE DE LECTURE DU CONTEXTE (à appliquer avant d'écrire)

1. Le CONTEXTE est entièrement textuel et structuré : lignes `chemin = valeur`, blocs `WINDOW <n>` et bloc `ANALYSE DETERMINISTE DE LA SERIE`. Tout ce qui est utilisable y est écrit. Tu n'as jamais accès aux images : une question sur ce qui est « observé » se lit dans les données OBSERVED (annotation du jeu de données), elle ne se refuse jamais au motif qu'une image n'est pas visible.
2. Localise d'abord le bloc qui porte la quantité demandée :
   - état d'une seule fenêtre → `get_current_inference` (current_state, next_phase, phase_probabilities, duration) ;
   - transitions annotées, leur fenêtre et leur instant → `get_transition_events.transitions` (liste chronologique ; « la dernière transition à/avant la fenêtre courante » est l'entrée dont window_start est le plus grand sans dépasser la fenêtre courante) ;
   - évolution sur plusieurs fenêtres → SERIE TEMPORELLE MULTI-FENETRES : une fenêtre par bloc, chaque valeur préfixée `W<numéro>` ; ne combine jamais deux préfixes différents dans une même affirmation ;
   - toute quantité déjà calculée (minimum, fenêtre de convergence, décalage, retours, sauts, régressions, entropie proche/loin) → ANALYSE DETERMINISTE DE LA SERIE.
3. Trois classes de provenance, jamais mélangées dans une même affirmation : MODEL-DERIVED (estimation du Semi-HMM), OBSERVED (annotation du jeu de données), DERIVED (calcul Python exact sur la série). Une question sur « la séquence prédite », « la prédiction » ou « la stabilité de la prédiction » se répond UNIQUEMENT avec `model_phase` / `model_phase_sequence` et les compteurs MODEL-DERIVED (`n_phase_returns`, `n_phase_skips_in_predicted_sequence`, `n_phase_regressions_in_predicted_sequence`). Une question sur « l'annotation » ou « ce qui a été observé » se répond avec OBSERVED / `get_transition_events`. Un changement de phase présent dans l'annotation n'est pas un changement de la prédiction, et inversement : ne transfère jamais un saut, une transition ou un instant d'une classe à l'autre.
4. Trois quantités distinctes, jamais interchangeables :
   - `offset_from_center` : position d'une fenêtre par rapport à la fenêtre centrale, en nombre de fenêtres ; ce n'est ni un temps ni un décalage modèle/annotation ;
   - `window_start_time` / `window_end_time` : un instant ;
   - `model_vs_observed_lag_windows` : le SEUL décalage entre le modèle et l'annotation. S'il vaut `null`, le décalage est « non calculable » : dis-le, cite la définition imprimée à côté, et n'en déduis jamais un à partir d'un offset, d'un instant ou d'une lecture visuelle des deux séquences.
5. Quand l'ANALYSE DETERMINISTE contient la quantité demandée, utilise-la telle quelle en la nommant (ex. `convergence_start_window`, `crossing_window`, `probability_gap_minimum`, `n_phase_returns`, `entropy_near_minus_far`) ; ne la recalcule pas à partir des fenêtres et ne la remplace pas par un champ voisin au nom ressemblant. Lis la définition imprimée à côté d'un champ avant de l'utiliser : un nom proche des mots de la question ne garantit pas que c'est la bonne quantité.
6. Une valeur explicitement `null` signifie « indisponible / non calculable ». Réponds alors « non calculable », avec la raison imprimée, jamais avec une estimation.
7. Incertitude : compare `entropy_mean_near_center_offset_le_1` (autour de la transition) à `entropy_mean_far_from_center_offset_ge_3` (loin d'elle) ; le signe de `entropy_near_minus_far` donne le verdict (positif : plus incertain autour de la transition ; négatif : moins). Précise que la comparaison porte sur la bande de fenêtres fournie, pas sur l'ensemble du jeu de données.
8. Probabilités : `phase_probability` est la probabilité de la phase prédite ; la deuxième phase la plus probable est `second_phase` (ou le deuxième rang de `phase_probabilities` trié par valeur décroissante) ; `next_phase_distribution` décrit la phase suivante attendue, une quantité différente.
9. Durée écoulée : une durée « depuis » se calcule entre deux instants présents dans le CONTEXTE (l'instant où la phase courante a commencé, par exemple la dernière transition annotée, et l'instant de la fenêtre courante). `expected_duration` est une durée attendue par le modèle, pas une durée écoulée. S'il manque l'un des deux instants, nomme celui qui manque.
10. « Je ne dispose pas de suffisamment d'informations » n'est correct qu'après avoir cherché la quantité dans les blocs ci-dessus ; nomme alors le champ cherché. Quand une partie seulement est disponible, réponds sur cette partie et nomme ce qui manque.

B. FORME DE LA RÉPONSE
- Première phrase : la réponse directe (une valeur, un verdict, ou « non calculable »).
- Ensuite : les valeurs utilisées, chacune avec son identifiant (préfixe `W<n>` ou nom de champ) et sa provenance (MODEL-DERIVED / OBSERVED / DERIVED / DOCUMENT).
- Enfin : les limites (avertissements du CONTEXTE, troncatures, ce qui manque).

C. INTERDICTIONS
1. Utilise UNIQUEMENT les informations du CONTEXTE. N'utilise jamais de connaissance externe pour une valeur numérique, un nom de phase, une métrique, un repère biologique ou un résultat expérimental.
2. N'invente jamais une probabilité, une phase, une métrique, une définition biologique ou une information documentaire. Si le CONTEXTE ne contient pas l'information (après la recherche du point A.10), dis-le explicitement.
3. Distingue toujours les sources DOCUMENT (documentation statique) des sources DYNAMIQUE (Reporting API) ; cite chaque affirmation DOCUMENT par sa source/section et chaque affirmation DYNAMIQUE par sa model_version. N'invente jamais une citation.
4. Signale chaque avertissement présent dans le CONTEXTE ; ne l'omets jamais.
5. Ne présente jamais un signal dérivé du GRU (s'il apparaît) comme une vérité biologique ou une probabilité calibrée.
6. Un raisonnement au-delà de ce que le CONTEXTE affirme est autorisé, mais doit être identifié comme ton propre raisonnement, jamais présenté avec la confiance d'un fait cité.
7. N'affirme jamais qu'une prédiction du modèle a été « observée », ni qu'une transition observée est ce que le modèle « prédit ».
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def _format_document_context(document_context: list) -> str:
    if not document_context:
        return "(aucun document récupéré)"
    lines = []
    for i, chunk in enumerate(document_context, start=1):
        lines.append(
            f"[DOC {i}] source={chunk.get('source')} section={chunk.get('section')!r} "
            f"status={chunk.get('status')}\n{chunk.get('content')}"
        )
    return "\n\n".join(lines)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)  # matches grounding_check._stringify_leaves' own str(value), so a
    # value restated verbatim from one of these lines is always found in the KNOWN set


# --- Priority reordering (docs/RAG_PROMPT_REORDERING_AB_TEST.md, Option A
# of docs/RAG_CONTEXT_FAILURE_ANALYSIS.md sec 7) ---
#
# Real finding: for a question needing get_transition_events' data, the
# relevant from_phase/to_phase/is_skip fields sat at ~82% of the way
# through the flattened prompt, behind ~30 largely-irrelevant
# phase_probabilities/next_phase_distribution numbers from
# get_current_inference (called first, per plan_tools()'s own order).
# llama3.2:latest frequently failed to use that correctly-present data.
#
# Fix: compact/important scalars are rendered FIRST (in their original
# relative order, across ALL tools), bulky same-type distributions
# (phase_probabilities, next_phase_distribution, duration_quantiles) are
# DEFERRED to the end. Every value is still preserved verbatim, in the
# same "path = value" format, addressed by the same JSON path -- only
# the RENDER POSITION changes, never the content, never a value.
_DISTRIBUTION_MIN_SIZE = 4  # a dict must have MORE than this many same-type
# scalar entries to be deferred -- calibrated against the real schemas this
# was designed against (Training/reporting/schemas.py): phase_probabilities/
# next_phase_distribution have 15 entries, duration_quantiles has 5 -- all
# safely deferred; identity-shaped dicts like SampleInfo's own 4 string
# fields (sample_id/video_name/patient_id/split) safely stay immediate.


def _is_homogeneous_scalar_distribution(value: Dict[str, Any]) -> bool:
    """Purely structural test, no field-name knowledge added (matches this
    module's existing "no schema-specific knowledge" discipline): a dict
    counts as a bulky "distribution" only if it has MORE than
    `_DISTRIBUTION_MIN_SIZE` entries, ALL of them scalars, ALL of the SAME
    Python primitive type (None values ignored for the type check, since
    `Optional[float]`-typed fields like frame_start/frame_end can be None
    without changing the field's own type identity). A mixed-type record
    (any real tool's own top-level fields; one get_transition_events
    TransitionEvent's from_phase/to_phase/is_skip/... -- str+int+bool+float
    together) never qualifies, regardless of how many fields it has."""
    if len(value) <= _DISTRIBUTION_MIN_SIZE:
        return False
    if not all(_is_scalar(v) for v in value.values()):
        return False
    types = {type(v) for v in value.values() if v is not None}
    return len(types) <= 1


def _sub_path(path: str, key: Any) -> str:
    key_str = str(key)
    if "." in key_str:
        # A dict key that itself contains "." (e.g. duration_quantiles'
        # own keys, "0.1"/"0.5"/"0.9") would be indistinguishable from
        # an extra nesting level if joined with the same "." separator
        # (path.0.1 could be read as key "0" -> key "1") -- bracket the
        # whole key instead so the path stays a single, unambiguous
        # pointer back into the original dict.
        return f'{path}["{key_str}"]' if path else f'["{key_str}"]'
    return f"{path}.{key_str}" if path else key_str


def _flatten_dynamic_context(value: Any, path: str, lines: List[str], deferred: List[str]) -> None:
    """Renders one JSON leaf per line, addressed by its full dotted/
    bracketed path (e.g. "get_current_inference.current_state.
    phase_probabilities.t6 = 0.5643") instead of nested `json.dumps`
    braces. Purely structural (recursion + key names already present in
    the data), no schema-specific knowledge of any one tool's shape, no
    NLP, no invented text -- every leaf value is preserved verbatim and
    every path is a literal JSON path into the original dict, so the
    transformation stays traceable back to context["dynamic_context"]
    (never a separate, potentially-diverging source of truth).

    Real motivation (docs/LLM_EVALUATION.md sec 0): a small local model
    (llama3.2:latest) failed to locate `t6`'s own probability inside a
    15-entry `phase_probabilities` dict nested several brace-levels deep
    in the raw JSON dump -- flattening turns that lookup into a single
    scannable line instead of a bracket-nesting/indentation exercise.

    Writes into ONE of two target lists (docs/RAG_PROMPT_REORDERING_AB_TEST.md):
    `lines` for compact/important fields (rendered first, in their
    original relative order), `deferred` for bulky same-type
    distributions (`_is_homogeneous_scalar_distribution()`, rendered
    last) -- a dict with only scalar values (a dense distribution like
    `phase_probabilities`/`next_phase_distribution`, or `duration_quantiles`)
    is exploded one key per line into `deferred` when it qualifies;
    everything else (including a mixed-type record like one
    get_transition_events TransitionEvent) is exploded into `lines`,
    unchanged from before this fix. A list of only scalars (e.g.
    `window_starts`, which can be hundreds of entries) is still rendered
    as one compact inline line into `lines`, since exploding it would
    make the context longer, not easier to scan, without helping any
    known real question -- unaffected by this fix, deferral only applies
    to dicts."""
    if isinstance(value, dict):
        if not value:
            lines.append(f"{path} = {{}}")
            return
        if _is_homogeneous_scalar_distribution(value):
            for key, sub in value.items():
                deferred.append(f"{_sub_path(path, key)} = {_format_scalar(sub)}")
            return
        for key, sub in value.items():
            _flatten_dynamic_context(sub, _sub_path(path, key), lines, deferred)
    elif isinstance(value, list):
        if not value or all(_is_scalar(v) for v in value):
            lines.append(
                f"{path} = {json.dumps(value, ensure_ascii=False, default=str)} ({len(value)} item(s))"
            )
        else:
            for i, sub in enumerate(value):
                _flatten_dynamic_context(sub, f"{path}[{i}]", lines, deferred)
    else:
        lines.append(f"{path} = {_format_scalar(value)}")


# --- Structured temporal rendering (2026-09-02 context restructuring) ---
#
# Two dynamic_context keys are rendered by `temporal_context`'s
# window-blocked renderers instead of the generic path flattener above:
# `get_inference_history` (a MULTI-WINDOW series, where generic
# `entries[i]...` indexing detaches every value from its real window id)
# and `series_analysis` (the deterministic per-series arithmetic the LLM
# must not be asked to redo). Every OTHER tool output keeps the exact
# flattening + deferral behaviour it had before -- this is an added
# special case for two known series-shaped payloads, not a replacement of
# the general mechanism.
_STRUCTURED_KEYS = ("get_inference_history", "series_analysis")


def _center_window_reconciliation(dynamic_context: dict) -> List[str]:
    """Audit defect B: when both tools are present, the SAME window is
    described twice (get_current_inference with the full distributions,
    the series' center entry with the top-k ones). Neither is dropped --
    dropping would lose the duration/frame fields only the single-window
    record carries -- so the identity is stated explicitly instead of
    left for the reader to infer."""
    history = dynamic_context.get("get_inference_history")
    current = dynamic_context.get("get_current_inference")
    if not isinstance(history, dict) or not isinstance(current, dict):
        return []
    center = history.get("center_window")
    window = (current.get("window") or {}).get("window_start")
    if center is None or window is None or center != window:
        return []
    return [
        f"NOTE DE RECONCILIATION : get_current_inference decrit EXACTEMENT la meme fenetre "
        f"que W{center} ci-dessus (fenetre centrale). Ses distributions sont completes "
        f"(15 phases) la ou celles de la serie sont tronquees au top-k ; ce ne sont pas deux "
        f"fenetres differentes, et une valeur de get_current_inference ne doit jamais etre "
        f"attribuee a une autre fenetre que W{center}.",
        "",
    ]


def _format_dynamic_context(dynamic_context: dict) -> str:
    if not dynamic_context:
        return "(aucune donnée dynamique récupérée)"
    structured: List[str] = []
    lines: List[str] = []
    deferred: List[str] = []
    for tool_name, value in dynamic_context.items():
        if tool_name == "get_inference_history" and isinstance(value, dict):
            structured.append(render_inference_history(value))
            structured.append("")
        elif tool_name == "series_analysis" and isinstance(value, dict):
            structured.append(render_series_analysis(value))
            structured.append("")
        else:
            _flatten_dynamic_context(value, tool_name, lines, deferred)
    if structured:
        structured.extend(_center_window_reconciliation(dynamic_context))
    return "\n".join(structured + lines + deferred)


# ---------------------------------------------------------------------------
# COMPACT_CONTEXT_V1 (2026-09-11) -- an ALTERNATIVE rendering of the same
# context dict, selected per call. The historical rendering ("v2") is the
# default and is byte-identical to what it was before this addition: every
# existing caller, test and artefact is unaffected unless it explicitly asks
# for the compact format.
#
# What changes in "compact", and nothing else:
#   1. the DOCUMENT (RAG) block is de-narrativised by `compact_context.py`
#      (deterministic line filter) and re-rendered as labelled
#      `[RAG DOCUMENT n]` records with explicit source/section/status/content
#      fields instead of one run-on header line + raw paragraph;
#   2. a short PROVENANCE legend naming the classes already present in the
#      context is printed before the dynamic block -- names of blocks only,
#      never a value, never a phase, never a number.
# The DYNAMIC block itself is produced by the SAME `_format_dynamic_context()`
# in both formats: no value is moved, reordered, rounded or dropped. The
# experiment's single variable is the document half.
CONTEXT_FORMAT_ENV_VAR = "EMBRYO_LLM_CONTEXT_FORMAT"
# "compact_v2" (2026-09-12): provenance-grouped rendering with LOCAL labels,
# curated project chunks, a separate [SCIENTIFIC] block for the Istanbul
# Consensus and a [LIMITATIONS] block -- see compact_v2_render.py. "v2" and
# "compact" are untouched so every historical artefact stays reproducible.
SUPPORTED_CONTEXT_FORMATS = (CONTEXT_FORMAT_V2, CONTEXT_FORMAT_COMPACT, CONTEXT_FORMAT_COMPACT_V2)


def resolve_context_format(explicit: Optional[str] = None) -> str:
    """explicit > $EMBRYO_LLM_CONTEXT_FORMAT > CONTEXT_FORMAT_V2.

    Fails loudly on an unknown value rather than silently falling back to the
    default: a benchmark arm that silently ran the wrong renderer would be
    indistinguishable from a null result."""
    for value in (explicit, os.environ.get(CONTEXT_FORMAT_ENV_VAR)):
        if value is None or str(value).strip() == "":
            continue
        candidate = str(value).strip().lower()
        if candidate not in SUPPORTED_CONTEXT_FORMATS:
            raise ValueError(
                f"Unknown context format {value!r}; supported: {SUPPORTED_CONTEXT_FORMATS}."
            )
        return candidate
    return CONTEXT_FORMAT_V2


def _format_document_context_compact(document_context: list) -> str:
    """COMPACT_CONTEXT_V1's document block. Structure over prose: one record
    per chunk, one field per line, the filtered content last. A chunk whose
    content was entirely narrative is NOT hidden -- it is printed as a stub
    with its source and the rule that removed it, so a retrieval that returned
    project narrative for a clinical question stays visible and auditable."""
    if not document_context:
        return "(aucun document recupere)"
    blocks = []
    for i, chunk in enumerate(compact_document_context(document_context), start=1):
        head = (
            f"[RAG DOCUMENT {i}]\n"
            f"source: {chunk.get('source')}\n"
            f"section: {chunk.get('section')}\n"
            f"status: {chunk.get('status')}"
        )
        if chunk.get("compact_dropped"):
            blocks.append(head + "\n" + MARKER_TEMPLATES["dropped_chunk"].format(
                reason=chunk.get("compact_drop_reason"), rule=chunk.get("compact_drop_rule")))
            continue
        body = f"{head}\ncontent:\n{chunk.get('compact_content')}"
        if chunk.get("compact_dropped_lines"):
            body += "\n" + MARKER_TEMPLATES["dropped_lines"].format(
                n=chunk["compact_dropped_lines"],
                rules=",".join(chunk.get("compact_rules_applied") or []))
        blocks.append(body)
    return "\n\n".join(blocks)


# Provenance legend, emitted in compact format only. Each entry is
# (dynamic_context key that must be present, class, the field prefixes).
# Pure labelling of blocks that are ALREADY in the context: it states no
# value, no phase name and no number, and it never moves a field from one
# class to another (a line that was MODEL-DERIVED stays MODEL-DERIVED).
_PROVENANCE_LEGEND = (
    ("get_transition_events", "OBSERVED", "get_transition_events.* (annotation)"),
    ("get_current_inference", "OBSERVED", "get_current_inference.ground_truth.* (annotation)"),
    ("get_current_inference", "MODEL-DERIVED",
     "get_current_inference.current_state/next_phase/duration.* (Semi-HMM)"),
    ("get_trajectory", "MODEL-DERIVED", "get_trajectory.transition_chain.* (Semi-HMM)"),
    ("get_inference_history", "MODEL-DERIVED", "blocs WINDOW <n>, section MODEL-DERIVED"),
    ("get_inference_history", "OBSERVED", "blocs WINDOW <n>, section OBSERVED"),
    ("series_analysis", "DERIVED", "ANALYSE DETERMINISTE DE LA SERIE (calcul exact)"),
)


def _format_provenance_legend(dynamic_context: dict, document_context: list) -> str:
    """Compact format only. Names the provenance class of blocks ALREADY in
    the context -- it states no value, no phase, no number, and it never moves
    a field from one class to another (pinned by
    Tests/orchestrator/test_prompt_compact_format.py)."""
    lines = [f"- {klass} : {description}"
             for key, klass, description in _PROVENANCE_LEGEND
             if key in (dynamic_context or {})]
    if document_context:
        lines.append("- DOCUMENT : bloc CONTEXTE SCIENTIFIQUE (documentation, jamais une "
                     "mesure sur cet embryon)")
    if not lines:
        return ""
    return "PROVENANCE DES BLOCS (rappel de classement, aucune valeur) :\n" + "\n".join(lines) + "\n\n"


def build_user_prompt(context: Dict[str, Any], context_format: Optional[str] = None) -> str:
    """Renders `context_builder.build_context()`'s output into one plain
    text block: QUESTION, CONTEXTE DOCUMENT (RAG), CONTEXTE DYNAMIQUE
    (Reporting API), AVERTISSEMENTS -- each explicitly labeled, matching
    the system prompt's rule 3 (never blend the two).

    `context_format` selects the renderer (see `resolve_context_format`);
    omitted, it is the historical "v2" rendering, unchanged."""
    resolved = resolve_context_format(context_format)
    warnings = context.get("warnings") or []
    warnings_block = "\n".join(f"- {w}" for w in warnings) if warnings else "(aucun)"
    document_context = context.get("document_context", [])
    dynamic_context = context.get("dynamic_context", {})

    if resolved == CONTEXT_FORMAT_COMPACT_V2:
        from .compact_v2_render import render_compact_v2  # noqa: WPS433 -- lazy: that module imports helpers from here
        return render_compact_v2(context)

    if resolved == CONTEXT_FORMAT_COMPACT:
        return (
            f"QUESTION:\n{context.get('question', '')}\n\n"
            f"CONTEXTE SCIENTIFIQUE (RAG, documentation statique ; "
            f"narration projet/publication ecartee et comptee) :\n"
            f"{_format_document_context_compact(document_context)}\n\n"
            f"{_format_provenance_legend(dynamic_context, document_context)}"
            f"CONTEXTE DYNAMIQUE (Reporting API, données en direct) :\n"
            f"{_format_dynamic_context(dynamic_context)}\n\n"
            f"AVERTISSEMENTS (à signaler dans la réponse, jamais à omettre) :\n"
            f"{warnings_block}\n"
        )

    return (
        f"QUESTION:\n{context.get('question', '')}\n\n"
        f"CONTEXTE DOCUMENT (RAG, connaissance statique documentée) :\n"
        f"{_format_document_context(document_context)}\n\n"
        f"CONTEXTE DYNAMIQUE (Reporting API, données en direct) :\n"
        f"{_format_dynamic_context(dynamic_context)}\n\n"
        f"AVERTISSEMENTS (à signaler dans la réponse, jamais à omettre) :\n"
        f"{warnings_block}\n"
    )

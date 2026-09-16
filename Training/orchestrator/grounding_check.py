"""
Executable hallucination boundary -- task's own KNOWN / INFERRED / UNKNOWN
policy (Phase 6 sec 6), and the concrete implementation of
`docs/LLM_PROMPT_CONTRACT.md`'s "Grounding check (design, not
implemented)" section, now implemented here.

Provider-agnostic: works on any `LLMProvider`'s generated text, given the
same `context` dict (`context_builder.build_context()`'s output) that
produced it. Never calls an LLM itself, never modifies `context`, never
modifies the answer text -- it only REPORTS what it found. The caller
(`orchestrator.answer_question()`) decides whether to flag, strip, or
regenerate.

Scope, stated honestly: this module automatically verifies two concrete,
checkable claim types -- **numbers** and **phase names** -- against the
turn's real tool outputs. It does NOT attempt to automatically verify
qualitative reasoning claims (whether a sentence is a fair MODEL
INTERPRETATION vs. unfounded LLM INFERENCE) -- that distinction is
enforced by the system prompt's own instruction
(`prompt.py`/`docs/LLM_PROMPT_CONTRACT.md`), not by NLP here. Building an
automatic qualitative-claim verifier would be exactly the "systeme NLP
complexe" earlier phases of this project were explicitly told not to
build (`Training/orchestrator/router.py`'s own docstring, same
discipline applied here).

Phase-name verification has two layers, not one (added
`docs/RAG_LLM_QUALITY_REPORT.md` P0 item 1, `docs/RAG_LLM_IMPROVEMENT_REPORT.md`):
(1) `_known_phase_tokens()`'s loose "known anywhere" set (document text,
the question, or dynamic tool output) -- unchanged, still the right
signal for a pure documentary discussion with no live data this turn;
(2) `_dynamic_phase_tokens()`'s stricter "live claim" set, used ONLY when
this turn actually fetched live phase data, to catch a wrong phase name
that happens to also be textually present in retrieved documentation
(the concrete false-negative that motivated this: a wrong phase passed
as grounded because the correct AND the wrong phase name were both
enumerated in a vocabulary table the RAG retrieval happened to surface).
See `check_grounding()`'s live-conflict block below for the exact rule.

Two DETERMINISTIC false negatives found by the 2026-09-07c replication
(reproduced 4/4 on byte-identical context) are fixed here, and nowhere
else in the pipeline:

  A. A transition pair the MODEL predicts (`series_analysis.convergence.
     all_model_phase_changes`, e.g. `t7 -> t8`) was always ungrounded,
     because `_real_observed_transitions()` is the only reference set and
     it holds ANNOTATED transitions only. The model's own changes are now
     a SECOND, separately-labelled reference set -- the MODEL-DERIVED /
     OBSERVED distinction is not dropped, it becomes explicit through the
     new `transition_claim_provenance` report (`matches` = "observed" /
     "model" / "both" / "none").
  B. A phase name that is only a KEY of a probability distribution
     (`phase_probabilities`) was invisible to `_collect_strings`, which
     walks dict VALUES. A correct answer naming the model's second-most-
     probable phase was therefore ungrounded 4/4. Only the TOP-2 keys of
     a dict that is structurally a phase distribution now count -- see
     `_phase_distribution_ranked_tokens()` for why not all 15.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .router import PHASE_TOKEN_PATTERN, extract_phase_tokens

# Matches integers/decimals, comma- or dot-separated, with an optional
# trailing percent sign -- e.g. "0.72", "72%", "12", "1,5". Deliberately
# permissive (a superset of what any provider is likely to write for a
# probability/duration/metric), since a missed real number is worse than
# a spurious low-value one (window indices, quantile levels) being
# checked and found grounded anyway.
#
# Leading sign accepts ASCII hyphen-minus (U+002D) OR the Unicode minus
# sign (U+2212) -- found via a real case (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md
# sec 10/11): `docs/GLOBAL_MODEL_COMPARISON.md`'s own prose writes negative
# deltas with a typographic minus ("Brier −0.0004"), which this pattern
# did not previously recognize as a sign at all (it extracted "0.0004",
# positive). Adding U+2212 does not introduce a new risk category: an
# unspaced subtraction expression immediately adjacent to a digit
# (e.g. "window_size-1=7") was ALREADY misread as a negative literal
# ("-1") for the ASCII hyphen before this change -- confirmed by testing
# the unmodified pattern -- so this only makes the Unicode minus sign
# behave consistently with the ASCII one, not qualitatively differently.
_NUMBER_PATTERN = re.compile(r"[-−]?\d+(?:[.,]\d+)?%?")


def _strip_phase_tokens(text: str) -> str:
    """Removes every phase-name match (e.g. "t6", "t9+") before number
    extraction, so the digit embedded in a phase name is never
    double-counted as a separate, free-floating numeric claim -- found
    via a real test (a grounded "t6"/"t9+" answer was being flagged
    ungrounded because "6"/"9" alone didn't appear anywhere in the
    context, even though the phase name itself was perfectly grounded)."""
    return PHASE_TOKEN_PATTERN.sub(" ", text or "")

# Numbers this common they would trigger constant false positives if
# treated as "claims requiring grounding" (list positions, small counts
# a provider might mention in passing, e.g. "2 sources") -- excluded from
# the ungrounded-number report, never from extraction itself (still
# visible in `checked_numbers` for full transparency).
_IGNORED_NUMBERS = {"0", "1", "2", "3"}


def extract_numbers(text: str) -> List[str]:
    return _NUMBER_PATTERN.findall(text or "")


def _stringify_leaves(value: Any, out: Set[str]) -> None:
    """Recursively collect every scalar leaf of a (possibly nested)
    dict/list, stringified -- the KNOWN set for a tool's raw output. A
    float is stringified via `repr`-free `str()` (e.g. 0.72 -> "0.72"),
    matching how a provider restating it verbatim would most plausibly
    write it; near-miss formatting differences (rounding, "72%" for
    0.72) are a known limitation, see docs/GROUNDED_GENERATION.md."""
    if isinstance(value, dict):
        for v in value.values():
            _stringify_leaves(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _stringify_leaves(v, out)
    elif isinstance(value, bool):
        return  # bool numbers ("True"/"False") are never a numeric claim
    elif isinstance(value, (int, float)):
        out.add(str(value))
    elif isinstance(value, str):
        out.update(extract_numbers(value))


def _known_numbers_from_dynamic_context(dynamic_context: Dict) -> Set[str]:
    known: Set[str] = set()
    _stringify_leaves(dynamic_context or {}, known)
    return known


def _canonical_number_form(token: str) -> str:
    """Equality-comparison form ONLY -- never used for extraction or for
    anything reported back to a caller (`checked_numbers`/`ungrounded_numbers`
    always show the LLM's own original text verbatim, unmodified).

    Replaces a comma decimal separator with a period, so "1,0" and "1.0"
    compare equal, AND a leading Unicode minus sign (U+2212) with an ASCII
    hyphen-minus, so "−0.0004" (as real project docs write it) and
    "-0.0004" (as an LLM/answer would write it) compare equal too. Safe
    specifically because `_NUMBER_PATTERN` (`[-−]?\\d+(?:[.,]\\d+)?%?`)
    structurally admits AT MOST ONE separator character and AT MOST ONE
    leading sign character per matched token -- there is no way this
    regex could have matched a thousands-grouped form like "1,234,567";
    any comma present in a token it produced is therefore unambiguously a
    decimal separator, never a grouping separator, so this substitution
    can never misread a different number as the one being checked.

    Deliberately NOT a general numeric-equivalence check: no float
    parsing, no rounding, no scientific-notation handling. "0.1" and
    "0.10" remain distinct strings after this function (a real, formal
    numeric-equivalence comparison was explicitly out of scope for this
    fix -- see docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md).

    Applied to two KNOWN sets: `dynamic_context`-derived numbers (below)
    and, since sec 6/10 of the report, numeric tokens EXTRACTED from
    `doc_text` via `_known_numbers_from_document_text` -- never to the raw
    `doc_text` string itself. That distinction is the safety boundary:
    free-form RAG/document prose is never comma->period substituted
    wholesale (a comma there can be a thousands separator, list
    enumeration, or ordinary punctuation); only tokens `_NUMBER_PATTERN`
    already judged to be a single plausible number (comma immediately
    followed by a digit, no space) are ever canonicalized."""
    return token.replace(",", ".").replace("−", "-")


def _known_numbers_canonical(dynamic_context: Dict) -> Set[str]:
    return {_canonical_number_form(n) for n in _known_numbers_from_dynamic_context(dynamic_context)}


def _known_numbers_from_document_text(doc_text: str) -> Set[str]:
    """Numeric tokens extracted from the retrieved RAG document text,
    using the SAME `_NUMBER_PATTERN` extraction as the LLM's own answer
    text -- this is what makes normalizing the document side safe where a
    blind, text-wide comma->period substitution over free-form prose
    would NOT be (docs/GROUNDING_NUMERIC_NORMALIZATION_REPORT.md sec 6's
    explicitly flagged risk). `_NUMBER_PATTERN` only merges a comma into
    a numeric token when it is immediately followed by a digit (no
    space) -- so "t6, puis t7" or "1, 2, 3" (comma-space, ordinary
    punctuation/enumeration) never produce a merged token, while a
    genuine decimal-comma number like "0,005" or "Brier=0,11483" does.
    Same structural guarantee as `_canonical_number_form`'s docstring:
    at most one separator per matched token, so a comma this function
    ever canonicalizes is unambiguously a decimal separator here too."""
    return set(extract_numbers(doc_text))


def _known_numbers_from_document_text_canonical(doc_text: str) -> Set[str]:
    return {_canonical_number_form(n) for n in _known_numbers_from_document_text(doc_text)}


def _reference_text(context: Dict) -> str:
    """Retrieved project chunks PLUS, since compact_v2 (2026-09-12), the
    Istanbul Consensus blocks of `context["scientific_context"]` -- a block id,
    a page number or a median timing quoted back from that block is presence,
    not invention. Contexts without the key are unchanged."""
    text = _document_text((context or {}).get("document_context", []))
    scientific = (context or {}).get("scientific_context")
    if scientific:
        from .scientific_reference import scientific_context_text  # noqa: WPS433
        text = text + " " + scientific_context_text(scientific)
    return text


def _document_text(document_context: List[Dict]) -> str:
    """Chunk content PLUS citation metadata (source, section) -- a
    section number quoted back in a citation (e.g. "[Source: ...
    sec '16']") is legitimate provenance, not a fabricated numeric
    claim, so it must count as grounded too. Found via a real test:
    a contradictory-context answer citing two real sources was flagged
    ungrounded solely because one chunk's own section number ("16")
    happened not to appear inside that chunk's `content` string."""
    parts: List[str] = []
    for c in document_context or []:
        parts.append(c.get("content") or "")
        parts.append(str(c.get("section") or ""))
        parts.append(str(c.get("source") or ""))
    return " ".join(parts)


# --- Phase names that exist only as distribution KEYS (fix B, 2026-09-07d) ---
#
# `_collect_strings` walks dict VALUES, so every phase name inside
# `phase_probabilities` / `next_phase_distribution` (`Dict[str, float]`) was
# invisible to both phase sets. Real, reproducible consequence (4/4 runs of
# the 2026-09-07c replication, benchmark Q4): an answer correctly naming the
# model's SECOND most probable phase was reported ungrounded, because that
# phase is a key and appears nowhere as a value.
#
# The obvious repair -- "count every key" -- is refused on purpose: the model
# returns a posterior over all 15 states every turn, so every phase name would
# become "live", and the live-conflict check
# (docs/RAG_LLM_QUALITY_REPORT.md P0 item 1) could never flag anything again.
# Measured on the real artefacts, the tail is exactly what must stay out: at
# the benchmark anchor `t4` carries p=3.8e-25 while being the phase the C2
# false-negative case wrongly claims.
#
# So only the TOP-2 entries of a distribution count. That is not a new
# threshold invented here: `temporal_context._entry_view()` ALREADY promotes
# exactly the top-2 of each window's distribution to explicit `model_phase` /
# `second_phase` VALUES, which are therefore already grounded through the
# normal path. This fix removes an asymmetry -- the single-window
# `get_current_inference` payload has no such promotion -- rather than adding
# a new rule. Measured effect on the real contexts: the live set grows by 1-3
# tokens per question (e.g. 3 -> 4 on Q4), never to the full vocabulary.
_PHASE_DISTRIBUTION_TOP_N = 2


def _is_phase_distribution(value: Any) -> bool:
    """Deliberately NARROW, and narrower than a generic "dict of scalars"
    test: a dict qualifies only if EVERY key is a phase token and EVERY
    value is a real number (bools excluded -- `True` is not a probability).
    A dict that fails either half is walked normally, so no non-phase key
    can ever be promoted to a phase token by this function."""
    if not isinstance(value, dict) or not value:
        return False
    for key, item in value.items():
        if not isinstance(key, str) or not PHASE_TOKEN_PATTERN.fullmatch(key):
            return False
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
    return True


def _phase_distribution_ranked_tokens(value: Any, out: Set[str]) -> None:
    """Collects the lower-cased top-`_PHASE_DISTRIBUTION_TOP_N` keys of every
    phase distribution reachable in `value`. Ties are broken by phase name so
    the result is deterministic across runs and platforms, never dict
    insertion order."""
    if isinstance(value, dict):
        if _is_phase_distribution(value):
            ranked = sorted(value.items(), key=lambda kv: (-kv[1], kv[0]))
            out.update(key.lower() for key, _ in ranked[:_PHASE_DISTRIBUTION_TOP_N])
            return
        for item in value.values():
            _phase_distribution_ranked_tokens(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _phase_distribution_ranked_tokens(item, out)


def _known_phase_tokens(context: Dict) -> Set[str]:
    """A phase name is grounded if it appears anywhere in the turn's own
    dynamic tool output, the retrieved document text, or the question
    itself (a provider answering "t6" because the user asked about t6 is
    not inventing a phase name)."""
    known: Set[str] = set()
    known.update(extract_phase_tokens(_reference_text(context)))
    known.update(extract_phase_tokens(context.get("question", "") or ""))
    dyn_text_parts: List[str] = []
    _collect_strings(context.get("dynamic_context", {}) or {}, dyn_text_parts)
    known.update(extract_phase_tokens(" ".join(dyn_text_parts)))
    _phase_distribution_ranked_tokens(context.get("dynamic_context", {}) or {}, known)
    return known


def _dynamic_phase_tokens(context: Dict) -> Set[str]:
    """Phase tokens present ONLY in this turn's live Reporting API output
    (`context["dynamic_context"]`) -- e.g. `current_state.current_phase`,
    `next_phase.most_likely_next_phase`, `ground_truth.ground_truth_phase`,
    `transition_chain` entries (`Training/reporting/schemas.py`). Excludes
    `document_context` and the question on purpose: this is the
    authoritative set used by the live-conflict check in
    `check_grounding()` below, distinct from `_known_phase_tokens()`'s
    loose "known anywhere" set used for the general per-token check.

    Since 2026-09-07d this set ALSO contains the top-2 keys of every
    `phase_probabilities`-shaped dict (`_phase_distribution_ranked_tokens`)
    -- previously no distribution key was reachable at all, because
    `_collect_strings` walks dict VALUES. The restriction to the top 2 is
    what preserves this check's power: the 13 negligible tail states of the
    posterior stay OUT of the live set, so the C2 case this check exists for
    (a wrong phase that is merely enumerated in retrieved documentation) is
    still caught."""
    parts: List[str] = []
    _collect_strings(context.get("dynamic_context", {}) or {}, parts)
    live = set(extract_phase_tokens(" ".join(parts)))
    _phase_distribution_ranked_tokens(context.get("dynamic_context", {}) or {}, live)
    return live


def _collect_strings(value: Any, out: List[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_strings(v, out)
    elif isinstance(value, str):
        out.append(value)


# --- Relational grounding for transition claims (docs/RELATIONAL_GROUNDING.md) ---
#
# extract_phase_tokens()/PHASE_TOKEN_PATTERN above verify individual
# phase-name TOKENS, never the RELATION between two of them -- a real,
# concrete false-negative found via real validation
# (docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md sec 10, question Q3):
# "il y a une transition entre phases t7 et t4" passed grounded=True
# because t7 and t4 each individually appear somewhere in this turn's
# dynamic_context (t7 as the model's current_phase, t4 as the real
# observed transition's from_phase) -- but the ORDERED PAIR (t7, t4) is
# not a real transition; the real one is (t4, t6).
#
# Deliberately NOT a general NLP parser (this project's own explicit
# discipline, router.py's own docstring) -- a narrow, explicit connector
# vocabulary, checked EXACTLY (after stripping accents/whitespace)
# against the text strictly BETWEEN two consecutive phase-token matches.
# Covers the 5 phrasings explicitly named in the task brief ("t4 -> t6",
# "t4 vers t6", "de t4 a t6", "transition t4-t6", "passage de t4 a t6")
# plus the exact real Q3 phrasing ("entre t7 et t4") via the dedicated
# "entre ... et" wrapper check below. Deliberately EXCLUDES bare
# enumeration connectors ("," "et"/"and" alone) so "t4, t6 apparaissent
# ici" or "t4 et t6" (no "entre") is never misread as a transition claim
# -- the task's own explicit anti-overcorrection requirement.
_TRANSITION_CONNECTORS = {"->", "→", "vers", "a", "-"}
_ENTRE_WORD = "entre"
_ENTRE_LOOKBACK_CHARS = 20  # how far before the first token to look for "entre"


def _normalize_ascii_lower(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def extract_transition_claims(text: str) -> List[Tuple[str, str]]:
    """Best-effort, deliberately narrow extraction of ORDERED phase-token
    pairs the answer text presents as a transition. Returns lower-cased
    (from, to) tuples, duplicates preserved (order in text). NOT a
    general parser -- see module-level comment above for exactly what it
    does and does not catch."""
    text = text or ""
    matches = list(PHASE_TOKEN_PATTERN.finditer(text))
    pairs: List[Tuple[str, str]] = []
    for a, b in zip(matches, matches[1:]):
        between = _normalize_ascii_lower(text[a.end():b.start()])
        is_connector = between in _TRANSITION_CONNECTORS
        if not is_connector and between in ("et", "and"):
            # "entre X et Y" ("between X and Y") unambiguously names a
            # relation, unlike a bare "X et Y" enumeration -- only
            # counted when "entre" actually appears shortly before the
            # first token, never inferred from "et"/"and" alone.
            lookback = _normalize_ascii_lower(text[max(0, a.start() - _ENTRE_LOOKBACK_CHARS):a.start()])
            is_connector = _ENTRE_WORD in lookback
        if is_connector:
            pairs.append((a.group(0).lower(), b.group(0).lower()))
    return pairs


_SKIP_CLAIM_MARKERS = ("skip", "saut de phase", "saute")  # narrow, explicit -- same
# pattern as router.py's own _MODEL_METADATA_TRIGGERS-style keyword sets, not a
# sentiment/intent classifier.


def _text_claims_skip(text: str) -> bool:
    normalized = _normalize_ascii_lower(text or "")
    return any(marker in normalized for marker in _SKIP_CLAIM_MARKERS)


def _real_observed_transitions(context: Dict):
    """The turn's real get_transition_events() output, verbatim -- the
    ONLY source of truth for relational grounding (task's own "la source
    de verite doit rester observed transition data" -- never
    recomputed/re-derived here). Returns None when get_transition_events
    was NOT called this turn -- nothing to verify a claim against, a
    graceful no-op (same "only fires when this turn actually fetched live
    data" discipline as the existing live-conflict check). Returns a
    list, possibly EMPTY, when it WAS called: an empty list here is real
    information ("no transition observed this turn"), not an absence of
    information -- a claimed pair still gets flagged ungrounded against
    a real empty list, correctly."""
    dynamic_context = context.get("dynamic_context", {}) or {}
    tool_output = dynamic_context.get("get_transition_events")
    if not isinstance(tool_output, dict):
        return None
    return list(tool_output.get("transitions", []) or [])


def _real_model_transitions(context: Dict):
    """The turn's own MODEL-DERIVED phase changes, from
    `series_analysis.convergence.all_model_phase_changes` -- computed
    deterministically by `temporal_context.derive_series_analysis()` from
    `get_inference_history`, never recomputed or re-derived here (same
    discipline as `_real_observed_transitions()` above).

    Returns None when `series_analysis` was NOT produced this turn (a
    single-window question: nothing model-side to verify a relation
    against). Returns a list, possibly EMPTY, when it WAS: an empty list
    is real information ("the model's phase never changes inside this
    band"), so a claimed model pair is correctly flagged against it.

    WHY this exists (2026-09-07c, reproduced across runs): the model's own
    `t7 -> t8` change is printed in the prompt, an answer restating it is
    correct, and it was nonetheless reported ungrounded -- because the only
    reference set was the ANNOTATED transition list, which by construction
    can never contain a model prediction. This does NOT merge the two
    classes: they stay two separate sets, and which one matched is reported
    per claim in `transition_claim_provenance`."""
    dynamic_context = context.get("dynamic_context", {}) or {}
    analysis = dynamic_context.get("series_analysis")
    if not isinstance(analysis, dict):
        return None
    convergence = analysis.get("convergence") or {}
    return list(convergence.get("all_model_phase_changes", []) or [])


def _model_skip_transitions(context: Dict):
    """`series_analysis.stability.phase_skips` -- the subset of the model's
    own phase changes that ARE skips, as counted by
    `n_phase_skips_in_predicted_sequence`.

    Load-bearing for the skip check: without it, accepting model pairs
    would have turned a real hallucination into a grounded answer (a run of
    the 2026-09-07c replication answered "il y a un saut de phase" about
    `t7 -> t8`, a distance-1 change with `n_phase_skips = 0`). The skip
    claim is verified against the model's own skip list, exactly as an
    observed skip claim is verified against `is_skip`."""
    dynamic_context = context.get("dynamic_context", {}) or {}
    analysis = dynamic_context.get("series_analysis")
    if not isinstance(analysis, dict):
        return None
    stability = analysis.get("stability") or {}
    return list(stability.get("phase_skips", []) or [])


def _pairs(changes) -> Set[Tuple[Any, Any]]:
    return {(c.get("from_phase"), c.get("to_phase")) for c in (changes or [])}



# ---------------------------------------------------------------------------
# Grounding checks v3 -- PROVENANCE (added after the three-condition benchmark,
# docs/VALIDATOR_V1_IMPLEMENTATION.md)
#
# WHY THIS IS SEPARATE FROM `grounded`
# The PREDICTION_ONLY run produced answers that are grounded and wrong: on Q3,
# 3 replications out of 3 restated the Semi-HMM's `current_phase = t7` as "la
# phase dominante juste avant la transition etait t7", while the annotation
# says t4. `t7` IS in the context, so every presence check passes -- correctly,
# because presence is what it measures.
#
# So `grounded` is NOT redefined here. Its meaning ("every checkable number,
# phase and transition in the answer exists in this turn's real tool output")
# is unchanged, and every test written against it keeps passing. What is added
# is a SECOND, independent report: does the claim's FRAMING match the
# provenance of the field that carries it? `provenance_consistent` is a
# separate property precisely so that "grounded != scientifically true" stays
# visible in the artefact instead of being collapsed into one boolean.
#
# Deliberately narrow: phase tokens only, reported not enforced. The full
# claim-level version lives in `Training/validator/`, which is additive and
# which this module does NOT import -- production must not depend on it.
# ---------------------------------------------------------------------------

PROVENANCE_OBSERVED = "OBSERVED"
PROVENANCE_MODEL_DERIVED = "MODEL-DERIVED"

_FRAMING_MODEL_MARKERS = (
    "le modele predit", "le modele prevoit", "selon le modele", "prediction",
    "predit", "predite", "model-derived", "semi-hmm", "current_phase",
    "current_state", "next_phase", "phase predite", "sequence predite",
)
_FRAMING_OBSERVED_MARKERS = (
    "annotation", "annote", "annotee", "annotees", "observe", "observee",
    "observed", "ground_truth", "jeu de donnees", "embryologiste",
    "get_transition_events", "verite terrain",
)


def _framing_of(text: str) -> Set[str]:
    """Which provenance class the ANSWER TEXT attributes itself to. An empty
    set means a bare assertion -- a statement of fact about the embryo."""
    low = _normalize_ascii_lower(text)
    found = set()
    if any(m in low for m in _FRAMING_MODEL_MARKERS):
        found.add(PROVENANCE_MODEL_DERIVED)
    if any(m in low for m in _FRAMING_OBSERVED_MARKERS):
        found.add(PROVENANCE_OBSERVED)
    return found


def _phase_tokens_by_class(context: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Per provenance class, the phase names that class actually carries in
    THIS context. Membership is STRUCTURAL -- which tool, which key -- never a
    token match, which is the whole point (a token match is what made `t7`
    look acceptable)."""
    dynamic = (context or {}).get("dynamic_context") or {}
    observed: Set[str] = set()
    model: Set[str] = set()

    def phases_of(value: Any) -> Set[str]:
        strings: List[str] = []
        _collect_strings(value, strings)
        keys: Set[str] = set()
        _phase_keys(value, keys)
        return set(extract_phase_tokens(" ".join(strings))) | keys

    current = dynamic.get("get_current_inference") or {}
    if isinstance(current, dict):
        if current.get("ground_truth"):
            observed |= phases_of(current["ground_truth"])
        for key in ("current_state", "next_phase", "duration"):
            if current.get(key):
                model |= phases_of(current[key])

    if dynamic.get("get_transition_events"):
        observed |= phases_of(dynamic["get_transition_events"])

    history = dynamic.get("get_inference_history") or {}
    if isinstance(history, dict):
        for entry in history.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("observed"):
                observed |= phases_of(entry["observed"])
            if entry.get("model_derived"):
                model |= phases_of(entry["model_derived"])

    if dynamic.get("observed_series_analysis"):
        observed |= phases_of(dynamic["observed_series_analysis"])

    for key in ("series_analysis", "model_series_analysis"):
        series = dynamic.get(key)
        if not isinstance(series, dict):
            continue
        for sub in ("per_window", "sequences", "convergence", "stability"):
            if series.get(sub):
                model |= phases_of(series[sub])
        per_window = series.get("per_window") or {}
        if isinstance(per_window, dict):
            for entry in per_window.values():
                if isinstance(entry, dict) and entry.get("observed_phase"):
                    observed.add(entry["observed_phase"])
                    model.discard(entry["observed_phase"])
        sequences = series.get("sequences") or {}
        if isinstance(sequences, dict):
            if sequences.get("observed_annotation_phase_sequence"):
                observed |= phases_of(sequences["observed_annotation_phase_sequence"])
        timing = series.get("timing") or {}
        if isinstance(timing, dict) and timing.get("all_observed_phase_changes"):
            observed |= phases_of(timing["all_observed_phase_changes"])
    return {PROVENANCE_OBSERVED: observed, PROVENANCE_MODEL_DERIVED: model}


def _phase_keys(value: Any, out: Set[str]) -> None:
    """Phase names reachable only as dict KEYS (`phase_probabilities`), which
    `_collect_strings` cannot see because it walks VALUES."""
    if isinstance(value, dict):
        for key, sub in value.items():
            if isinstance(key, str) and PHASE_TOKEN_PATTERN.fullmatch(key):
                out.add(key)
            _phase_keys(sub, out)
    elif isinstance(value, list):
        for sub in value:
            _phase_keys(sub, out)


def check_phase_provenance(answer_text: str, context: Dict[str, Any]) -> List[Dict]:
    """One record per phase token claimed in the answer: where the context
    actually carries it, how the sentence frames it, and whether the two
    disagree. Reported, never enforced -- the caller decides."""
    by_class = _phase_tokens_by_class(context)
    observed, model = by_class[PROVENANCE_OBSERVED], by_class[PROVENANCE_MODEL_DERIVED]
    if not observed and not model:
        return []
    framing = _framing_of(answer_text)
    out: List[Dict] = []
    for token in sorted(extract_phase_tokens(answer_text)):
        sources = []
        if any(token.lower() == t.lower() for t in observed):
            sources.append(PROVENANCE_OBSERVED)
        if any(token.lower() == t.lower() for t in model):
            sources.append(PROVENANCE_MODEL_DERIVED)
        if not sources:
            continue
        mismatch, why = False, None
        if sources == [PROVENANCE_MODEL_DERIVED] and PROVENANCE_OBSERVED in framing:
            mismatch, why = True, "model_value_framed_as_observation"
        elif sources == [PROVENANCE_MODEL_DERIVED] and not framing:
            mismatch, why = True, "model_value_asserted_without_attribution"
        elif sources == [PROVENANCE_OBSERVED] and PROVENANCE_MODEL_DERIVED in framing:
            mismatch, why = True, "observed_value_framed_as_prediction"
        out.append({"phase": token, "sources": sources,
                    "framing": sorted(framing), "mismatch": mismatch,
                    "reason": why})
    return out


@dataclass
class GroundingResult:
    checked_numbers: List[str] = field(default_factory=list)
    ungrounded_numbers: List[str] = field(default_factory=list)
    checked_phase_tokens: List[str] = field(default_factory=list)
    ungrounded_phase_tokens: List[str] = field(default_factory=list)
    conflicting_phase_tokens: List[str] = field(default_factory=list)
    checked_transition_claims: List[Dict] = field(default_factory=list)
    ungrounded_transition_claims: List[Dict] = field(default_factory=list)
    skip_claim_mismatches: List[Dict] = field(default_factory=list)
    # Per-claim provenance, added 2026-09-07d. Deliberately a SEPARATE list
    # rather than an extra key inside the three lists above: those shapes are
    # part of this module's existing contract and are asserted verbatim by
    # tests written before this fix. `matches` is one of "observed" (the pair
    # is an annotated transition), "model" (the model's own predicted change),
    # "both", or "none" (ungrounded). This is what keeps the MODEL-DERIVED /
    # OBSERVED distinction visible instead of collapsing it.
    transition_claim_provenance: List[Dict] = field(default_factory=list)
    # v3, additive: per-phase provenance. NOT folded into `grounded` -- see the
    # module-level note above `check_phase_provenance()`.
    phase_provenance: List[Dict] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return (
            not self.ungrounded_numbers and not self.ungrounded_phase_tokens
            and not self.ungrounded_transition_claims and not self.skip_claim_mismatches
        )

    @property
    def provenance_violations(self) -> List[Dict]:
        return [p for p in self.phase_provenance if p.get("mismatch")]

    @property
    def provenance_consistent(self) -> bool:
        """Deliberately SEPARATE from `grounded`: an answer can be perfectly
        grounded and still attribute a model output to the annotation."""
        return not self.provenance_violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked_numbers": self.checked_numbers, "ungrounded_numbers": self.ungrounded_numbers,
            "checked_phase_tokens": self.checked_phase_tokens,
            "ungrounded_phase_tokens": self.ungrounded_phase_tokens,
            "conflicting_phase_tokens": self.conflicting_phase_tokens,
            "checked_transition_claims": self.checked_transition_claims,
            "ungrounded_transition_claims": self.ungrounded_transition_claims,
            "skip_claim_mismatches": self.skip_claim_mismatches,
            "transition_claim_provenance": self.transition_claim_provenance,
            "phase_provenance": self.phase_provenance,
            "provenance_violations": self.provenance_violations,
            "provenance_consistent": self.provenance_consistent,
            "grounded": self.grounded,
        }


def check_grounding(answer_text: str, context: Dict[str, Any]) -> GroundingResult:
    """The post-hoc check `docs/LLM_ORCHESTRATION.md` sec 3 point 2
    describes: extract every number/phase-name from `answer_text` and
    verify it against the ACTUAL tool outputs from this turn
    (`context["dynamic_context"]`) or the retrieved document text
    (`context["document_context"]`) -- never against outside knowledge.
    A number/phase found in neither is reported as ungrounded; the
    caller decides what to do about it (never this function -- it has no
    side effects and never edits `answer_text`)."""
    known_numbers_canonical = _known_numbers_canonical(context.get("dynamic_context", {}))
    doc_text = _reference_text(context)
    known_doc_numbers_canonical = _known_numbers_from_document_text_canonical(doc_text)

    checked_numbers = extract_numbers(_strip_phase_tokens(answer_text))
    ungrounded_numbers = [
        n for n in checked_numbers
        if n not in _IGNORED_NUMBERS
        and _canonical_number_form(n) not in known_numbers_canonical
        and n not in doc_text
        and _canonical_number_form(n) not in known_doc_numbers_canonical
    ]

    checked_phases = extract_phase_tokens(answer_text)
    known_phases = _known_phase_tokens(context)
    ungrounded_phases = [p for p in checked_phases if p not in known_phases]

    # Live-conflict check (docs/RAG_LLM_QUALITY_REPORT.md P0 item 1): a
    # phase name merely enumerated somewhere in retrieved document text
    # (e.g. a vocabulary table) is not the same as that phase being this
    # turn's live claim -- `known_phases` above conflates the two, which
    # is exactly how a real false-negative slipped through (C2: "t4"
    # stated, real value "t7", both textually present in `document_context`,
    # `grounded=True`, zero warnings). When this turn actually fetched live
    # phase data, require the answer's phase-token claims to have SOME
    # overlap with that live set -- a token that is grounded only via
    # `document_context` while being completely disjoint from the live
    # data is now caught here, even though the loose `known_phases` check
    # above would have accepted it. Deliberately an "any overlap" test,
    # not "every token must match the live set": an answer legitimately
    # citing an ADDITIONAL phase from documentation alongside the correct
    # live one (e.g. explaining what typically follows the current phase)
    # must not be penalized for that second, correctly-doc-sourced token.
    dynamic_phases = _dynamic_phase_tokens(context)
    conflicting_phases: List[str] = []
    if dynamic_phases and checked_phases and not (set(checked_phases) & dynamic_phases):
        conflicting_phases = [p for p in checked_phases if p not in ungrounded_phases]
        ungrounded_phases = ungrounded_phases + conflicting_phases

    # Relational grounding (docs/RELATIONAL_GROUNDING.md): verifies the
    # ORDERED PAIR a transition claim asserts, not just each token's
    # individual presence -- see extract_transition_claims()'s own
    # docstring for what this does and does not catch. Only runs when
    # get_transition_events was actually called this turn (real_transitions
    # below is [] otherwise) -- no relational opinion is ever formed when
    # there is nothing real to check a claim against.
    observed_transitions = _real_observed_transitions(context)
    model_transitions = _real_model_transitions(context)
    claimed_pairs = extract_transition_claims(answer_text)

    checked_transition_claims: List[Dict] = []
    ungrounded_transition_claims: List[Dict] = []
    skip_claim_mismatches: List[Dict] = []
    transition_claim_provenance: List[Dict] = []
    # Fires as soon as EITHER reference set exists this turn. Unchanged when
    # only get_transition_events was called (the historical case); the new
    # branch is a turn that carries series_analysis, where the model's own
    # phase changes are a real, checkable reference. When neither is present
    # there is still nothing to verify a relation against -- same graceful
    # no-op as before.
    if observed_transitions is not None or model_transitions is not None:
        observed_pairs = _pairs(observed_transitions)
        model_pairs = _pairs(model_transitions)
        # A skip claim is only accepted for a pair the DATA itself marks as a
        # skip: `is_skip` on the annotated side, membership of
        # `stability.phase_skips` on the model side. The LLM's own "saut"
        # wording is never taken at face value (docs/RELATIONAL_GROUNDING.md).
        skip_pairs = {(t.get("from_phase"), t.get("to_phase"))
                      for t in (observed_transitions or []) if t.get("is_skip")}
        skip_pairs |= _pairs(_model_skip_transitions(context))
        text_claims_skip = _text_claims_skip(answer_text)
        for from_phase, to_phase in claimed_pairs:
            pair = (from_phase, to_phase)
            claim = {"from_phase": from_phase, "to_phase": to_phase}
            in_observed, in_model = pair in observed_pairs, pair in model_pairs
            matches = ("both" if in_observed and in_model
                       else "observed" if in_observed
                       else "model" if in_model else "none")
            checked_transition_claims.append(claim)
            transition_claim_provenance.append({**claim, "matches": matches})
            if matches == "none":
                ungrounded_transition_claims.append(claim)
                continue
            if text_claims_skip and pair not in skip_pairs:
                skip_claim_mismatches.append({**claim, "real_is_skip": False})

    return GroundingResult(
        checked_numbers=checked_numbers, ungrounded_numbers=ungrounded_numbers,
        checked_phase_tokens=checked_phases, ungrounded_phase_tokens=ungrounded_phases,
        conflicting_phase_tokens=conflicting_phases,
        checked_transition_claims=checked_transition_claims,
        ungrounded_transition_claims=ungrounded_transition_claims,
        skip_claim_mismatches=skip_claim_mismatches,
        transition_claim_provenance=transition_claim_provenance,
        phase_provenance=check_phase_provenance(answer_text, context),
    )


NO_INFORMATION_MESSAGE = (
    "Je ne dispose pas de suffisamment d'informations pour répondre "
    "(aucune donnée documentaire ni donnée dynamique récupérée pour cette question)."
)


def has_any_context(context: Dict[str, Any]) -> bool:
    return bool(context.get("document_context")) or bool(context.get("dynamic_context"))


def estimate_confidence(context: Dict[str, Any], grounding: GroundingResult) -> str:
    """A qualitative label ONLY -- "high"/"medium"/"low"/"unknown" --
    NEVER a fabricated numeric probability. This project already has a
    hard-won lesson about presenting a number as a calibrated confidence
    when it isn't one (the Semi-HMM's own phase-posterior miscalibration,
    docs/REPORTING_API.md's Known Limitations) -- this function is
    designed not to repeat that mistake at the LLM layer."""
    if not has_any_context(context):
        return "unknown"
    if not grounding.grounded:
        return "low"
    if context.get("warnings"):
        return "medium"
    return "high"

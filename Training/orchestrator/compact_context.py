"""
COMPACT_CONTEXT_V1 -- deterministic de-narrativisation of the RAG block.

WHY THIS MODULE EXISTS
----------------------
Audit of the real prompts actually sent to the LLM (the artefacts under
`Results/evaluation/event_anomaly_rag_inventory/`, 22 distinct chunks served
across the frozen Q1-Q15 benchmark) found that a large part of the DOCUMENT
half of the context is PROJECT NARRATIVE, not scientific evidence:

  docs/RESEARCH_BLUEPRINT.md :: Part VII -- Publication Strategy
      paper title, abstract claim, main figure, key table, reviewer's
      main criticism  -- 2254 characters, zero fact usable by any of the
      15 questions.
  docs/HANDOFF.md :: 7. The Target Scientific Narrative
      explicitly labelled "aspirational -- NOT a claimed result", with a
      "one-sentence reduction" slogan -- 2185 characters.
  docs/BLUEPRINT_SUMMARY.md :: Publication Goal
      "2-yr horizon", "domain-venue tier, not flagship" -- 530 characters.

The hypothesis this module exists to TEST (not to assert) is that a shorter,
structured, factual context improves answer quality by removing distractors.

WHAT IT DOES -- AND ONLY THAT
-----------------------------
`compact_document_context()` takes the real `context_builder.build_context()`
`document_context` list and returns a copy in which each chunk's content has
been filtered LINE BY LINE against an explicit, frozen rule table. It is a
SUBTRACTIVE, deterministic, rule-based filter:

  * no LLM, no summarisation, no paraphrase, no re-generation;
  * every surviving line is a VERBATIM line of the original chunk (pinned by
    `Tests/orchestrator/test_compact_context.py`);
  * no chunk metadata is altered: `source`, `section`, `status`,
    `authoritative`, `score` and `tool` are passed through untouched, so every
    surviving sentence stays traceable to its source;
  * nothing is added to a chunk except the frozen, value-free marker lines in
    `MARKER_TEMPLATES` (a dropped-line count and its rule ids) -- the filter
    can therefore never inject an expected answer, a phase name, a number, or
    a biological fact that was not already in the corpus;
  * a chunk whose every line is narrative is not silently deleted: it is kept
    in the list, marked `compact_dropped=True` with its reason, and rendered
    as a one-line traceable stub. A retrieval that returns project narrative
    for a clinical question stays VISIBLE in the prompt, which is what makes
    the retrieval failure auditable instead of hidden.

TWO LEVELS, ONE OVERRIDE
------------------------
1. SECTION level (`SECTION_RULES`) -- a chunk whose `section` title names a
   programme/publication/planning section is dropped wholesale. Matching is on
   the section TITLE, never on the body, so the decision is readable from the
   citation itself.
2. LINE level (`LINE_RULES`) -- inside a kept chunk, a line that carries a
   publication/aspirational/slogan/planning marker is dropped.
3. KEEP override (`KEEP_RULES`) -- a line carrying a scientific limitation or a
   measured quantity is ALWAYS kept, even if a LINE rule also matches it.
   Precedence is keep > drop, deliberately: the failure mode this project
   cannot accept is losing a stated limitation, not keeping one extra line.

WHAT IT DOES NOT TOUCH
----------------------
The dynamic (Reporting API) half of the context is NOT filtered by this module
and is rendered byte-identically in both context formats: the experiment's
single variable is the DOCUMENT half. The dataset, the annotations, the
Semi-HMM, the embeddings, the Chroma index, the corpus files, the router, the
grounding checker and the 15 frozen questions are all untouched -- this module
is a pure function over an already-built context dict.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

CONTEXT_FORMAT_V2 = "v2"           # the historical renderer, unchanged
CONTEXT_FORMAT_COMPACT = "compact"  # COMPACT_CONTEXT_V1


def _norm(text: str) -> str:
    """Lowercase, accent-stripped, whitespace-collapsed -- the same
    normalisation discipline the Router uses for its own keyword tables, so a
    rule matches `Étape` and `etape` alike and never depends on the document's
    accentuation."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped.lower()).strip()


# ---------------------------------------------------------------------------
# RULE TABLES -- frozen, explicit, each with its own id and stated reason.
# A rule is (rule_id, markers, reason). Markers are matched against the
# NORMALISED text at a WORD START (`_compile_marker`): no regex written by
# hand, no NLP, no model. Anything not matched is kept.
# ---------------------------------------------------------------------------

# Level 1 -- matched against the chunk's SECTION TITLE only.
SECTION_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("S1", ("publication",),
     "section de strategie/objectif de publication (titre, abstract, figure, revue)"),
    ("S2", ("target scientific narrative", "narration scientifique"),
     "narration scientifique aspirationnelle, explicitement non un resultat"),
    ("S3", ("current objective", "objectif courant"),
     "objectif de projet (gestion), aucune donnee scientifique"),
    ("S4", ("next step", "prochaine etape de", "next action"),
     "etape suivante de projet (gestion), aucune donnee scientifique"),
    ("S5", ("plan d'implementation", "plan d implementation", "execution plan"),
     "planification d'implementation (gestion), aucune donnee scientifique"),
)

# Level 2 -- matched against each LINE of a kept chunk.
LINE_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("L1", ("publication", "*title*", "abstract claim", "reviewer's main criticism",
            "domain-venue", "flagship", "relegated to supplement", "under review pressure"),
     "ligne de strategie de publication"),
    ("L2", ("aspirational", "best-case", "hypothesized discovery", "hypothesized best-case",
            "if the project fully succeeds", "if the realistic plan",
            "2-year-horizon", "2-yr horizon", "long-term aspirational"),
     "enonce aspirationnel / meilleur cas, explicitement non etabli"),
    ("L3", ("one-sentence reduction", "one-sentence version", "one sentence version"),
     "slogan / reduction en une phrase"),
    ("L4", ("the one figure", "*main figure*", "*key table*", "*experimental story*",
            "(4 panels)"),
     "plan de figure/tableau d'un futur article"),
    ("L5", ("hors scope de ce plan", "decision required", "roadmap", "nvidia-smi",
            "screen -ls", "pas de lancement automatique"),
     "gestion de projet / discipline d'execution, sans contenu scientifique"),
    ("L6", ("core hypotheses", "**h1**", "**h2**", "**h3**", "**h4**",
            "gates everything"),
     "programme d'hypotheses du projet, non un fait etabli"),
)

# Level 3 -- OVERRIDE. A line matching any of these is kept even if a LINE rule
# also matches it. Keep > drop, always.
KEEP_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("K1", ("limit", "caveat", "not significant", "non significatif", "censor",
            "locked", "verrouille", "unverified", "non verifie", "bottleneck",
            "val-only", "untried", "never scored", "not computed", "gap"),
     "limitation scientifique declaree -- jamais supprimee"),
    ("K2", ("auroc", "brier", "ece", "pr-auc", "p=", "n=", "auc", "accuracy",
            "recall", "log loss", "dmax", "entropie", "entropy"),
     "quantite mesuree ou nommee -- jamais supprimee"),
)

# The ONLY text this module may add to a prompt. Value-free by construction:
# no phase name, no number from the data, no answer -- only a count and rule
# ids. Pinned by Tests/orchestrator/test_compact_context.py.
MARKER_TEMPLATES = {
    "dropped_lines": "(note: {n} ligne(s) narrative(s) ecartee(s) -- regles {rules})",
    "dropped_chunk": "(document ecarte du contexte: {reason} -- regle {rule}; "
                     "aucune ligne factuelle retenue)",
}


def _compile_marker(marker: str) -> "re.Pattern[str]":
    """A marker matches at a WORD START, never inside a word.

    Plain substring matching was tried first and rejected on a real case: the
    marker `ece` (the calibration metric) matched the `ece` inside
    "n-ece-ssary", so a hypothesis line survived a rule that was meant to drop
    it. A word-START boundary (not a full-word one) keeps the intended
    stemming -- `limit` still matches "limitation"/"limited", `gap` still
    matches "gaps" -- while ending the inside-a-word collisions. Markers that
    begin with a non-word character (`*title*`, `**h1**`, `(4 panels)`) get no
    boundary, since `\b` before `*` would never match."""
    prefix = r"\b" if marker[:1].isalnum() else ""
    return re.compile(prefix + re.escape(marker))


_COMPILED: Dict[str, "re.Pattern[str]"] = {}


def _matching_rule(text: str,
                   rules: Tuple[Tuple[str, Tuple[str, ...], str], ...]) -> Optional[Tuple[str, str]]:
    """First (rule_id, reason) whose marker occurs in `text`, else None. The
    table order is the precedence order, and it is frozen."""
    normalised = _norm(text)
    if not normalised:
        return None
    for rule_id, markers, reason in rules:
        for marker in markers:
            pattern = _COMPILED.get(marker)
            if pattern is None:
                pattern = _COMPILED[marker] = _compile_marker(marker)
            if pattern.search(normalised):
                return rule_id, reason
    return None


def classify_line(line: str) -> Dict[str, Any]:
    """Verdict for ONE line: {"keep": bool, "rule": id|None, "reason": str}.

    A blank line is kept (it carries the chunk's own paragraph structure and
    costs nothing). The KEEP override is evaluated FIRST, so a limitation
    stated inside a publication paragraph survives."""
    if not line.strip():
        return {"keep": True, "rule": None, "reason": "ligne vide (structure)"}
    keep = _matching_rule(line, KEEP_RULES)
    if keep is not None:
        return {"keep": True, "rule": keep[0], "reason": keep[1]}
    drop = _matching_rule(line, LINE_RULES)
    if drop is not None:
        return {"keep": False, "rule": drop[0], "reason": drop[1]}
    return {"keep": True, "rule": None, "reason": "aucune regle narrative ne matche"}


def classify_section(section: Optional[str]) -> Optional[Tuple[str, str]]:
    """(rule_id, reason) if this section TITLE is a project-narrative section,
    else None."""
    if not section:
        return None
    return _matching_rule(section, SECTION_RULES)


def compact_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """One chunk in, one chunk out. Metadata is passed through unchanged; only
    `content` is filtered, and the filtering is recorded on the returned dict:

        compact_dropped          bool   -- whole chunk classified as narrative
        compact_drop_rule        str    -- which SECTION rule fired
        compact_drop_reason      str
        compact_content          str    -- the kept lines, verbatim, in order
        compact_dropped_lines    int
        compact_rules_applied    [str]  -- LINE rule ids that fired, sorted
        compact_original_chars   int
        compact_kept_chars       int
    """
    out = dict(chunk)
    original = chunk.get("content") or ""
    out["compact_original_chars"] = len(original)

    section_rule = classify_section(chunk.get("section"))
    if section_rule is not None:
        rule_id, reason = section_rule
        out.update({
            "compact_dropped": True, "compact_drop_rule": rule_id,
            "compact_drop_reason": reason, "compact_content": "",
            "compact_dropped_lines": len(original.splitlines()),
            "compact_rules_applied": [rule_id], "compact_kept_chars": 0,
        })
        return out

    kept: List[str] = []
    rules_fired: List[str] = []
    dropped = 0
    for line in original.splitlines():
        verdict = classify_line(line)
        if verdict["keep"]:
            kept.append(line)
        else:
            dropped += 1
            if verdict["rule"] and verdict["rule"] not in rules_fired:
                rules_fired.append(verdict["rule"])

    # Leading/trailing blank lines left behind by a removal carry no
    # information; interior blank lines are kept (paragraph structure).
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    content = "\n".join(kept)

    if not content.strip():
        out.update({
            "compact_dropped": True, "compact_drop_rule": rules_fired[0] if rules_fired else "L0",
            "compact_drop_reason": "toutes les lignes du chunk sont narratives",
            "compact_content": "", "compact_dropped_lines": dropped,
            "compact_rules_applied": sorted(rules_fired), "compact_kept_chars": 0,
        })
        return out

    out.update({
        "compact_dropped": False, "compact_drop_rule": None, "compact_drop_reason": None,
        "compact_content": content, "compact_dropped_lines": dropped,
        "compact_rules_applied": sorted(rules_fired), "compact_kept_chars": len(content),
    })
    return out


def compact_document_context(document_context: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The whole RAG block, chunk by chunk, order preserved. Never reorders,
    never re-ranks, never adds a chunk: retrieval is NOT modified by this
    module (the same chunks, from the same query, in the same order)."""
    return [compact_chunk(chunk) for chunk in (document_context or [])]


def compaction_stats(compacted: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measurement of what the filter did -- for the experiment artefact, not
    for the prompt."""
    original_chars = sum(c.get("compact_original_chars", 0) for c in compacted)
    kept_chars = sum(c.get("compact_kept_chars", 0) for c in compacted)
    rules: List[str] = []
    for chunk in compacted:
        for rule in chunk.get("compact_rules_applied", []) or []:
            if rule not in rules:
                rules.append(rule)
    return {
        "n_chunks": len(compacted),
        "n_chunks_dropped": sum(1 for c in compacted if c.get("compact_dropped")),
        "n_lines_dropped": sum(c.get("compact_dropped_lines", 0) for c in compacted),
        "original_content_chars": original_chars,
        "kept_content_chars": kept_chars,
        "removed_content_chars": original_chars - kept_chars,
        "removed_fraction": round((original_chars - kept_chars) / original_chars, 4)
        if original_chars else 0.0,
        "rules_applied": sorted(rules),
    }


# ===========================================================================
# COMPACT_CONTEXT_V2 (2026-09-12) -- chunk SELECTION on top of line filtering.
#
# compact_v1 filtered LINES inside the five chunks the retriever returned and
# printed a stub for a fully-narrative chunk. The graded compact_v1 artefact
# showed the limit of that: for Q14, 2 of the 5 chunks were HANDOFF section 7
# (dropped as stubs) and a third was "Open Research Questions" -- the LLM was
# left with almost nothing useful. compact_v2 therefore SELECTS chunks from a
# larger candidate pool BEFORE rendering: a chunk from a document that carries
# no fact about the embryo or the model (engineering, product, planning) or
# from a programme section is skipped, and the next candidate takes its place.
# Selection is explicit and auditable (a frozen table of document paths and
# section titles), never a re-ranking, never a similarity threshold, never a
# model. The corpus and the index are untouched; only the choice among what
# the index already returned changes, and every decision is recorded.
# ===========================================================================

SECTION_RULES_V2: Tuple[Tuple[str, Tuple[str, ...], str], ...] = SECTION_RULES + (
    ("S6", ("open research questions", "questions de recherche ouvertes", "questions ouvertes"),
     "questions de recherche ouvertes (programme), aucune donnee scientifique"),
    ("S7", ("ce qui reste experimental", "what remains experimental", "cross-cutting notes",
            "roadmap", "feuille de route"),
     "note de produit / feuille de route, aucune donnee scientifique"),
)

# Documents that carry NO fact about the embryo, the annotation or the model --
# engineering, product, planning and research-programme prose. Matched on the
# chunk's `source` path, exactly as the inventory names it.
SOURCE_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("D1", ("docs/RESEARCH_BLUEPRINT.md", "docs/BLUEPRINT_SUMMARY.md"),
     "programme d'hypotheses / strategie de recherche : aucune mesure sur les donnees"),
    ("D2", ("docs/PRODUCT_ROADMAP.md", "docs/PRODUCT_ARCHITECTURE.md", "docs/PROJECT_TO_PRODUCT.md",
            "docs/WEBAPP_ARCHITECTURE.md", "docs/WEBAPP_DATA_REQUIREMENTS.md",
            "docs/LLM_ORCHESTRATION.md", "docs/RAG_ARCHITECTURE.md", "docs/RAG_DATA_MODEL.md",
            "docs/REPORTING_API.md", "docs/REPORTING_API_IMPLEMENTATION_REPORT.md",
            "docs/REPORTING_CACHE.md", "docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md",
            "docs/REPRODUCIBILITY.md", "docs/DATA_LINEAGE.md", "docs/HMM_PHASE0_EXECUTION_PLAN.md"),
     "documentation d'ingenierie / produit / planification : aucun fait sur l'embryon ni sur le modele"),
)

DEFAULT_RETRIEVAL_CANDIDATE_POOL = 15   # candidates asked from the index
DEFAULT_SELECTED_CHUNKS = 5             # chunks finally rendered (unchanged from v1/v2)


def _chunk_source(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return str(meta.get("source") or chunk.get("source") or "")


def _chunk_section(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return str(meta.get("section") or chunk.get("section") or "")


def classify_source(source: Optional[str]) -> Optional[Tuple[str, str]]:
    """(rule_id, reason) if this document path is a narrative/engineering
    document, else None. Exact path match, never a substring."""
    if not source:
        return None
    for rule_id, paths, reason in SOURCE_RULES:
        if source in paths:
            return rule_id, reason
    return None


def select_relevant_chunks(candidates: List[Dict[str, Any]],
                           top_k: int = DEFAULT_SELECTED_CHUNKS,
                           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep the first `top_k` candidates (in the retriever's own rank order)
    that are neither from a narrative document (SOURCE_RULES) nor from a
    programme section (SECTION_RULES_V2). Returns (kept, audit): `kept` is a
    list of the ORIGINAL chunk dicts, untouched; `audit` records one decision
    per candidate so a skipped chunk is never silently gone."""
    kept: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for rank, chunk in enumerate(candidates or [], start=1):
        source, section = _chunk_source(chunk), _chunk_section(chunk)
        rule = classify_source(source)
        if rule is None:
            section_rule = _matching_rule(section, SECTION_RULES_V2) if section else None
            rule = section_rule
        entry = {"rank": rank, "source": source, "section": section,
                 "score": chunk.get("score")}
        if rule is not None:
            entry.update({"decision": "skipped", "rule": rule[0], "reason": rule[1]})
        elif len(kept) >= top_k:
            entry.update({"decision": "not_needed", "rule": None,
                          "reason": f"les {top_k} chunks retenus etaient deja atteints"})
        else:
            entry.update({"decision": "kept", "rule": None, "reason": None})
            kept.append(chunk)
        audit.append(entry)
    return kept, audit


def compact_chunk_v2(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """compact_chunk() with the v2 section table (S1-S7). Line rules and the
    keep override are the same frozen tables as v1."""
    section_rule = classify_section_v2(chunk.get("section"))
    if section_rule is None:
        return compact_chunk(chunk)
    out = dict(chunk)
    original = chunk.get("content") or ""
    rule_id, reason = section_rule
    out.update({
        "compact_original_chars": len(original),
        "compact_dropped": True, "compact_drop_rule": rule_id,
        "compact_drop_reason": reason, "compact_content": "",
        "compact_dropped_lines": len(original.splitlines()),
        "compact_rules_applied": [rule_id], "compact_kept_chars": 0,
    })
    return out


def classify_section_v2(section: Optional[str]) -> Optional[Tuple[str, str]]:
    if not section:
        return None
    return _matching_rule(section, SECTION_RULES_V2)


def compact_document_context_v2(document_context: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [compact_chunk_v2(chunk) for chunk in (document_context or [])]

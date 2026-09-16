"""
SCIENTIFIC REFERENCE for the LLM context -- Istanbul Consensus 2025 blocks.

WHY
---
Until compact_v2 the only "documentary" knowledge the LLM ever saw was the
project's own `docs/*.md` corpus (HANDOFF, RESEARCH_BLUEPRINT, reports, plans).
The Istanbul Consensus corpus (`docs/corpus/ISTANBUL_CONSENSUS_2025.md`, 138
identified blocks with type / page / table / evidence grade) was read by the
Scientific Validator only. So a question such as "est-ce compatible avec les
reperes morphocinetiques ?" could only be answered from project prose, and the
graded runs show the LLM citing HANDOFF section 7 (an aspirational narrative)
as its "reference".

This module puts the Consensus in the context as a SEPARATE class,
[SCIENTIFIC], never mixed with project documentation, never mixed with the
embryo's data, and always with the limitation blocks that must travel with
the evidence (IC2025-T1's own "qualifications that must travel with any use of
this table", the absent-definition block for abnormal cleavage, the GPR
disclaimer for anything clinical).

WHAT IT IS NOT
--------------
No embedding, no Chroma, no LLM summary: the retrieval is the validator's own
transparent lexical search (`validator.retrieval.search`), the text is the
corpus text verbatim, and a field the corpus does not print stays absent. The
corpus file itself is never modified here. If the corpus is missing, the block
is reported unavailable -- nothing is reconstructed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from validator.corpus import SOURCE_NAME, Corpus, CorpusBlock
from validator.extraction import markers_in
from validator.retrieval import search

DEFAULT_TOP_K = 3

# Topic markers (validator.extraction.MARKERS keys) for which the Consensus can
# carry evidence at all. A question with none of them (e.g. "quelle est la
# deuxieme phase la plus probable ?") gets NO scientific block: the Consensus
# says nothing about a Semi-HMM posterior, and printing an unrelated block
# would be exactly the lexical-proximity distraction compact_v2 exists to end.
SCIENTIFIC_MARKERS: Sequence[str] = (
    "expected_development", "compatibility", "timing", "direct_cleavage",
    "reverse_cleavage", "chaotic_division", "biological", "clinical", "skip",
    "regression",
)
# Only these markers TRIGGER a scientific block on their own. `timing`, `skip`
# and `regression` are support topics: they add their limitation blocks when a
# trigger is present, but alone they describe a model/annotation quantity the
# Consensus says nothing about. Measured on the real pipeline (2026-09-12
# probe): Q7 "avec quel decalage ?" (timing only) would otherwise receive
# Table 1's reference timings -- lexically close, scientifically unrelated to a
# model-vs-annotation lag, i.e. the exact distraction this block must not add.
TRIGGER_MARKERS: Sequence[str] = (
    "expected_development", "compatibility", "direct_cleavage", "reverse_cleavage",
    "chaotic_division", "biological", "clinical",
)

# Limitation / scope blocks attached to the evidence, per topic. These are the
# SAME block ids the validator rules cite (`validator/rules.py`), so the LLM
# reads, before answering, the exact limitation the validator will hold its
# answer to afterwards. Every id is verified against the corpus by
# Tests/orchestrator/test_scientific_reference.py.
LIMITATIONS_BY_MARKER: Dict[str, Sequence[str]] = {
    "timing": ("IC2025-CP1-01", "IC2025-LIM-09", "IC2025-CP1-02", "IC2025-TIME-01"),
    "expected_development": ("IC2025-CP1-02", "IC2025-CP1-04", "IC2025-LIM-01"),
    "compatibility": ("IC2025-LIM-01", "IC2025-CP1-02", "IC2025-LIM-09"),
    "skip": ("IC2025-AC-B01", "IC2025-ABSENT-01"),
    "regression": ("IC2025-AC-B01", "IC2025-ABSENT-01"),
    "direct_cleavage": ("IC2025-AC-B01", "IC2025-ABSENT-01", "IC2025-AC-E01", "IC2025-AC-E02"),
    "reverse_cleavage": ("IC2025-AC-B01", "IC2025-ABSENT-01", "IC2025-AC-E01", "IC2025-AC-E02"),
    "chaotic_division": ("IC2025-AC-B01", "IC2025-ABSENT-01", "IC2025-AC-E01", "IC2025-AC-E02"),
    "clinical": ("IC2025-LIM-01", "IC2025-BLA-03", "IC2025-LIM-16"),
    "biological": ("IC2025-LIM-01", "IC2025-LIM-06"),
}

_corpus_singleton: Optional[Corpus] = None


def default_corpus() -> Corpus:
    """Parsed once per process; the file is read-only."""
    global _corpus_singleton
    if _corpus_singleton is None:
        _corpus_singleton = Corpus()
    return _corpus_singleton


def _block_dict(block: CorpusBlock, score: Optional[float], role: str) -> Dict[str, Any]:
    return {
        "source": SOURCE_NAME,
        "block_id": block.block_id,
        "subject": block.subject,
        "evidence_type": block.evidence_type,
        "evidence_grade": block.evidence_grade,
        "page": block.page,
        "table": block.table,
        "section": block.section,
        "text": block.text,
        "score": score,
        "role": role,                      # "evidence" | "limitation"
    }


def retrieve_scientific_reference(question: str, top_k: int = DEFAULT_TOP_K,
                                  corpus: Optional[Corpus] = None) -> Dict[str, Any]:
    """The [SCIENTIFIC] block for one question, as a plain dict:

        {source, available, markers, scientific_markers, blocks[], limitations[],
         warnings[]}

    `blocks` are the top-k evidence blocks for the question's scientific
    topics; `limitations` the scope/limitation blocks attached to those topics
    (deduplicated against `blocks`). Both lists are empty, with the reason in
    `warnings`, when the corpus is absent or the question carries no
    scientific topic."""
    corpus = corpus if corpus is not None else default_corpus()
    markers = markers_in(question or "")
    topical = [m for m in markers if m in SCIENTIFIC_MARKERS]
    out: Dict[str, Any] = {
        "source": SOURCE_NAME, "available": corpus.available, "markers": markers,
        "scientific_markers": topical, "blocks": [], "limitations": [], "warnings": [],
    }
    if not corpus.available:
        out["warnings"].append(
            "Corpus Istanbul indisponible sur cette machine : aucune reference scientifique "
            "n'est fournie (rien n'est reconstruit).")
        return out
    if not any(m in TRIGGER_MARKERS for m in topical):
        out["warnings"].append(
            "La question ne porte sur aucun sujet couvert par le Consensus : aucune reference "
            "scientifique n'est fournie.")
        return out

    hits = search(corpus, question, topical, top_k=top_k)
    seen = set()
    for block, score in hits:
        out["blocks"].append(_block_dict(block, score, "evidence"))
        seen.add(block.block_id)
    for marker in topical:
        for block_id in LIMITATIONS_BY_MARKER.get(marker, ()):
            if block_id in seen:
                continue
            block = corpus.get(block_id)
            if block is None:
                out["warnings"].append(f"bloc de limitation {block_id} absent du corpus")
                continue
            out["limitations"].append(_block_dict(block, None, "limitation"))
            seen.add(block_id)
    if not out["blocks"]:
        out["warnings"].append(
            "Aucun bloc du Consensus ne recoupe lexicalement la question : seules les "
            "limitations de portee sont fournies.")
    return out


def scientific_context_text(scientific_context: Optional[Dict[str, Any]]) -> str:
    """Every string the block carries, for grounding (presence) checks."""
    if not scientific_context:
        return ""
    parts: List[str] = []
    for entry in list(scientific_context.get("blocks") or []) + list(
            scientific_context.get("limitations") or []):
        for key in ("block_id", "subject", "text", "page", "table", "section",
                    "evidence_type", "evidence_grade"):
            value = entry.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts)

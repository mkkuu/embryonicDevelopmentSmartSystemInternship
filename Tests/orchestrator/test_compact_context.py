"""Pure tests for compact_context.py -- no LLM, no torch, no chromadb, no GPU.

These pin the SAFETY properties of COMPACT_CONTEXT_V1: the filter may only
subtract, it may never rewrite, never invent, never lose a declared scientific
limitation, and never touch a chunk's provenance metadata.
"""

import pytest

from orchestrator import compact_context as C

# Real chunks, copied verbatim from the artefacts of a real run
# (Results/evaluation/event_anomaly_rag_inventory/fixed_question_benchmark_
# mistral-nemo_12b_prompt_v2_reading_method.json). Not invented fixtures: the
# rules are tested against the text they were designed on.
NARRATIVE_CHUNK = {
    "tool": "retrieve_documents",
    "source_type": "document",
    "source": "docs/RESEARCH_BLUEPRINT.md",
    "section": "Part VII — Publication Strategy",
    "status": "authoritative",
    "authoritative": True,
    "score": 0.5565307140350342,
    "content": (
        "**If the project fully succeeds** (RQ1, RQ2, and RQ4 all land — a "
        "2-year-horizon scenario, not the near-term deliverable):\n\n"
        "- *Title*: \"A single hidden coordinate governs human embryo development.\"\n"
        "- *Main figure* (4 panels): (a) trajectories collapsing onto one shared curve.\n"
    ),
}

SCIENTIFIC_CHUNK = {
    "tool": "retrieve_documents",
    "source_type": "document",
    "source": "docs/MODEL_COMPARISON.md",
    "section": "Semi-HMM",
    "status": "current",
    "authoritative": False,
    "score": 0.5793,
    "content": (
        "- **Architecture**: composes the identical HMM emission/transition logic + "
        "per-phase negative-binomial duration model, dmax=268 (zero truncation).\n"
        "- **Limit**: the gain is **not significant at trajectory level** "
        "(p=0.29, 58/106 videos favor it — near coin-flip).\n"
        "- **Result**: AUROC 0.64271, Brier 0.14029, ECE 0.12307.\n"
    ),
}

MIXED_CHUNK = {
    "tool": "retrieve_documents",
    "source_type": "document",
    "source": "docs/RESEARCH_BLUEPRINT.md",
    "section": "Part I — Scientific Foundation",
    "status": "authoritative",
    "authoritative": True,
    "score": 0.6173790693283081,
    "content": (
        "**Motivation.** Morphokinetic timing (not just sequence) is already known in "
        "the clinical literature to carry prognostic signal.\n\n"
        "**Core hypotheses** (pruned to what is actually testable):\n\n"
        "- **H1**: explicit dynamics modeling improves prediction beyond "
        "window-independent classification.\n"
        "- **H4** (aspirational, deferred — not in the near-term budget): the natural "
        "coordinate system is more fundamental than the hand-designed state.\n"
    ),
}


# ---------------------------------------------------------------------------
# 6. the RAG stays traceable / 7. no expected answer is injected
# ---------------------------------------------------------------------------

def test_metadata_is_passed_through_untouched():
    """Every field a citation is built from survives byte-identically -- a
    compacted chunk must stay attributable to its source."""
    out = C.compact_chunk(MIXED_CHUNK)
    for key in ("tool", "source_type", "source", "section", "status", "authoritative", "score"):
        assert out[key] == MIXED_CHUNK[key]


def test_every_kept_line_is_verbatim_from_the_original_chunk():
    """The filter subtracts; it never rewrites, reorders or paraphrases."""
    for chunk in (SCIENTIFIC_CHUNK, MIXED_CHUNK, NARRATIVE_CHUNK):
        out = C.compact_chunk(chunk)
        original_lines = chunk["content"].splitlines()
        kept = out["compact_content"].splitlines()
        for line in kept:
            assert line in original_lines, f"{line!r} is not a line of the source chunk"
        # order preserved
        indices = [original_lines.index(line) for line in kept if line.strip()]
        assert indices == sorted(indices)


def test_filter_adds_no_text_other_than_the_frozen_value_free_markers():
    """The only strings this layer may add are the two templates -- so it can
    never inject a value, a phase name, or an expected answer."""
    assert set(C.MARKER_TEMPLATES) == {"dropped_lines", "dropped_chunk"}
    for template in C.MARKER_TEMPLATES.values():
        # a template carries only a count and rule ids, never data
        assert "{n}" in template or "{reason}" in template
        assert not any(token in template for token in ("t7", "t8", "0.", "window", "phase"))


def test_a_dropped_chunk_is_reported_not_silently_deleted():
    out = C.compact_chunk(NARRATIVE_CHUNK)
    assert out["compact_dropped"] is True
    assert out["compact_drop_rule"] == "S1"
    assert out["compact_drop_reason"]
    # still present in the list, still carrying its source
    compacted = C.compact_document_context([NARRATIVE_CHUNK, SCIENTIFIC_CHUNK])
    assert len(compacted) == 2
    assert compacted[0]["source"] == NARRATIVE_CHUNK["source"]


# ---------------------------------------------------------------------------
# scientific-content preservation
# ---------------------------------------------------------------------------

def test_a_scientific_chunk_is_kept_whole():
    out = C.compact_chunk(SCIENTIFIC_CHUNK)
    assert out["compact_dropped"] is False
    assert out["compact_dropped_lines"] == 0
    assert out["compact_content"].strip() == SCIENTIFIC_CHUNK["content"].strip()


def test_a_declared_limitation_is_never_removed():
    """KEEP > DROP. The failure this project cannot accept is losing a stated
    limitation."""
    line = ("- *Publication limit*: the gain is not significant at trajectory level "
            "(p=0.29).")
    verdict = C.classify_line(line)
    assert verdict["keep"] is True and verdict["rule"] == "K1"


def test_a_measured_quantity_is_never_removed():
    line = "- *Abstract claim*: AUROC 0.64271, Brier 0.14029, ECE 0.12307."
    verdict = C.classify_line(line)
    assert verdict["keep"] is True and verdict["rule"] == "K2"


def test_markers_match_at_a_word_start_only():
    """Regression: plain substring matching made the metric marker `ece` fire
    inside "necessary", keeping a hypothesis line it was meant to drop."""
    line = "- **H2**: a discrete regime variable is necessary for this programme."
    verdict = C.classify_line(line)
    assert verdict["keep"] is False and verdict["rule"] == "L6"


def test_mixed_chunk_keeps_the_science_and_drops_the_programme():
    out = C.compact_chunk(MIXED_CHUNK)
    assert out["compact_dropped"] is False
    assert "Morphokinetic timing" in out["compact_content"]
    assert "**H1**" not in out["compact_content"]
    assert "aspirational" not in out["compact_content"]
    assert out["compact_dropped_lines"] >= 3
    assert "L6" in out["compact_rules_applied"]


def test_empty_document_context_is_handled():
    assert C.compact_document_context(None) == []
    assert C.compact_document_context([]) == []
    stats = C.compaction_stats([])
    assert stats["n_chunks"] == 0 and stats["removed_fraction"] == 0.0


def test_filter_is_deterministic_and_pure():
    first = C.compact_document_context([MIXED_CHUNK, NARRATIVE_CHUNK, SCIENTIFIC_CHUNK])
    second = C.compact_document_context([MIXED_CHUNK, NARRATIVE_CHUNK, SCIENTIFIC_CHUNK])
    assert [c["compact_content"] for c in first] == [c["compact_content"] for c in second]
    # inputs untouched
    assert "**H1**" in MIXED_CHUNK["content"]
    assert "compact_content" not in MIXED_CHUNK


def test_retrieval_order_and_count_are_not_changed():
    """This layer filters content; it never re-ranks, filters out, or adds a
    chunk -- retrieval itself is untouched by the experiment."""
    chunks = [NARRATIVE_CHUNK, SCIENTIFIC_CHUNK, MIXED_CHUNK]
    compacted = C.compact_document_context(chunks)
    assert [c["source"] for c in compacted] == [c["source"] for c in chunks]
    assert [c["score"] for c in compacted] == [c["score"] for c in chunks]


def test_stats_are_consistent_with_the_chunks():
    compacted = C.compact_document_context([NARRATIVE_CHUNK, SCIENTIFIC_CHUNK, MIXED_CHUNK])
    stats = C.compaction_stats(compacted)
    assert stats["n_chunks"] == 3
    assert stats["n_chunks_dropped"] == 1
    assert stats["kept_content_chars"] < stats["original_content_chars"]
    assert 0.0 < stats["removed_fraction"] < 1.0


@pytest.mark.parametrize("section,expected_rule", [
    ("Part VII — Publication Strategy", "S1"),
    ("Publication Goal", "S1"),
    ("7. The Target Scientific Narrative (aspirational — NOT a claimed result)", "S2"),
    ("2. Current Objective", "S3"),
    ("17. Next step", "S4"),
    ("12. Étape 12 — Plan d'implémentation", "S5"),
])
def test_narrative_sections_are_recognised_by_their_title(section, expected_rule):
    assert C.classify_section(section)[0] == expected_rule


@pytest.mark.parametrize("section", [
    "Semi-HMM", "2. Dataset", "5. Transition comparison", "Cross-cutting notes",
    "Part II — Mathematical Formulation", "9. Open Research Questions", None, "",
])
def test_scientific_sections_are_not_dropped_by_title(section):
    assert C.classify_section(section) is None

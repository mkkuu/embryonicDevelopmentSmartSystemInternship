"""compact_v2 chunk SELECTION -- explicit, auditable, order-preserving."""

from orchestrator import compact_context as cc


def _chunk(source, section, content="fact line\n", score=0.5):
    return {"content": content, "score": score,
            "metadata": {"source": source, "section": section, "status": "current",
                         "authoritative": False}}


CANDIDATES = [
    _chunk("docs/HANDOFF.md", "7. The Target Scientific Narrative (aspirational)", score=0.9),
    _chunk("docs/RESEARCH_BLUEPRINT.md", "Part I — Scientific Foundation", score=0.8),
    _chunk("docs/SCIENTIFIC_REPORT.md", "2. Dataset", score=0.7),
    _chunk("docs/HANDOFF.md", "9. Open Research Questions", score=0.6),
    _chunk("docs/WEBAPP_DATA_REQUIREMENTS.md", "Cross-cutting notes", score=0.5),
    _chunk("docs/GLOBAL_MODEL_COMPARISON.md", "5. Transition comparison", score=0.4),
    _chunk("docs/HMM_RESEARCH_PLAN.md", "7. Étape 7 — Reporting probabiliste", score=0.3),
    _chunk("docs/MODEL_COMPARISON.md", "Semi-HMM", score=0.2),
]


def test_narrative_documents_and_sections_are_skipped_and_order_is_kept():
    kept, audit = cc.select_relevant_chunks(CANDIDATES, top_k=3)
    assert [c["metadata"]["source"] for c in kept] == [
        "docs/SCIENTIFIC_REPORT.md", "docs/GLOBAL_MODEL_COMPARISON.md", "docs/HMM_RESEARCH_PLAN.md"]
    decisions = [(a["rank"], a["decision"], a["rule"]) for a in audit]
    assert decisions == [(1, "skipped", "S2"), (2, "skipped", "D1"), (3, "kept", None),
                         (4, "skipped", "S6"), (5, "skipped", "D2"), (6, "kept", None),
                         (7, "kept", None), (8, "not_needed", None)]


def test_kept_chunks_are_the_original_objects_untouched():
    kept, _ = cc.select_relevant_chunks(CANDIDATES, top_k=5)
    assert all(any(c is k for c in CANDIDATES) for k in kept)
    assert kept[0] == _chunk("docs/SCIENTIFIC_REPORT.md", "2. Dataset", score=0.7)


def test_every_candidate_has_an_audit_entry_even_when_none_is_kept():
    only_narrative = CANDIDATES[:2]
    kept, audit = cc.select_relevant_chunks(only_narrative, top_k=5)
    assert kept == []
    assert len(audit) == 2 and all(a["decision"] == "skipped" for a in audit)


def test_source_rule_is_an_exact_path_match():
    assert cc.classify_source("docs/RESEARCH_BLUEPRINT.md")[0] == "D1"
    assert cc.classify_source("docs/RESEARCH_BLUEPRINT_NOTES.md") is None
    assert cc.classify_source("RESEARCH_BLUEPRINT.md") is None
    assert cc.classify_source(None) is None


def test_factual_documents_are_never_in_the_source_rules():
    listed = {p for _, paths, _ in cc.SOURCE_RULES for p in paths}
    for factual in ("docs/SCIENTIFIC_REPORT.md", "docs/SCIENTIFIC_RESULTS.md",
                    "docs/GLOBAL_MODEL_COMPARISON.md", "docs/MODEL_COMPARISON.md",
                    "docs/HMM_RESEARCH_PLAN.md", "docs/HANDOFF.md", "docs/SEMI_HMM_PHASE_1_6_REPORT.md",
                    "docs/GRU_IDENTITY_ANALYSIS.md", "docs/INFERENCE_SCHEMA.md"):
        assert factual not in listed


def test_v2_section_table_extends_v1_without_changing_it():
    assert cc.SECTION_RULES_V2[:len(cc.SECTION_RULES)] == cc.SECTION_RULES
    assert {r[0] for r in cc.SECTION_RULES_V2} == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    assert cc.classify_section("9. Open Research Questions") is None          # v1 keeps it
    assert cc.classify_section_v2("9. Open Research Questions")[0] == "S6"    # v2 drops it


def test_compact_chunk_v2_stubs_a_v2_section_and_delegates_the_rest_to_v1():
    stub = cc.compact_chunk_v2({"source": "docs/HANDOFF.md", "section": "9. Open Research Questions",
                                "content": "a\nb\n"})
    assert stub["compact_dropped"] is True and stub["compact_drop_rule"] == "S6"
    assert stub["compact_dropped_lines"] == 2 and stub["compact_kept_chars"] == 0
    kept = cc.compact_chunk_v2({"source": "docs/SCIENTIFIC_REPORT.md", "section": "2. Dataset",
                                "content": "AUROC 0.9\n"})
    assert kept == cc.compact_chunk({"source": "docs/SCIENTIFIC_REPORT.md", "section": "2. Dataset",
                                     "content": "AUROC 0.9\n"})


def test_selection_accepts_build_context_shaped_chunks_too():
    chunks = [{"source": "docs/BLUEPRINT_SUMMARY.md", "section": "Publication Goal", "content": "x"},
              {"source": "docs/SCIENTIFIC_REPORT.md", "section": "2. Dataset", "content": "y"}]
    kept, audit = cc.select_relevant_chunks(chunks, top_k=5)
    assert [c["source"] for c in kept] == ["docs/SCIENTIFIC_REPORT.md"]
    assert audit[0]["rule"] == "D1"

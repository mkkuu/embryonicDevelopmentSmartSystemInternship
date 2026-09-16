"""Unit tests for rag.schemas -- pure stdlib, no docs/ or optional
dependency required, runs everywhere."""

from rag.schemas import Chunk, DocumentRecord, build_chunk_list_ids, build_document_id


def test_build_document_id_slugifies_source_path():
    assert build_document_id("docs/SEMI_HMM_PHASE_1_6_REPORT.md") == "semi_hmm_phase_1_6_report"


def test_build_document_id_handles_mixed_separators():
    assert build_document_id("docs/GRU_IDENTITY-ANALYSIS v2.md") == "gru_identity_analysis_v2"


def test_build_chunk_list_ids_zero_padded_and_ordered():
    ids = build_chunk_list_ids("foo", 3)
    assert ids == ["foo::0000", "foo::0001", "foo::0002"]


def test_document_record_to_dict_round_trips_fields():
    rec = DocumentRecord(
        document_id="foo", source="docs/FOO.md", title="Foo", document_type="project",
        status="current", authoritative=True, format="markdown", content_hash="abc123",
        updated_at="2026-08-25T00:00:00Z",
    )
    d = rec.to_dict()
    assert d["document_id"] == "foo"
    assert d["authoritative"] is True
    assert d["supersedes"] is None


def test_chunk_char_count_computed_automatically():
    c = Chunk(
        chunk_id="foo::0000", document_id="foo", document_type="project", source="docs/FOO.md",
        title="Foo", status="current", authoritative=False, version="abc123", section="Intro",
        section_level=1, chunk_index=0, content="hello world", content_hash="xyz",
    )
    assert c.char_count == len("hello world")


def test_chunk_to_metadata_is_flat_json_primitives():
    c = Chunk(
        chunk_id="foo::0000", document_id="foo", document_type="methodological", source="docs/FOO.md",
        title="Foo", status="authoritative", authoritative=True, version="abc123", section="1. Intro",
        section_level=2, chunk_index=0, content="hello world", content_hash="xyz", contains_table=True,
        created_at="2026-08-25T00:00:00Z",
    )
    meta = c.to_metadata()
    assert meta["document_id"] == "foo"
    assert meta["authoritative"] is True
    assert meta["contains_table"] is True
    assert "content" not in meta  # metadata is small/structured, content is stored separately by the vector store
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool))  # Chroma metadata must be flat JSON primitives, never None/nested

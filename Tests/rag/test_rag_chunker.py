"""
Unit tests for rag.chunker -- synthetic Markdown fixtures only (this
project's established convention: unit tests never depend on real
docs/ content, see Tests/evaluation/'s own synthetic-only discipline).
Pure stdlib, no optional dependency required, runs everywhere.
"""

from rag.chunker import build_chunks, chunk_document, extract_title, split_into_sections

SIMPLE_DOC = """# My Document

Intro paragraph, before any subsection.

## 1. First Section

Some content in the first section.

## 2. Second Section

More content here.
"""

TABLE_DOC = """# Table Document

## Results

| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |

Some prose right after the table, in the same section.
"""

LONG_DOC = """# Long Document

## Big Section

""" + "\n\n".join(f"Paragraph {i}. " + ("word " * 60) for i in range(20))


def test_extract_title():
    assert extract_title(SIMPLE_DOC) == "My Document"


def test_extract_title_missing_returns_empty():
    assert extract_title("no heading here") == ""


def test_split_into_sections_preserves_intro_and_headings():
    sections = split_into_sections(SIMPLE_DOC)
    headings = [(s.heading, s.level) for s in sections]
    assert headings == [
        ("My Document", 1),
        ("1. First Section", 2),
        ("2. Second Section", 2),
    ]


def test_chunk_document_never_splits_across_a_heading():
    chunks = chunk_document(SIMPLE_DOC)
    sections_seen = {c["section"] for c in chunks}
    assert sections_seen == {"My Document", "1. First Section", "2. Second Section"}
    for c in chunks:
        assert "## " not in c["content"], "a chunk must never contain a heading from a DIFFERENT section"


def test_chunk_document_table_stays_intact_in_one_chunk():
    chunks = chunk_document(TABLE_DOC)
    table_chunks = [c for c in chunks if c["contains_table"]]
    assert len(table_chunks) == 1
    content = table_chunks[0]["content"]
    for row in ("| A | B |", "| 1 | 2 |", "| 3 | 4 |"):
        assert row in content


def test_chunk_document_respects_max_size_except_atomic_blocks():
    chunks = chunk_document(LONG_DOC, target_chars=300, max_chars=500)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["content"]) <= 500 + 250  # small slack for the "[...]"-separated overlap prefix


def test_chunk_document_no_content_lost():
    chunks = chunk_document(LONG_DOC, target_chars=300, max_chars=500)
    for i in range(20):
        assert any(f"Paragraph {i}." in c["content"] for c in chunks), f"paragraph {i} missing from every chunk"


def test_chunk_document_overlap_present_when_section_splits():
    chunks = chunk_document(LONG_DOC, target_chars=300, max_chars=500)
    section_chunks = [c for c in chunks if c["section"] == "Big Section"]
    assert len(section_chunks) > 1
    assert any("[...]" in c["content"] for c in section_chunks[1:])


def test_chunk_document_empty_string_returns_no_chunks():
    assert chunk_document("") == []


def test_build_chunks_produces_valid_schema_objects():
    chunks = build_chunks(
        document_id="my_doc", document_type="project", source="docs/MY_DOC.md",
        title="My Document", status="current", authoritative=False, version="abc123",
        text=SIMPLE_DOC,
    )
    assert len(chunks) == 3
    ids = [c.chunk_id for c in chunks]
    assert ids == ["my_doc::0000", "my_doc::0001", "my_doc::0002"]
    assert all(c.document_id == "my_doc" for c in chunks)
    assert all(c.version == "abc123" for c in chunks)
    assert len({c.content_hash for c in chunks}) == 3  # distinct content -> distinct hashes


def test_build_chunks_content_hash_is_stable_for_unchanged_content():
    a = build_chunks("d", "project", "docs/D.md", "D", "current", False, "v1", SIMPLE_DOC)
    b = build_chunks("d", "project", "docs/D.md", "D", "current", False, "v1", SIMPLE_DOC)
    assert [c.content_hash for c in a] == [c.content_hash for c in b]

"""Unit tests for rag.hashing -- pure stdlib, no docs/ or optional
dependency required, runs everywhere."""

from rag.hashing import hash_file, hash_text


def test_hash_text_deterministic():
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_text_sensitive_to_content():
    assert hash_text("hello world") != hash_text("hello world!")


def test_hash_text_is_sha256_hex():
    h = hash_text("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_file_matches_hash_text(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome content.\n", encoding="utf-8")
    assert hash_file(p) == hash_text("# Title\n\nSome content.\n")


def test_hash_file_changes_when_content_changes(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("v1", encoding="utf-8")
    h1 = hash_file(p)
    p.write_text("v2", encoding="utf-8")
    h2 = hash_file(p)
    assert h1 != h2

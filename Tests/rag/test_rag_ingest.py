"""
Integration tests for rag.ingest -- require chromadb + sentence-transformers
(this phase's new, optional dependencies) and network access on first run
(to download the embedding model). Skips cleanly if either package isn't
importable, matching Tests/reporting/test_integration_real_val.py's
convention for optional real-environment dependencies.

Uses a small SYNTHETIC document set (monkeypatched over
rag.inventory.INCLUDED) and an isolated tmp_path persist_dir -- never
touches the real docs/ corpus or RagIndex/, so these tests are fast,
deterministic, and safe to run in any order/environment.
"""

import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from rag import ingest, inventory, vectorstore  # noqa: E402


def _write_doc(tmp_path, name, content):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / name).write_text(content, encoding="utf-8")
    return f"docs/{name}"


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    source_a = _write_doc(tmp_path, "A.md", "# Doc A\n\nContent about apples and orchards.\n")
    source_b = _write_doc(tmp_path, "B.md", "# Doc B\n\nContent about rockets and orbital mechanics.\n")
    entries = [
        inventory.InventoryEntry(source_a, "scientific", "current", False, reason="test fixture"),
        inventory.InventoryEntry(source_b, "project", "current", False, reason="test fixture"),
    ]
    monkeypatch.setattr(inventory, "INCLUDED", entries)
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)
    persist_dir = tmp_path / "RagIndex"
    return persist_dir


def test_ingestion_creates_manifest_and_chunks(fake_corpus):
    report = ingest.run_ingestion(persist_dir=fake_corpus)
    assert len(report["ingested"]) == 2
    assert report["skipped"] == []
    manifest = ingest.load_manifest(fake_corpus / "manifest.json")
    assert set(manifest["documents"].keys()) == {"a", "b"}
    assert report["total_chunks_in_collection"] >= 2


def test_ingestion_is_idempotent_on_unchanged_content(fake_corpus):
    ingest.run_ingestion(persist_dir=fake_corpus)
    report2 = ingest.run_ingestion(persist_dir=fake_corpus)
    assert report2["ingested"] == []
    assert set(report2["skipped"]) == {"a", "b"}


def test_ingestion_reembeds_when_a_document_changes(fake_corpus, tmp_path):
    ingest.run_ingestion(persist_dir=fake_corpus)
    (tmp_path / "docs" / "A.md").write_text("# Doc A\n\nCompletely different content now.\n", encoding="utf-8")
    report2 = ingest.run_ingestion(persist_dir=fake_corpus)
    assert [i["document_id"] for i in report2["ingested"]] == ["a"]
    assert report2["ingested"][0]["reason"] == "content_changed"
    assert report2["skipped"] == ["b"]


def test_ingestion_prunes_documents_removed_from_inventory(fake_corpus, monkeypatch):
    ingest.run_ingestion(persist_dir=fake_corpus)
    monkeypatch.setattr(inventory, "INCLUDED", inventory.INCLUDED[:1])  # drop "b"
    report2 = ingest.run_ingestion(persist_dir=fake_corpus)
    assert report2["removed"] == ["b"]
    manifest = ingest.load_manifest(fake_corpus / "manifest.json")
    assert "b" not in manifest["documents"]
    collection = vectorstore.get_collection(persist_dir=fake_corpus)
    assert vectorstore.existing_chunk_ids(collection, "b") == []


def test_dry_run_never_writes_manifest_or_collection(fake_corpus):
    report = ingest.run_ingestion(persist_dir=fake_corpus, dry_run=True)
    assert len(report["ingested"]) == 2  # still reports what WOULD be ingested
    assert not (fake_corpus / "manifest.json").exists()


def test_manifest_records_embedding_version(fake_corpus):
    from rag.embeddings import EMBEDDING_VERSION

    ingest.run_ingestion(persist_dir=fake_corpus)
    manifest = ingest.load_manifest(fake_corpus / "manifest.json")
    assert manifest["embedding_version"] == EMBEDDING_VERSION

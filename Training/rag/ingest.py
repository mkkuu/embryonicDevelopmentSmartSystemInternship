"""
Ingestion pipeline (Product Roadmap Phase 3 / task Phase 7):

    documents -> parse -> chunk -> metadata -> embed -> vector DB

Idempotent by design (task Phase 7's explicit requirement): a manifest at
`RagIndex/manifest.json` records each ingested document's content_hash
and the embedding-model version that produced its vectors. Re-running
`run_ingestion()` re-embeds ONLY documents whose file content changed
(hash mismatch) or whose embedding model version changed (a full-corpus
re-embed, deliberate and logged, never silent) -- an unchanged document
is skipped entirely, no vector DB write, no embedding call.

A document removed from `inventory.INCLUDED` since the last run has its
chunks deleted from the collection and its manifest entry dropped --
the corpus never accumulates orphaned entries for documents that are no
longer part of it.

CLI:
    cd Training && python -m rag.ingest [--persist-dir PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from . import inventory, vectorstore
from .chunker import build_chunks
from .embeddings import EMBEDDING_VERSION, embed_texts
from .hashing import hash_text
from .schemas import DocumentRecord, build_document_id

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = vectorstore.DEFAULT_PERSIST_DIR / "manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"embedding_version": None, "documents": {}}


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def build_document_record(entry: inventory.InventoryEntry) -> DocumentRecord:
    path = REPO_ROOT / entry.source
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("#").strip() if text else entry.source
    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    return DocumentRecord(
        document_id=build_document_id(entry.source),
        source=entry.source,
        title=title,
        document_type=entry.document_type,
        status=entry.status,
        authoritative=entry.authoritative,
        format="markdown",
        content_hash=hash_text(text),
        updated_at=mtime,
        supersedes=entry.supersedes,
        notes=entry.reason,
    )


def run_ingestion(persist_dir: Path = None, dry_run: bool = False) -> dict:
    persist_dir = persist_dir or vectorstore.DEFAULT_PERSIST_DIR
    manifest_path = persist_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    embedding_version_changed = manifest.get("embedding_version") not in (None, EMBEDDING_VERSION)
    if embedding_version_changed:
        # A full-corpus re-embed is a deliberate, visible event, not a silent
        # side effect -- every document is treated as changed this run.
        pass

    collection = None if dry_run else vectorstore.get_collection(persist_dir=persist_dir)

    report = {"skipped": [], "ingested": [], "removed": [], "errors": [],
              "embedding_version": EMBEDDING_VERSION, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    current_ids = set()
    for entry in inventory.INCLUDED:
        path = REPO_ROOT / entry.source
        if not path.exists():
            report["errors"].append({"source": entry.source, "error": "file not found"})
            continue
        record = build_document_record(entry)
        current_ids.add(record.document_id)

        prior = manifest["documents"].get(record.document_id)
        unchanged = (
            prior is not None
            and prior["content_hash"] == record.content_hash
            and not embedding_version_changed
        )
        if unchanged:
            report["skipped"].append(record.document_id)
            continue

        text = path.read_text(encoding="utf-8")
        chunks = build_chunks(
            document_id=record.document_id, document_type=record.document_type,
            source=record.source, title=record.title, status=record.status,
            authoritative=record.authoritative, version=record.content_hash, text=text,
        )
        if not dry_run:
            vectorstore.delete_document(collection, record.document_id)  # clean slate for this doc's chunks
            vectors = embed_texts([c.content for c in chunks])
            vectorstore.upsert_chunks(collection, chunks, vectors)

        manifest["documents"][record.document_id] = {
            **record.to_dict(), "n_chunks": len(chunks), "ingested_at": report["started_at"],
        }
        report["ingested"].append({"document_id": record.document_id, "n_chunks": len(chunks),
                                     "reason": "new" if prior is None else "content_changed" if prior["content_hash"] != record.content_hash else "embedding_version_changed"})

    # prune documents no longer in inventory.INCLUDED
    for doc_id in list(manifest["documents"].keys()):
        if doc_id not in current_ids:
            if not dry_run:
                vectorstore.delete_document(collection, doc_id)
            del manifest["documents"][doc_id]
            report["removed"].append(doc_id)

    manifest["embedding_version"] = EMBEDDING_VERSION
    manifest["last_run_at"] = report["started_at"]
    manifest["vector_db"] = {"backend": "chroma", "collection": vectorstore.COLLECTION_NAME, "persist_dir": str(persist_dir)}

    if not dry_run:
        save_manifest(manifest, manifest_path)
        report["total_chunks_in_collection"] = vectorstore.collection_count(collection)

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persist-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="Parse/chunk/hash only, never embeds or writes to the vector DB or manifest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    persist_dir = Path(args.persist_dir) if args.persist_dir else None
    report = run_ingestion(persist_dir=persist_dir, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""
Chroma wrapper -- the ONLY module in this package that talks to the
vector DB, so every other module (ingest.py, retrieval.py) is agnostic
to which store is behind this interface. Chosen per
docs/RAG_ARCHITECTURE.md §3's existing recommendation, re-checked this
session (Phase 5 of the task) rather than assumed still valid:

- Embedded, no separate server process -- this project has no existing
  always-on service to host a vector DB against (unlike pgvector, whose
  advantage is reusing WebApplication/'s already-running PostgreSQL;
  that migration remains the documented future option, not reversed
  here).
- Native metadata filtering (document_type/status/authoritative/source)
  -- required by Phase 10's data/RAG separation test and Phase 2's
  "retrieval should prefer CURRENT/AUTHORITATIVE" requirement; FAISS
  would need a companion metadata store bolted on, undoing the
  simplicity this prototype needs.
- Corpus scale (30 documents, a few hundred chunks, see
  docs/RAG_PHASE_3_REPORT.md §2) is exactly the "low hundreds of
  chunks" scale docs/RAG_ARCHITECTURE.md already sized Chroma for --
  still true, re-verified, not re-derived from scratch.

New dependency, added to requirements.txt this phase: `chromadb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .schemas import Chunk

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "RagIndex"
COLLECTION_NAME = "docs_v1"  # versioned name -- a future embedding-model change gets a new
# collection name (e.g. docs_v2), never a silent in-place mix of two embedding spaces in one
# collection (see docs/RAG_INGESTION.md's versioning section for the full rule).


def get_client(persist_dir: Optional[Path] = None):
    import chromadb

    persist_dir = persist_dir or DEFAULT_PERSIST_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(client=None, persist_dir: Optional[Path] = None, name: str = COLLECTION_NAME):
    client = client or get_client(persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(collection, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[c.to_metadata() for c in chunks],
    )


def delete_document(collection, document_id: str) -> None:
    """Removes every chunk belonging to one document_id -- used when a
    document's content_hash changes (old chunk_ids may not line up 1:1
    with the new chunking, e.g. a document that grew a new section), so
    a re-ingested document is replaced cleanly rather than leaving
    orphaned stale chunks behind."""
    collection.delete(where={"document_id": document_id})


def existing_chunk_ids(collection, document_id: str) -> List[str]:
    result = collection.get(where={"document_id": document_id}, include=[])
    return result.get("ids", [])


def query(collection, query_embedding: List[float], top_k: int = 5,
          where: Optional[Dict] = None) -> List[dict]:
    result = collection.query(
        query_embeddings=[query_embedding], n_results=top_k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({
            "chunk_id": cid, "content": doc, "metadata": meta,
            "distance": dist, "score": 1.0 - dist,  # cosine distance -> similarity, for a more intuitive "higher is better" score
        })
    return out


def collection_count(collection) -> int:
    return collection.count()

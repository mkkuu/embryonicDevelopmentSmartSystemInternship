"""
Retrieval prototype (task Phase 9) -- question in, ranked chunks out.
Deliberately minimal: no LLM, no answer synthesis, no orchestrator (all
explicitly out of scope for this phase, see docs/RAG_PHASE_3_REPORT.md
§16 and the task's own "PHASE 17 -- CE QU'IL NE FAUT PAS FAIRE").

CLI:
    cd Training && python -m rag.retrieval "Qu'est-ce que le Semi-HMM ?" --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import vectorstore
from .embeddings import embed_query


def retrieve(
    query: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
    status: Optional[str] = None,
    authoritative_only: bool = False,
    persist_dir: Optional[Path] = None,
) -> List[dict]:
    """Returns top-k chunks, each carrying score/document/section/metadata/
    content -- exactly the shape the task's Phase 9 asks for. Filters are
    applied natively by Chroma (metadata filtering, not a post-hoc Python
    filter), matching docs/RAG_ARCHITECTURE.md §3's retrieval-strategy
    recommendation."""
    collection = vectorstore.get_collection(persist_dir=persist_dir)
    where: Dict = {}
    conditions = []
    if document_type is not None:
        conditions.append({"document_type": document_type})
    if status is not None:
        conditions.append({"status": status})
    if authoritative_only:
        conditions.append({"authoritative": True})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    query_embedding = embed_query(query)
    return vectorstore.query(collection, query_embedding, top_k=top_k, where=where or None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--document-type", default=None, choices=["scientific", "project", "methodological"])
    p.add_argument("--status", default=None, choices=["authoritative", "current", "historical", "experimental"])
    p.add_argument("--authoritative-only", action="store_true")
    p.add_argument("--persist-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve(
        args.query, top_k=args.top_k, document_type=args.document_type,
        status=args.status, authoritative_only=args.authoritative_only,
        persist_dir=Path(args.persist_dir) if args.persist_dir else None,
    )
    for r in results:
        print(f"[{r['score']:.3f}] {r['metadata']['source']} :: {r['metadata']['section']} "
              f"({r['metadata']['status']}{'★' if r['metadata']['authoritative'] else ''})")
        print(f"    {r['content'][:200].replace(chr(10), ' ')}...")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

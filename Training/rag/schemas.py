"""
Document/chunk data model for the RAG corpus -- plain dataclasses,
JSON-serializable, matching this project's established "never pickle,
always inspectable JSON" convention (see e.g. evaluation/models/*.py's
own save()/load()).

Two levels: DocumentRecord (one per source file, carries the
inventory/status/versioning metadata) and Chunk (one per retrievable
unit, carries a copy of the parent document's metadata plus its own
section/content -- denormalized deliberately, so a retrieval result is
self-contained and never needs a second lookup to know what it is or
whether it should be trusted).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Status vocabulary (Product Roadmap Phase 3, Phase 2 of the task)
# ---------------------------------------------------------------------------
# AUTHORITATIVE: the single reference for its topic, retrieval should prefer
#   it over any other document on the same subject.
# CURRENT: accurate and actively relevant, not necessarily THE reference.
# HISTORICAL: accurate but describes a closed/superseded branch or an earlier
#   snapshot -- still citable (this project never deletes a true negative
#   result), but should not outrank a CURRENT/AUTHORITATIVE doc on the same
#   question.
# OBSOLETE: known to be wrong or contradicted by a later document -- not used
#   in this corpus (see inventory.py's EXCLUDED list); kept here as a valid
#   enum value in case a future ingestion pass needs to tag one instead of
#   silently dropping it.
# EXPERIMENTAL: a plan/proposal never executed, or an in-progress branch.
DOCUMENT_STATUS_VALUES = ("authoritative", "current", "historical", "obsolete", "experimental")

DOCUMENT_TYPE_VALUES = ("scientific", "project", "methodological")


@dataclass
class DocumentRecord:
    document_id: str          # stable slug derived from the source path, e.g. "semi_hmm_phase_1_6_report"
    source: str                # repo-relative path, e.g. "docs/SEMI_HMM_PHASE_1_6_REPORT.md"
    title: str                  # first H1 in the file
    document_type: str           # one of DOCUMENT_TYPE_VALUES
    status: str                   # one of DOCUMENT_STATUS_VALUES
    authoritative: bool
    format: str                    # "markdown" (only format supported this phase, see chunker.py)
    content_hash: str               # sha256 of the raw file bytes -- drives ingestion idempotence
    updated_at: str                  # file mtime, ISO 8601
    supersedes: Optional[str] = None       # document_id this one replaces, or None
    superseded_by: Optional[str] = None     # document_id that replaces this one, or None
    notes: str = ""                          # why this classification, in the inventory author's own words

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str              # f"{document_id}::{chunk_index:04d}"
    document_id: str
    document_type: str          # denormalized from DocumentRecord, for metadata-filtered retrieval
    source: str                  # denormalized
    title: str                    # denormalized
    status: str                    # denormalized
    authoritative: bool
    version: str                    # denormalized DocumentRecord.content_hash -- "which doc version produced this chunk"
    section: str                     # heading path, e.g. "9. Benchmark before/after" or "" for pre-heading content
    section_level: int                # 0 = document-level (no heading yet), 1 = H1, 2 = H2, ...
    chunk_index: int
    content: str
    content_hash: str                  # sha256 of `content` -- drives re-embedding idempotence
    char_count: int = field(init=False)
    contains_table: bool = False
    created_at: str = ""                 # ingestion run timestamp, ISO 8601

    def __post_init__(self) -> None:
        self.char_count = len(self.content)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_metadata(self) -> dict:
        """Chroma metadata dict -- flat, JSON-primitive values only (Chroma
        rejects None/nested values in metadata), everything needed to
        judge a retrieval result's trustworthiness WITHOUT a second
        lookup: document_type/status/authoritative for filtering and
        ranking, source/section/title for citation, version for "which
        doc version produced this answer" (docs/PRODUCT_ARCHITECTURE.md
        §11's versioning goal, applied to the RAG side)."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "source": self.source,
            "title": self.title,
            "status": self.status,
            "authoritative": self.authoritative,
            "version": self.version,
            "section": self.section,
            "section_level": self.section_level,
            "chunk_index": self.chunk_index,
            "contains_table": self.contains_table,
            "created_at": self.created_at,
        }


def build_document_id(source_path: str) -> str:
    """docs/SEMI_HMM_PHASE_1_6_REPORT.md -> semi_hmm_phase_1_6_report.
    Deterministic, human-readable, stable across re-ingestion runs as
    long as the filename doesn't change -- content changes are tracked
    separately via content_hash, not by minting a new document_id."""
    import re
    from pathlib import Path

    stem = Path(source_path).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug


def build_chunk_list_ids(document_id: str, n: int) -> List[str]:
    return [f"{document_id}::{i:04d}" for i in range(n)]

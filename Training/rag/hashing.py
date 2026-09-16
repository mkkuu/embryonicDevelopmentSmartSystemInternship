"""Content hashing -- the sole mechanism driving ingestion idempotence
(Phase 7's "document hash -> if unchanged -> don't re-embed" requirement).
Plain sha256 over UTF-8 bytes, nothing model- or library-specific, so a
hash computed today is reproducible by any future tool that reads the
same file."""

from __future__ import annotations

import hashlib


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path) -> str:
    from pathlib import Path

    return hash_text(Path(path).read_text(encoding="utf-8"))

"""
Reader for the curated Istanbul Consensus 2025 corpus.

WHY A READER AND NOT THE RAG
----------------------------
The corpus lives at `docs/corpus/ISTANBUL_CONSENSUS_2025.md`, deliberately in a
SUBDIRECTORY so that `rag.inventory`'s non-recursive `docs/*.md` glob cannot see
it (see `docs/corpus/ISTANBUL_CONSENSUS_2025_TRACEABILITY.md` and
`Training/rag/validate_istanbul_corpus.py`). It is NOT indexed: no chunk of it
is in Chroma, and this module adds none. The 432 existing embeddings are
untouched, and nothing here imports `rag`.

v1 retrieval is therefore LEXICAL over the structured blocks below, not
semantic. That is a deliberate v1 choice, not an oversight: the corpus is
already organised as ~138 identified blocks carrying their own type, page,
table and evidence grade, so a structured lexical match returns a citable block
rather than a similarity score. A Chroma namespace remains the natural next
step and is recorded as such in `docs/VALIDATOR_V1_IMPLEMENTATION.md`.

This module re-parses the Markdown rather than reusing
`rag.validate_istanbul_corpus.build_traceability()`: that function is an
auditing tool and discards the block BODY, which is exactly what retrieval
needs.

NOTHING IS INVENTED HERE. A block's text is the corpus text; a field that the
corpus does not print stays `None`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "corpus" / "ISTANBUL_CONSENSUS_2025.md"
SOURCE_NAME = "ISTANBUL_CONSENSUS_2025"

_GRADE_PATTERN = re.compile(r"\b(Very low|Low|Moderate|High)\s*\(?([⊕○]{2,4})?\)?")
_PAGE_PATTERN = re.compile(r"pp?\.\s*\d{1,3}(?:[–-]\d{1,3})?")
_TABLE_PATTERN = re.compile(r"\bTables?\s+\d{1,2}|\bFigure\s+\d{1,2}")


@dataclass
class CorpusBlock:
    block_id: str
    subject: str
    text: str
    evidence_type: Optional[str] = None
    page: Optional[str] = None
    table: Optional[str] = None
    section: Optional[str] = None
    evidence_grade: Optional[str] = None

    @property
    def is_limitation(self) -> bool:
        return bool(self.evidence_type and "limitation" in self.evidence_type.lower())

    @property
    def is_knowledge_gap(self) -> bool:
        return bool(self.evidence_type and "gap" in self.evidence_type.lower())

    @property
    def is_recommendation(self) -> bool:
        return bool(self.evidence_type and "recommendation" in self.evidence_type.lower())

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"block_id": self.block_id, "subject": self.subject,
                "evidence_type": self.evidence_type, "page": self.page,
                "table": self.table, "section": self.section,
                "evidence_grade": self.evidence_grade, "text": self.text}


def _locators(raw: str) -> Dict[str, Optional[str]]:
    pages = _PAGE_PATTERN.findall(raw)
    tables = _TABLE_PATTERN.findall(raw)
    sections = re.findall(r"Section\s+\d", raw)
    return {"page": ", ".join(dict.fromkeys(pages)) or None,
            "table": ", ".join(dict.fromkeys(tables)) or None,
            "section": ", ".join(dict.fromkeys(sections)) or None}


def _grade(raw: str) -> Optional[str]:
    m = _GRADE_PATTERN.search(raw)
    if not m:
        return None
    return f"{m.group(1)} ({m.group(2)})" if m.group(2) else m.group(1)


def _clean(text: str) -> str:
    lines = [re.sub(r"^\s*>+\s?", "", line).strip() for line in text.splitlines()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def parse_corpus(path: Path = DEFAULT_CORPUS_PATH) -> List[CorpusBlock]:
    """Every identified block of the curated corpus, in document order.

    Three block shapes exist and all three are parsed:
      A. `### IC2025-X — Subject` + `- **type:**` / `- **source:**` + body
      B. `- **IC2025-X** — …` prose bullets
      C. index-table rows `| IC2025-X | subject | source |`
    """
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    blocks: List[CorpusBlock] = []
    seen: set = set()

    # -- shape A ------------------------------------------------------------
    for chunk in re.split(r"\n(?=### )", raw):
        m = re.match(r"### ((?:IC2025-[A-Z0-9-]+)(?:\s*…\s*IC2025-[A-Z0-9-]+)?)\s*—\s*(.+)", chunk)
        if not m:
            continue
        bid, subject = m.group(1).strip(), m.group(2).strip()
        tm = re.search(r"- \*\*type:\*\*\s*(.+)", chunk)
        sm = re.search(r"- \*\*source:\*\*\s*(.+)", chunk)
        src = sm.group(1).strip() if sm else chunk
        loc = _locators(src)
        blocks.append(CorpusBlock(block_id=bid, subject=subject, text=_clean(chunk),
                                  evidence_type=(tm.group(1).strip() if tm else None),
                                  evidence_grade=_grade(chunk), **loc))
        seen.add(bid)

    # -- shape B ------------------------------------------------------------
    for m in re.finditer(
            r"- \*\*(IC2025-[A-Z0-9-]+)\*\*\s*—\s*(.*?)(?=\n\s*- \*\*IC2025-|\n### |\n## |\n---|\Z)",
            raw, re.S):
        bid, body = m.group(1), m.group(2)
        if bid in seen:
            continue
        tm = re.search(r"\*\*type:\*\*\s*([^.\n]+)", body)
        sm = re.search(r"\*\*source:\*\*\s*([^\n]+)", body)
        loc = _locators(sm.group(1) if sm else body)
        blocks.append(CorpusBlock(block_id=bid, subject=_clean(body)[:120],
                                  text=_clean(body),
                                  evidence_type=(tm.group(1).strip() if tm else _type_for(bid)),
                                  evidence_grade=_grade(body), **loc))
        seen.add(bid)

    # -- shape C ------------------------------------------------------------
    for m in re.finditer(r"^\|\s*(IC2025-[A-Z0-9-]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$",
                         raw, re.MULTILINE):
        bid, subject, src = m.group(1), m.group(2), m.group(3)
        if bid in seen:
            continue
        loc = _locators(src)
        blocks.append(CorpusBlock(block_id=bid, subject=_clean(subject),
                                  text=_clean(subject + " " + src),
                                  evidence_type=_type_for(bid),
                                  evidence_grade=_grade(subject), **loc))
        seen.add(bid)
    return blocks


_TYPE_BY_PREFIX = {
    "IC2025-AC-A": "term_mention", "IC2025-AC-B": "absent_definition",
    "IC2025-AC-C": "review finding", "IC2025-AC-D": "recommendation",
    "IC2025-AC-E": "limitation", "IC2025-ABSENT": "absent_from_source",
    "IC2025-VOC": "vocabulary", "IC2025-KG": "knowledge_gap",
    "IC2025-LIM": "limitation", "IC2025-CP": "consensus point",
}


def _type_for(block_id: str) -> Optional[str]:
    for prefix, typ in _TYPE_BY_PREFIX.items():
        if block_id.startswith(prefix):
            return typ
    return None


class Corpus:
    """Loaded once, queried many times. Read-only."""

    def __init__(self, path: Path = DEFAULT_CORPUS_PATH):
        self.path = path
        self.blocks: List[CorpusBlock] = parse_corpus(path)
        self._by_id = {b.block_id: b for b in self.blocks}

    @property
    def available(self) -> bool:
        return bool(self.blocks)

    def get(self, block_id: str) -> Optional[CorpusBlock]:
        return self._by_id.get(block_id)

    def limitations(self) -> List[CorpusBlock]:
        return [b for b in self.blocks if b.is_limitation]

    def knowledge_gaps(self) -> List[CorpusBlock]:
        return [b for b in self.blocks if b.is_knowledge_gap]

    def stage_vocabulary_block(self) -> Optional[CorpusBlock]:
        """IC2025-VOC-01 -- the exhaustive list of morphokinetic variables the
        source actually prints, plus the ones it never uses."""
        return self.get("IC2025-VOC-01")

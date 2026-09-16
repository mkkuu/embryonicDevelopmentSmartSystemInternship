"""
Structure-aware Markdown chunking (Product Roadmap Phase 3 / task Phase 4).

Every document in this corpus is Markdown (see inventory.py -- `format`
is always "markdown" this phase; PDF/other formats are explicitly out of
scope, see module docstring below). Chunking rule, in order of priority:

1. NEVER split across a heading boundary -- a chunk belongs to exactly
   one section. This is what "preserve titles/sections/subsections"
   means concretely: the section boundary is the primary, non-negotiable
   split point, character count is secondary.
2. NEVER split a fenced code block (```...```) or a Markdown table (a
   contiguous run of `|`-led lines) across two chunks -- these are
   treated as atomic, even if that pushes one chunk over the target
   size. A table row or an equation fragment split mid-block is useless
   to a reader and useless to a retriever.
3. Within a section, split on paragraph boundaries (blank-line-separated
   blocks) toward a TARGET size, never exceeding a hard MAX -- except for
   a single atomic block (table/code) larger than MAX on its own, which
   is kept whole regardless (rule 2 always wins over the size cap).
4. When a section produces more than one chunk, carry a small trailing
   overlap (the previous chunk's tail) into the next chunk's start, so a
   referent ("this gain", "the same finding") isn't lost across the cut
   -- matches the overlap already specified in docs/RAG_ARCHITECTURE.md
   §3, applied here for the first time.

Target/max sizes are in characters, not tokens (no tokenizer dependency
for this foundation pass) -- ~4 characters/token is a standard rough
English-prose estimate, so TARGET_CHARS=1800/MAX_CHARS=2600 approximates
RAG_ARCHITECTURE.md §3's existing "300-500 tokens" recommendation
(1800/4≈450, 2600/4≈650, biased slightly larger to reduce how often rule
2's atomic-block exception is needed).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Tuple

from .schemas import Chunk

TARGET_CHARS = 1800
MAX_CHARS = 2600
OVERLAP_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE_RE = re.compile(r"^\s*```")


@dataclass
class _Section:
    heading: str      # "" for the pre-heading intro block
    level: int         # 0 for intro, 1 for H1, 2 for H2, ...
    content: str


def extract_title(text: str) -> str:
    m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def split_into_sections(text: str) -> List[_Section]:
    """Splits on EVERY heading line (any level 1-6), producing a flat
    sequence -- deliberately not a nested tree: this project's docs are
    shallow enough (verified this session: at most H1 -> H2/H3, never
    deeper) that a flat sequence with the heading's own level attached is
    sufficient for both chunk metadata and human-readable citation, and
    is far simpler to chunk correctly than a nested-tree walk."""
    matches = list(_HEADING_RE.finditer(text))
    sections: List[_Section] = []

    first_start = matches[0].start() if matches else len(text)
    intro = text[:first_start].strip()
    if intro:
        sections.append(_Section(heading="", level=0, content=intro))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[body_start:body_end].strip()
        if content:
            sections.append(_Section(heading=heading, level=level, content=content))
    return sections


def _split_atomic_blocks(content: str) -> List[Tuple[str, str]]:
    """Splits section content into (block_text, kind) pieces on
    blank-line paragraph boundaries, EXCEPT that any fenced-code-block or
    contiguous Markdown-table run is merged into one atomic piece even if
    it spans what would otherwise be several blank-line-separated
    paragraphs (a table's own rows are never blank-line-separated in
    practice, but a code fence containing a blank line must still not be
    torn apart -- handled by scanning line-by-line, not by a naive
    ``split("\\n\\n")``). kind is one of "prose"/"table"/"fence" -- kept
    distinct (not collapsed into one "atomic" bool) so a chunk can be
    correctly labeled `contains_table` without also firing on an
    unrelated code fence."""
    lines = content.split("\n")
    blocks: List[Tuple[str, str]] = []
    buf: List[str] = []
    mode = None  # None | "table" | "fence"

    def flush(kind: str) -> None:
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append((text, kind))
            buf.clear()

    for line in lines:
        if mode == "fence":
            buf.append(line)
            if _FENCE_RE.match(line):
                flush(kind="fence")
                mode = None
            continue
        if _FENCE_RE.match(line):
            flush(kind="prose")
            buf.append(line)
            mode = "fence"
            continue
        if mode == "table":
            if _TABLE_LINE_RE.match(line):
                buf.append(line)
                continue
            flush(kind="table")
            mode = None
            # fall through: this line starts a new (non-table) block
        if _TABLE_LINE_RE.match(line):
            flush(kind="prose")
            buf.append(line)
            mode = "table"
            continue
        if line.strip() == "" and mode is None:
            flush(kind="prose")
            continue
        buf.append(line)

    flush(kind=(mode or "prose"))
    return blocks


def _pack_blocks(blocks: List[Tuple[str, str]], target: int, max_chars: int) -> List[Tuple[str, bool, bool]]:
    """Greedily packs (block, kind) pieces into chunks up to `target`
    chars, never exceeding `max_chars` UNLESS a single atomic (table or
    fence) block already exceeds it on its own (rule 2 beats the size
    cap). Returns (content, contains_table, is_pure_atomic):
    - contains_table: True iff ANY block folded into this pack was a
      table, regardless of position (a table preceded/followed by prose
      in the same chunk still counts).
    - is_pure_atomic: True iff EVERY block folded into this pack was
      atomic (table or fence) -- i.e. this pack is nothing but
      table/code content, not table+prose mixed. Used only to decide
      whether prepending trailing-context overlap is safe (§ module
      docstring rule 2) -- overlap is skipped for a pure-atomic pack so
      a retrieved table/code chunk stays exactly what it claims to be."""
    packed: List[Tuple[str, bool, bool]] = []
    current: List[str] = []
    current_len = 0
    current_has_table = False
    current_all_atomic = True

    def flush() -> None:
        nonlocal current, current_len, current_has_table, current_all_atomic
        if current:
            packed.append(("\n\n".join(current), current_has_table, current_all_atomic))
        current, current_len, current_has_table, current_all_atomic = [], 0, False, True

    for block, kind in blocks:
        is_atomic = kind in ("table", "fence")
        block_len = len(block)
        would_be = current_len + (2 if current else 0) + block_len
        if current and would_be > max_chars:
            flush()
        if not current and block_len > max_chars:
            packed.append((block, kind == "table", is_atomic))  # oversized atomic block, kept whole
            continue
        current.append(block)
        current_len += (2 if len(current) > 1 else 0) + block_len
        current_has_table = current_has_table or (kind == "table")
        current_all_atomic = current_all_atomic and is_atomic
        if current_len >= target:
            flush()
    flush()
    return packed


def chunk_document(
    text: str,
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[dict]:
    """Returns a list of {"section", "section_level", "content",
    "contains_table"} dicts, global chunk order preserved -- caller
    (ingest.py) attaches document-level metadata and content hashes to
    build real Chunk objects; kept dict-only here so this function stays
    testable without constructing a full DocumentRecord first."""
    sections = split_into_sections(text)
    out: List[dict] = []

    for section in sections:
        blocks = _split_atomic_blocks(section.content)
        packed = _pack_blocks(blocks, target_chars, max_chars)
        prev_tail = ""
        for i, (content, contains_table, is_pure_atomic) in enumerate(packed):
            full_content = content
            if i > 0 and prev_tail and not is_pure_atomic:
                # trailing-context overlap -- never prepended onto a pack that
                # is nothing but table/code content, which must stay exactly
                # what it claims to be for a reader/retriever to trust it
                # verbatim (§ module docstring rule 2).
                full_content = prev_tail + "\n\n[...]\n\n" + content
            out.append({
                "section": section.heading or "(intro)",
                "section_level": section.level,
                "content": full_content,
                "contains_table": contains_table,
            })
            prev_tail = content[-overlap_chars:] if len(content) > overlap_chars else content
    return out


def build_chunks(document_id: str, document_type: str, source: str, title: str,
                  status: str, authoritative: bool, version: str, text: str) -> List[Chunk]:
    from .hashing import hash_text

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    raw_chunks = chunk_document(text)
    chunks: List[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=f"{document_id}::{idx:04d}",
            document_id=document_id,
            document_type=document_type,
            source=source,
            title=title,
            status=status,
            authoritative=authoritative,
            version=version,
            section=rc["section"],
            section_level=rc["section_level"],
            chunk_index=idx,
            content=rc["content"],
            content_hash=hash_text(rc["content"]),
            contains_table=rc["contains_table"],
            created_at=now,
        ))
    return chunks

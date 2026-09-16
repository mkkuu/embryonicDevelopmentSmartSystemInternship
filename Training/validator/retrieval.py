"""
Lexical retrieval over the curated Istanbul corpus.

NO embedding, NO Chroma, NO change to the 432 existing vectors. Scoring is a
transparent term overlap with small, explicit boosts -- a reader can reproduce
any score by hand, which is the point for a validator whose output must be
auditable.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .corpus import Corpus, CorpusBlock
from .provenance import _norm

_WORD = re.compile(r"[a-z0-9+]{3,}")
_STOP = frozenset("""the and for with that this from are was were not但 les des une que qui
pour dans est sont avec sur par plus pas ete etre cette ces son ses aux aussi
selon donc mais tout tous elle nous vous leur leurs the of to in on at as it is
""".split())

# Topic -> extra terms that make the right blocks win without inventing content.
TOPIC_TERMS: Dict[str, Tuple[str, ...]] = {
    "direct_cleavage": ("direct", "cleavage", "abnormal", "dc1", "dc2"),
    "reverse_cleavage": ("reverse", "cleavage", "abnormal"),
    "chaotic_division": ("irregular", "chaotic", "division", "abnormal"),
    "expected_development": ("cleavage", "doubles", "cell", "number", "order", "stage"),
    "timing": ("hpi", "hours", "post-insemination", "median", "timing"),
    "skip": ("cleavage", "abnormal"),
    "uncertainty": ("evidence", "limitation"),
    "compatibility": ("recommendation", "consensus", "order"),
    "clinical": ("clinical", "standard", "care", "judgement"),
}


def _terms(text: str) -> List[str]:
    return [w for w in _WORD.findall(_norm(text)) if w not in _STOP]


def search(corpus: Corpus, query: str, markers: Optional[List[str]] = None,
           top_k: int = 3) -> List[Tuple[CorpusBlock, float]]:
    """Top-k corpus blocks for `query`, each with its score. Empty when the
    corpus is absent -- the caller must then report `retrieval_miss`, never
    invent content."""
    if not corpus.available:
        return []
    q = set(_terms(query))
    for m in (markers or []):
        q |= set(TOPIC_TERMS.get(m, ()))
    if not q:
        return []
    scored: List[Tuple[CorpusBlock, float]] = []
    for block in corpus.blocks:
        terms = set(_terms(block.text))
        if not terms:
            continue
        overlap = len(q & terms)
        if not overlap:
            continue
        score = overlap / (len(q) ** 0.5 + len(terms) ** 0.5)
        # A block that literally names the queried pattern outranks a block
        # that merely shares vocabulary with it.
        if any(t in _norm(block.text) for t in ("direct cleavage", "reverse cleavage",
                                                "irregular chaotic division")) \
                and any(m in (markers or []) for m in ("direct_cleavage", "reverse_cleavage",
                                                       "chaotic_division")):
            score *= 2.0
        scored.append((block, round(score, 4)))
    scored.sort(key=lambda pair: (-pair[1], pair[0].block_id))
    return scored[:top_k]

"""
Text embedding for the RAG corpus -- a SEPARATE embedding space from the
scientific pipeline's ResNet18 image embeddings (Training/embeddings/).
Never imports from, or shares a model/dimension/cache with, that package
-- see this task's own explicit instruction and docs/RAG_ARCHITECTURE.md
§3 ("Embedding model for the RAG corpus is a separate decision from the
ResNet18 embeddings").

Model choice: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
----------------------------------------------------------------------------
- 384-dim, ~470MB, runs on CPU in well under a second per chunk at this
  corpus's scale (a few hundred chunks) -- no GPU required, unlike the
  scientific pipeline.
- MULTILINGUAL, not just English -- required, not a nicety: this
  project's docs/ corpus is almost entirely English prose, but the
  actual users (this task's own Phase 9 benchmark questions) ask in
  FRENCH ("Qu'est-ce que le Semi-HMM ?", "Pourquoi utiliser un modèle de
  durée explicite ?"). An English-only model (the initial choice this
  session, `all-MiniLM-L6-v2`) was tried first and measurably failed
  cross-lingual retrieval in testing (Tests/rag/test_retrieval.py's own
  French-query fixture retrieved an unrelated document first) -- switched
  to this multilingual model specifically because of that measured
  failure, not a hypothetical concern. Retested after switching: same
  fixture now retrieves the correct document (see
  docs/RAG_PHASE_3_REPORT.md §6 for the before/after).
- Local inference, no API key, no per-call cost, no data leaving the
  machine -- matches this project's "minimize what crosses an external
  boundary" posture (docs/LLM_ORCHESTRATION.md §6) and keeps this
  prototype runnable without a Decision Required on an external
  embedding API.
- Requires ONE network fetch on first use (the model weights, from
  Hugging Face) -- documented, not hidden: EMBEDDING_MODEL_NAME below is
  the exact, pinned identifier a fresh environment needs to reproduce
  this corpus's embeddings bit-for-bit (same model = same vectors, this
  model is deterministic at inference time, no sampling involved).
- Not a hard requirement of the rest of this package: schemas.py,
  chunker.py, inventory.py, hashing.py all import and run with zero
  dependency on `sentence_transformers` -- only this module and
  ingest.py's/retrieval.py's actual embedding step need it, so a
  contributor without the package installed can still read/test the
  document model and chunking logic (see Tests/rag/'s skip guards).

New dependency, added to requirements.txt this phase: `sentence-transformers`.
"""

from __future__ import annotations

from typing import List, Optional

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDING_VERSION = f"{EMBEDDING_MODEL_NAME}@dim{EMBEDDING_DIM}"  # part of every chunk's stored provenance

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """L2-normalized embeddings (normalize_embeddings=True) -- required
    for cosine-similarity search to be equivalent to a plain dot product,
    which is what Chroma's default "cosine" space assumes; documented
    here rather than left implicit so a future re-implementation doesn't
    silently drop the normalization and quietly change every similarity
    score."""
    model = _get_model()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]

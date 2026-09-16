"""
Integration tests for rag.retrieval -- require chromadb + sentence-transformers,
skip cleanly otherwise (see test_ingest.py's module docstring for the
same convention). Builds a small SYNTHETIC, topically-distinct index in
an isolated tmp_path so relevance and metadata-filtering assertions are
unambiguous, rather than depending on the real docs/ corpus's content.
"""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from rag import retrieval, vectorstore  # noqa: E402
from rag.chunker import build_chunks  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402


@pytest.fixture
def small_index(tmp_path):
    docs = [
        dict(document_id="hmm_doc", document_type="methodological", source="docs/HMM.md",
             title="HMM", status="authoritative", authoritative=True, version="v1",
             text="# HMM\n\n## Definition\n\nA Hidden Markov Model represents developmental phase transitions as a discrete latent state sequence, with a chronological transition matrix estimated by counting observed segment-to-segment moves in the training videos.\n"),
        dict(document_id="semihmm_doc", document_type="methodological", source="docs/SEMIHMM.md",
             title="Semi-HMM", status="authoritative", authoritative=True, version="v1",
             text="# Semi-HMM\n\n## Why an explicit duration model\n\nThe Semi-HMM replaces the HMM's implicit geometric self-loop with an explicit per-phase duration distribution, motivated by observed dwell-time heterogeneity across phases.\n"),
        dict(document_id="gru_doc", document_type="project", source="docs/GRU.md",
             title="GRU vs Identity Dynamics", status="current", authoritative=False, version="v1",
             text="# GRU vs Identity Dynamics\n\n## Comparison\n\nThe GRU outperforms Identity Dynamics on AUROC and PR-AUC but is not a calibrated classifier and should be treated as a signal, not a final answer.\n"),
        dict(document_id="test_lock_doc", document_type="project", source="docs/TESTLOCK.md",
             title="Test Split Policy", status="authoritative", authoritative=True, version="v1",
             text="# Test Split Policy\n\n## Why Test is locked\n\nThe Test split is reserved for a single pre-registered final evaluation and must never be used during iterative model development to avoid overfitting to it.\n"),
    ]
    collection = vectorstore.get_collection(persist_dir=tmp_path)
    for d in docs:
        chunks = build_chunks(d["document_id"], d["document_type"], d["source"], d["title"],
                               d["status"], d["authoritative"], d["version"], d["text"])
        vectors = embed_texts([c.content for c in chunks])
        vectorstore.upsert_chunks(collection, chunks, vectors)
    return tmp_path


def test_retrieval_returns_topically_relevant_chunk_first(small_index):
    results = retrieval.retrieve("Pourquoi utiliser un modèle de durée explicite ?", top_k=2, persist_dir=small_index)
    assert results[0]["metadata"]["document_id"] == "semihmm_doc"


def test_retrieval_distinguishes_gru_vs_identity_question(small_index):
    results = retrieval.retrieve("Quelle est la différence entre GRU et Identity Dynamics ?", top_k=2, persist_dir=small_index)
    assert results[0]["metadata"]["document_id"] == "gru_doc"


def test_retrieval_finds_test_lock_rationale(small_index):
    results = retrieval.retrieve("Pourquoi le Test est-il verrouillé ?", top_k=2, persist_dir=small_index)
    assert results[0]["metadata"]["document_id"] == "test_lock_doc"


def test_retrieval_result_shape_has_required_fields(small_index):
    results = retrieval.retrieve("HMM", top_k=1, persist_dir=small_index)
    r = results[0]
    for key in ("chunk_id", "content", "metadata", "distance", "score"):
        assert key in r
    for key in ("document_id", "document_type", "source", "title", "status", "authoritative", "section"):
        assert key in r["metadata"]


def test_retrieval_document_type_filter(small_index):
    results = retrieval.retrieve("comparaison de modèles", top_k=10, document_type="methodological", persist_dir=small_index)
    assert results
    assert all(r["metadata"]["document_type"] == "methodological" for r in results)


def test_retrieval_authoritative_only_filter(small_index):
    results = retrieval.retrieve("comparaison de modèles", top_k=10, authoritative_only=True, persist_dir=small_index)
    assert results
    assert all(r["metadata"]["authoritative"] is True for r in results)
    assert all(r["metadata"]["document_id"] != "gru_doc" for r in results)  # gru_doc is authoritative=False


def test_retrieval_top_k_respected(small_index):
    results = retrieval.retrieve("modèle", top_k=1, persist_dir=small_index)
    assert len(results) == 1

"""
Integration tests for the RAG-backed tools (retrieve_documents,
get_phase_information) -- require chromadb + sentence-transformers, skip
cleanly otherwise (matches Tests/rag/test_rag_retrieval.py's own
convention). Builds a small SYNTHETIC, topically-distinct index in an
isolated tmp_path -- never touches the real docs/ corpus or RagIndex/.
"""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from orchestrator import tools  # noqa: E402
from rag import vectorstore  # noqa: E402
from rag.chunker import build_chunks  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402


@pytest.fixture
def small_index(tmp_path):
    docs = [
        dict(document_id="semihmm_doc", document_type="methodological", source="docs/SEMIHMM.md",
             title="Semi-HMM", status="authoritative", authoritative=True, version="v1",
             text="# Semi-HMM\n\n## Why an explicit duration model\n\nThe Semi-HMM replaces the HMM's "
                  "implicit geometric self-loop with an explicit per-phase duration distribution.\n"),
        dict(document_id="phase_doc", document_type="scientific", source="docs/PHASES.md",
             title="Developmental phases", status="authoritative", authoritative=True, version="v1",
             text="# Developmental phases\n\n## t6\n\nt6 marks the six-cell developmental stage, one of "
                  "the phases the shared emission classifier struggles with most.\n"),
    ]
    collection = vectorstore.get_collection(persist_dir=tmp_path)
    for d in docs:
        chunks = build_chunks(d["document_id"], d["document_type"], d["source"], d["title"],
                               d["status"], d["authoritative"], d["version"], d["text"])
        vectors = embed_texts([c.content for c in chunks])
        vectorstore.upsert_chunks(collection, chunks, vectors)
    return tmp_path


def test_retrieve_documents_returns_chunk_shape(small_index):
    results = tools.retrieve_documents("Pourquoi un modele de duree explicite ?", top_k=2, persist_dir=small_index)
    assert results
    for key in ("chunk_id", "content", "metadata", "distance", "score"):
        assert key in results[0]


def test_retrieve_documents_finds_the_topically_relevant_chunk(small_index):
    results = tools.retrieve_documents("Pourquoi un modele de duree explicite ?", top_k=1, persist_dir=small_index)
    assert results[0]["metadata"]["document_id"] == "semihmm_doc"


def test_get_phase_information_returns_results_and_a_gap_warning(small_index):
    out = tools.get_phase_information("t6", persist_dir=small_index)
    assert out["phase_id"] == "t6"
    assert "results" in out
    assert any("no dedicated phase-metadata index" in w for w in out["warnings"])


def test_get_phase_information_never_returns_a_live_numeric_prediction(small_index):
    # Structural check mirroring docs/RAG_PHASE_3_REPORT.md sec 12: the RAG
    # corpus (real or synthetic) never contains a live per-window probability.
    out = tools.get_phase_information("t6", persist_dir=small_index)
    for chunk in out["results"]:
        assert "phase_probabilities" not in chunk["content"]

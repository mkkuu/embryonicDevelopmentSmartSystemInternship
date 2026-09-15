"""Tests for the phase-vocabulary enrichment of the RAG query
(`Training/orchestrator/orchestrator.py`, 2026-09-07e).

Measured problem on the real index, reproduced across the four
`prompt_v2_reading_method` runs: the chunk spelling out the canonical phase
order is at rank 6 for Q11 and outside the top 25 for Q12, while `top_k` is
5. The corpus contains the ordering; the questions carry no phase vocabulary.

Two layers, deliberately:
  * the query-construction contract -- pure, no index, no embedder, no GPU;
  * one integration test on a small SYNTHETIC index (same fixture style as
    Tests/rag/test_rag_retrieval.py) proving the enrichment actually changes
    which chunk is retrieved. The production-index ranking itself is
    measured by `experiments.rag_llm_diagnostics.measure_retrieval_rank`, not asserted here:
    RagIndex/ is gitignored and absent from a fresh checkout.
"""

import pytest

from orchestrator import orchestrator as orch
from orchestrator.context_builder import SOURCE_DOCUMENT, SOURCE_REPORTING_API, ToolCallResult
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.router import _normalize

QUESTION = {q.question_id: q.question for q in FIXED_QUESTIONS}


def _results(**values):
    return {name: ToolCallResult(tool_name=name, source_type=SOURCE_REPORTING_API, ok=True,
                                 value=value) for name, value in values.items()}


_INFERENCE = {"current_state": {"current_phase": "t7",
                                "phase_probabilities": {"t7": 0.76, "t8": 0.24, "t5": 0.0017}},
              "next_phase": {"most_likely_next_phase": "t7"}}
_EVENTS = {"transitions": [{"from_phase": "t4", "to_phase": "t6", "is_skip": True}]}


# --- which questions are enriched (test 7: the others must not change) ------

def test_only_the_order_and_reference_questions_are_enriched():
    results = _results(get_current_inference=_INFERENCE, get_transition_events=_EVENTS)
    enriched, untouched = [], []
    for q in FIXED_QUESTIONS:
        built = orch.build_retrieval_query(q.question, results)
        (enriched if built != q.question else untouched).append(q.question_id)
    assert enriched == ["Q11", "Q12"]
    # every other question that reaches the RAG tool keeps a byte-identical query
    for qid in ("Q7", "Q10", "Q13", "Q14", "Q15"):
        assert qid in untouched
        assert orch.build_retrieval_query(QUESTION[qid], results) == QUESTION[qid]


def test_the_triggers_match_nothing_outside_q11_q12():
    """Collision guard, checked against the frozen question set itself rather
    than against a hand-written list."""
    for q in FIXED_QUESTIONS:
        fires = any(t in _normalize(q.question)
                    for t in orch._PHASE_VOCABULARY_QUERY_TRIGGERS)
        assert fires == (q.question_id in {"Q11", "Q12"}), q.question_id


# --- what the enriched query contains (test 5: nothing is invented) ---------

def test_the_query_is_the_question_plus_context_phases_only():
    results = _results(get_current_inference=_INFERENCE, get_transition_events=_EVENTS)
    built = orch.build_retrieval_query(QUESTION["Q11"], results)
    assert built.startswith(QUESTION["Q11"]), "the original question must be preserved verbatim"
    added = built[len(QUESTION["Q11"]):].split()
    assert added == ["t7", "t8", "t5", "t4", "t6"]


def test_no_phase_is_invented_every_added_token_comes_from_the_payload():
    """The added vocabulary is a SUBSET of what the turn's own tool outputs
    contain -- a distribution with three keys contributes three names, never
    the project's full 15-phase taxonomy."""
    small = {"current_state": {"current_phase": "t2", "phase_probabilities": {"t2": 0.9, "t3": 0.1}}}
    built = orch.build_retrieval_query(QUESTION["Q12"], _results(get_current_inference=small))
    added = built[len(QUESTION["Q12"]):].split()
    assert added == ["t2", "t3"]
    for absent in ("tpb2", "tpna", "t7", "teb"):
        assert absent not in added


def test_vocabulary_is_deduplicated_and_deterministic():
    results = _results(a=_INFERENCE, b=_INFERENCE, c=_EVENTS)
    tokens = orch._context_phase_vocabulary(results)
    assert tokens == ["t7", "t8", "t5", "t4", "t6"]
    assert len(tokens) == len(set(tokens))
    assert orch._context_phase_vocabulary(results) == tokens


def test_non_phase_dictionaries_never_contribute_keys():
    """Only a dict whose keys are ALL phase tokens is read as a vocabulary;
    anything else is walked normally."""
    results = _results(x={"metrics": {"brier": 0.1, "t7": 0.9}, "labels": {"phase": "t6"}})
    assert orch._context_phase_vocabulary(results) == ["t6"]


# --- degradation (test 6: still functional with nothing to add) -------------

def test_question_without_any_available_phase_is_left_unchanged():
    assert orch.build_retrieval_query(QUESTION["Q11"], {}) == QUESTION["Q11"]
    assert orch.build_retrieval_query(QUESTION["Q12"], {}) == QUESTION["Q12"]


def test_failed_tool_results_contribute_nothing():
    failed = {"get_current_inference": ToolCallResult(
        tool_name="get_current_inference", source_type=SOURCE_REPORTING_API, ok=False,
        error="Reporting API unavailable")}
    assert orch.build_retrieval_query(QUESTION["Q11"], failed) == QUESTION["Q11"]


def test_a_documentary_question_with_no_live_data_is_unchanged():
    assert orch.build_retrieval_query("Qu'est-ce qu'un Semi-HMM ?", {}) == "Qu'est-ce qu'un Semi-HMM ?"


# --- the retrieval call itself (tests 3 and 4) ------------------------------

def test_execute_plan_passes_the_enriched_query_and_never_changes_top_k(monkeypatch):
    from orchestrator import tools

    seen = {}

    def fake_retrieve_documents(query, top_k=5, **kwargs):
        seen["query"] = query
        seen["top_k"] = top_k
        seen["kwargs"] = kwargs
        return [{"chunk_id": "c1", "content": "ordre canonique", "score": 0.9,
                 "metadata": {"source": "docs/X.md", "section": "2", "status": "current",
                              "authoritative": True}}]

    monkeypatch.setattr(tools, "get_current_inference", lambda *a, **k: _INFERENCE)
    monkeypatch.setattr(tools, "get_transition_events", lambda *a, **k: _EVENTS)
    monkeypatch.setattr(tools, "retrieve_documents", fake_retrieve_documents)

    question = QUESTION["Q11"]
    plan = orch.plan_tools(question, orch.classify(question), video_id="Patient_319", window=156)
    results = orch.execute_plan(plan, question, "Patient_319", 156, "val")

    assert seen["query"].startswith(question) and seen["query"] != question
    assert seen["top_k"] == 5, "top_k must stay at its documented default"
    assert seen["kwargs"] == {}, "no filter or top_k override may be introduced"
    assert results["retrieve_documents"].ok


def test_no_document_is_appended_to_the_context_after_retrieval(monkeypatch):
    """The intervention is a QUERY change: what the retriever returns is what
    reaches the context, unmodified and un-augmented."""
    from orchestrator import tools

    returned = [{"chunk_id": "only", "content": "un seul chunk", "score": 0.5,
                 "metadata": {"source": "docs/Y.md", "section": "1", "status": "current",
                              "authoritative": False}}]
    monkeypatch.setattr(tools, "get_current_inference", lambda *a, **k: _INFERENCE)
    monkeypatch.setattr(tools, "get_transition_events", lambda *a, **k: _EVENTS)
    monkeypatch.setattr(tools, "retrieve_documents", lambda query, **k: returned)

    question = QUESTION["Q12"]
    plan = orch.plan_tools(question, orch.classify(question), video_id="Patient_319", window=156)
    results = orch.execute_plan(plan, question, "Patient_319", 156, "val")
    assert results["retrieve_documents"].value == returned
    assert results["retrieve_documents"].source_type == SOURCE_DOCUMENT


# --- integration on a small synthetic index (tests 1 and 2, mechanism) ------

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from rag import retrieval, vectorstore  # noqa: E402
from rag.chunker import build_chunks  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402


@pytest.fixture
def order_index(tmp_path):
    """A synthetic index reproducing the real failure shape: several chunks of
    project-strategy prose that a bare "ordre attendu / prochaine etape"
    question is closest to, plus ONE chunk that actually enumerates the phase
    taxonomy but shares little wording with the question."""
    docs = [
        dict(document_id="taxonomy", document_type="project", source="docs/DATASET.md",
             title="Dataset", status="current", authoritative=True, version="v1",
             text="# Dataset\n\n## 2. Dataset\n\nSequences d'images, 15 phases developpementales "
                  "chronologiques (`tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, "
                  "tB, tEB`). Les splits sont filtres par coherence de fenetre.\n"),
        dict(document_id="strategy", document_type="project", source="docs/STRATEGY.md",
             title="Publication Strategy", status="current", authoritative=False, version="v1",
             text="# Publication Strategy\n\n## Prochaine etape\n\nLa prochaine etape attendue du "
                  "projet suit l'ordre attendu du programme de publication et du developpement "
                  "logiciel, selon le referentiel methodologique retenu.\n"),
        dict(document_id="roadmap", document_type="project", source="docs/ROADMAP.md",
             title="Roadmap", status="current", authoritative=False, version="v1",
             text="# Roadmap\n\n## Etapes\n\nL'ordre attendu des etapes de developpement du "
                  "produit est decrit ici, avec la prochaine etape developpementale du plan "
                  "d'implementation.\n"),
    ]
    collection = vectorstore.get_collection(persist_dir=tmp_path)
    for d in docs:
        chunks = build_chunks(d["document_id"], d["document_type"], d["source"], d["title"],
                              d["status"], d["authoritative"], d["version"], d["text"])
        vectorstore.upsert_chunks(collection, chunks, embed_texts([c.content for c in chunks]))
    return tmp_path


def _taxonomy_rank(query, index, top_k=3):
    for i, r in enumerate(retrieval.retrieve(query, top_k=top_k, persist_dir=index), 1):
        if "tPB2" in (r.get("content") or ""):
            return i
    return None


@pytest.mark.parametrize("question_id", ["Q11", "Q12"])
def test_enrichment_lifts_the_taxonomy_chunk_on_a_synthetic_index(order_index, question_id):
    """Mechanism proof: the same index, the same top_k, only the query changes."""
    question = QUESTION[question_id]
    results = _results(get_current_inference=_INFERENCE, get_transition_events=_EVENTS)
    enriched = orch.build_retrieval_query(question, results)
    assert enriched != question

    before = _taxonomy_rank(question, order_index)
    after = _taxonomy_rank(enriched, order_index)
    assert after is not None, "the taxonomy chunk must be retrieved once the query carries phases"
    assert before is None or after < before, f"rank did not improve: {before} -> {after}"


def test_the_index_is_never_written_to_by_a_retrieval(order_index):
    """Test 8, at the mechanism level: retrieval is read-only -- the same query
    twice returns the same chunk ids and the collection size is unchanged."""
    collection = vectorstore.get_collection(persist_dir=order_index)
    before = collection.count()
    query = orch.build_retrieval_query(
        QUESTION["Q11"], _results(get_current_inference=_INFERENCE))
    first = [r["chunk_id"] for r in retrieval.retrieve(query, top_k=3, persist_dir=order_index)]
    second = [r["chunk_id"] for r in retrieval.retrieve(query, top_k=3, persist_dir=order_index)]
    assert first == second
    assert vectorstore.get_collection(persist_dir=order_index).count() == before

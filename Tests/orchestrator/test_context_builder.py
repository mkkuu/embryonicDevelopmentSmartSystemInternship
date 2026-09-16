"""
Pure tests for context_builder.build_context() -- uses hand-built
ToolCallResult objects, never a real Reporting API/RAG call, so these
always run without torch/chromadb. Verifies the one property this module
exists to guarantee: document_context and dynamic_context are physically
separate (docs/LLM_ORCHESTRATION.md sec 3's FACT/DOCUMENTED separation),
provenance is recorded per piece of information, and a failed tool call
becomes a warning, never a silent gap.
"""

from orchestrator.context_builder import SOURCE_DOCUMENT, SOURCE_REPORTING_API, ToolCallResult, build_context
from orchestrator.router import classify


def _route():
    return classify("Pourquoi le modele predit-il t6 pour cette fenetre ?")


def test_successful_document_tool_populates_document_context_only():
    results = {
        "retrieve_documents": ToolCallResult(
            tool_name="retrieve_documents", source_type=SOURCE_DOCUMENT, ok=True,
            value=[{"chunk_id": "a::0001", "content": "text", "score": 0.8,
                    "metadata": {"source": "docs/A.md", "section": "1", "status": "authoritative",
                                 "authoritative": True}}],
        ),
    }
    ctx = build_context("q", _route(), results)
    assert len(ctx["document_context"]) == 1
    assert ctx["document_context"][0]["source"] == "docs/A.md"
    assert ctx["dynamic_context"] == {}


def test_successful_reporting_tool_populates_dynamic_context_only():
    results = {
        "get_current_inference": ToolCallResult(
            tool_name="get_current_inference", source_type=SOURCE_REPORTING_API, ok=True,
            value={"current_state": {"current_phase": "t6"}, "model": {"model_version": "v1"}},
        ),
    }
    ctx = build_context("q", _route(), results)
    assert ctx["document_context"] == []
    assert ctx["dynamic_context"]["get_current_inference"]["current_state"]["current_phase"] == "t6"


def test_document_and_dynamic_context_never_mixed_in_one_block():
    results = {
        "retrieve_documents": ToolCallResult(
            tool_name="retrieve_documents", source_type=SOURCE_DOCUMENT, ok=True,
            value=[{"chunk_id": "a::0001", "content": "text", "score": 0.8,
                    "metadata": {"source": "docs/A.md", "section": "1", "status": "authoritative",
                                 "authoritative": True}}],
        ),
        "get_current_inference": ToolCallResult(
            tool_name="get_current_inference", source_type=SOURCE_REPORTING_API, ok=True,
            value={"current_state": {"current_phase": "t6"}, "model": {"model_version": "v1"}},
        ),
    }
    ctx = build_context("q", _route(), results)
    assert len(ctx["document_context"]) == 1
    assert "get_current_inference" in ctx["dynamic_context"]
    # no document leaked into dynamic_context and vice versa
    assert "current_state" not in ctx["document_context"][0]
    assert not any("chunk_id" in v for v in ctx["dynamic_context"].values() if isinstance(v, dict))


def test_failed_tool_call_becomes_a_warning_not_a_silent_gap():
    results = {
        "get_current_inference": ToolCallResult(
            tool_name="get_current_inference", source_type=SOURCE_REPORTING_API, ok=False,
            error="model unavailable", error_type="ToolUnavailableError",
        ),
    }
    ctx = build_context("q", _route(), results)
    assert ctx["dynamic_context"] == {}
    assert any("model unavailable" in w for w in ctx["warnings"])
    assert any(p["status"] == "error" for p in ctx["provenance"])


def test_get_phase_information_warnings_propagate_into_context_warnings():
    results = {
        "get_phase_information": ToolCallResult(
            tool_name="get_phase_information", source_type=SOURCE_DOCUMENT, ok=True,
            value={"phase_id": "t6", "results": [], "warnings": ["no dedicated phase index"]},
        ),
    }
    ctx = build_context("q", _route(), results)
    assert "no dedicated phase index" in ctx["warnings"]


def test_provenance_records_one_entry_per_document_chunk():
    results = {
        "retrieve_documents": ToolCallResult(
            tool_name="retrieve_documents", source_type=SOURCE_DOCUMENT, ok=True,
            value=[
                {"chunk_id": "a::0001", "content": "t1", "score": 0.8,
                 "metadata": {"source": "docs/A.md", "section": "1", "status": "authoritative", "authoritative": True}},
                {"chunk_id": "a::0002", "content": "t2", "score": 0.7,
                 "metadata": {"source": "docs/A.md", "section": "2", "status": "authoritative", "authoritative": True}},
            ],
        ),
    }
    ctx = build_context("q", _route(), results)
    assert len(ctx["provenance"]) == 2


def test_context_always_carries_the_route_category():
    ctx = build_context("q", _route(), {})
    assert ctx["route"]["category"] == "HYBRID"
    assert ctx["question"] == "q"

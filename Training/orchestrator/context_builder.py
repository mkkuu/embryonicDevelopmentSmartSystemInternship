"""
Assembles heterogeneous tool outputs (RAG chunks, Reporting API records,
tool errors) into ONE normalized, provenance-tagged structure -- task's
own "Context Builder" (Phase 10):

{
  "question": str,
  "route": {...},              # router.RouteDecision.to_dict()
  "document_context": [...],   # RAG chunks, each source_type="document"
  "dynamic_context": {...},    # Reporting API tool outputs, keyed by tool
                                # name, each carrying source_type="reporting_api"
  "model_metadata": {...} | None,
  "warnings": [...],           # collected from every tool call + this builder
  "provenance": [...],         # one entry per piece of information actually used
  "tools_called": [...],       # tool names actually executed this turn (Phase 6:
                                # lets an LLMProvider/prompt builder report used_tools
                                # without re-deriving it from provenance)
}

Never merges document_context and dynamic_context into one
undifferentiated blob -- docs/LLM_ORCHESTRATION.md sec 3's "FACT/
DOCUMENTED blocks kept physically separate" requirement, enforced here
structurally (two distinct top-level keys, populated by two disjoint
tool-name -> source_type mappings), not by prompt-wording convention
alone. This is the mechanism, not the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .router import RouteDecision

SOURCE_DOCUMENT = "document"
SOURCE_REPORTING_API = "reporting_api"
# compact_v2 (2026-09-12): two more source types, each landing in its OWN
# context key so neither is ever blended with the RAG chunks or the live data.
SOURCE_SCIENTIFIC = "scientific"   # Istanbul Consensus blocks -> context["scientific_context"]
SOURCE_AUDIT = "audit"             # retrieval selection decisions -> context["retrieval_audit"]


@dataclass
class ToolCallResult:
    """One tool invocation's outcome -- exactly one of `value` (ok=True)
    or `error`/`error_type` (ok=False) is populated, never both, never
    neither (a tool call that produced nothing must say so explicitly,
    not be represented as an empty success)."""

    tool_name: str
    source_type: str  # SOURCE_DOCUMENT | SOURCE_REPORTING_API
    ok: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name, "source_type": self.source_type, "ok": self.ok,
            "value": self.value, "error": self.error, "error_type": self.error_type,
        }


def build_context(
    question: str, route: RouteDecision, tool_results: Dict[str, ToolCallResult],
    tools_called: Optional[List[str]] = None,
) -> Dict:
    document_context: List[Dict] = []
    dynamic_context: Dict[str, Any] = {}
    model_metadata: Optional[Dict] = None
    warnings: List[str] = []
    provenance: List[Dict] = []
    scientific_context: Optional[Dict] = None
    retrieval_audit: Optional[List[Dict]] = None

    for name, result in tool_results.items():
        if not result.ok:
            warnings.append(f"{name} failed ({result.error_type}): {result.error}")
            provenance.append({
                "tool": name, "source_type": result.source_type, "status": "error",
            })
            continue

        if result.source_type == SOURCE_DOCUMENT:
            # Either a raw retrieve_documents() list of chunks, or
            # get_phase_information()'s {"phase_id", "results", "warnings"} dict.
            if isinstance(result.value, dict):
                chunks = result.value.get("results", [])
                warnings.extend(result.value.get("warnings", []))
            else:
                chunks = result.value or []
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                document_context.append({
                    "tool": name,
                    "source_type": SOURCE_DOCUMENT,
                    "content": chunk.get("content"),
                    "source": meta.get("source"),
                    "section": meta.get("section"),
                    "status": meta.get("status"),
                    "authoritative": meta.get("authoritative"),
                    "score": chunk.get("score"),
                })
                provenance.append({
                    "tool": name, "source_type": SOURCE_DOCUMENT,
                    "source": meta.get("source"), "section": meta.get("section"),
                    "status": meta.get("status"),
                })

        elif result.source_type == SOURCE_REPORTING_API:
            dynamic_context[name] = result.value
            model_version = None
            if isinstance(result.value, dict):
                model_block = result.value.get("model")
                if isinstance(model_block, dict):
                    model_version = model_block.get("model_version")
                elif "model_version" in result.value:
                    model_version = result.value.get("model_version")
            provenance.append({
                "tool": name, "source_type": SOURCE_REPORTING_API,
                "model_version": model_version,
            })
            if name == "get_model_metadata":
                model_metadata = result.value

        elif result.source_type == SOURCE_SCIENTIFIC:
            scientific_context = result.value
            warnings.extend((result.value or {}).get("warnings", []) or [])
            provenance.append({
                "tool": name, "source_type": SOURCE_SCIENTIFIC,
                "source": (result.value or {}).get("source"),
                "block_ids": [b.get("block_id") for b in (result.value or {}).get("blocks", [])],
            })

        elif result.source_type == SOURCE_AUDIT:
            retrieval_audit = result.value
            provenance.append({"tool": name, "source_type": SOURCE_AUDIT})

        else:
            warnings.append(f"{name}: unrecognized source_type {result.source_type!r}, dropped.")

    return {
        "question": question,
        "route": route.to_dict(),
        "document_context": document_context,
        "dynamic_context": dynamic_context,
        "model_metadata": model_metadata,
        "warnings": warnings,
        "provenance": provenance,
        "tools_called": list(tools_called or []),
        # Present only when a compact_v2 turn produced them (None otherwise), so
        # every historical context dict keeps its exact previous key set VALUES.
        "scientific_context": scientific_context,
        "retrieval_audit": retrieval_audit,
    }

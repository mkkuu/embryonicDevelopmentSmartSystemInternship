"""
Typed wrappers around the Reporting API (Training/reporting/) and the RAG
retrieval prototype (Training/rag/) -- the ONLY data path an
orchestrator/LLM is ever allowed to use for a scientific model output or
a documentary answer (docs/PRODUCT_ARCHITECTURE.md sec 4: "the LLM never
gets a second, parallel way to obtain a number"). No function here
reimplements anything -- every call delegates to the already-tested
reporting/rag modules and returns their output, unmodified, as a plain
dict/list.

Full contracts (name / description / input schema / output schema /
source of truth / errors / freshness / determinism / side effects):
docs/TOOL_CONTRACTS.md. This module is the executable half of that
document -- keep them in sync; do not let the doc and the code diverge.

Deferred imports: `from reporting import ...` needs torch (transitively,
via Training/embeddings/dataset.py); `from rag import retrieval` succeeds
without chromadb/sentence-transformers (those are only imported inside
vectorstore.get_collection()/embeddings._get_model()), but the actual
retrieval call still needs them. Neither dependency is guaranteed present
in every environment that might import this module (e.g. this repo's own
bare local checkout, per CLAUDE.md) -- every function below defers its
import to call time and raises ToolUnavailableError explicitly rather
than failing this module's own import, or silently returning nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))


class ToolError(Exception):
    """Base class for every error a tool call in this module can raise --
    always one of the typed subclasses below, never a bare Exception the
    caller has to guess about (this project's "explicit fallback, never a
    silent no-op" discipline, already used throughout
    Training/evaluation/models/*.py, applied here to the tool layer)."""


class ToolUnavailableError(ToolError):
    """The underlying package (torch, chromadb, sentence_transformers) or
    artifact (frozen model checkpoint, embeddings cache, vector index) is
    not present/loadable in this environment. Distinct from a bad request
    (ToolInputError) or a genuinely missing resource that DOES exist in
    principle but not for this key (ToolNotFoundError)."""


class ToolInputError(ToolError):
    """The caller passed an invalid parameter -- e.g. split='test'
    (rejected here too, a third redundant guard on top of the two already
    in Training/reporting/ itself -- defense in depth, matching
    docs/LLM_ORCHESTRATION.md sec 2's own "every tool validates split
    too" rule), or a malformed value."""


class ToolNotFoundError(ToolError):
    """The requested resource does not exist: unknown video_id, or a
    window_start not present in that video's trajectory."""


# ---------------------------------------------------------------------------
# get_current_inference
# ---------------------------------------------------------------------------

def get_current_inference(video_id: str, window: int, split: str = "val") -> Dict:
    """Tool: get_current_inference(video_id, window). Wraps
    reporting.inference_service.infer() -- the Reporting API's own
    causal, per-window inference call (docs/REPORTING_API.md,
    docs/INFERENCE_SCHEMA.md). Source of truth: the frozen Semi-HMM via
    the Reporting API. Read-only; the underlying forward pass is disk
    cached (Training/reporting/cache.py) but this function itself never
    writes anything the caller needs to know about. Deterministic given
    (video_id, window, split) and the frozen model version."""
    try:
        from reporting import inference_service, model_loader
        from reporting.trajectory_service import TestSplitLockedError
    except ImportError as e:
        raise ToolUnavailableError(
            f"get_current_inference: Reporting API dependency unavailable ({e}). "
            f"Requires the embryo_env conda environment (torch) and the real "
            f"frozen model/embeddings artifacts -- see CLAUDE.md."
        ) from e
    try:
        record = inference_service.infer(video_id, split, window)
    except TestSplitLockedError as e:
        raise ToolInputError(str(e)) from e
    except model_loader.ModelIncompatibleError as e:
        raise ToolUnavailableError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolUnavailableError(str(e)) from e
    except ValueError as e:
        raise ToolInputError(str(e)) from e
    except KeyError as e:
        raise ToolNotFoundError(str(e)) from e
    return record.to_dict()


# ---------------------------------------------------------------------------
# get_trajectory
# ---------------------------------------------------------------------------

def get_trajectory(video_id: str, split: str = "val") -> Dict:
    """Tool: get_trajectory(video_id). Wraps
    reporting.inference_service.trajectory_summary() -- RETROSPECTIVE
    (needs the full trajectory; includes the segment-level
    transition_chain, docs/INFERENCE_SCHEMA.md). Never use this for a
    live/in-progress framing -- that is get_current_inference's job.
    Source of truth: the frozen Semi-HMM via the Reporting API."""
    try:
        from reporting import inference_service, model_loader
        from reporting.trajectory_service import TestSplitLockedError
    except ImportError as e:
        raise ToolUnavailableError(
            f"get_trajectory: Reporting API dependency unavailable ({e})."
        ) from e
    try:
        summary = inference_service.trajectory_summary(video_id, split)
    except TestSplitLockedError as e:
        raise ToolInputError(str(e)) from e
    except model_loader.ModelIncompatibleError as e:
        raise ToolUnavailableError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolUnavailableError(str(e)) from e
    except ValueError as e:
        raise ToolInputError(str(e)) from e
    except KeyError as e:
        raise ToolNotFoundError(str(e)) from e
    return summary.to_dict()


# ---------------------------------------------------------------------------
# get_transition_events
# ---------------------------------------------------------------------------

def get_transition_events(video_id: str, split: str = "val", window: Optional[int] = None) -> Dict:
    """Tool: get_transition_events(video_id, split, window=None). OBSERVED
    phase transitions ONLY -- wraps reporting.transition_events.transition_events(),
    which derives from ground-truth annotation segments
    (HMMModel._segments(), reused, never reimplemented, never a model
    prediction). Every returned transition carries observed=True,
    provenance="observed_annotation" -- there is no model-derived field
    anywhere in this tool's output; a caller wanting the model's own view
    around the same window range calls get_current_inference separately
    (kept as two distinct tool calls by design,
    docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md sec 3/7).

    Direct cleavage / reversal / fragmentation / multinucleation are
    NEVER computed here -- this tool structurally cannot return them (no
    such field exists on TransitionEvent at all), rather than an empty/
    null placeholder that could be misread as "checked, found none" for
    those specific concepts (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 3).

    When `window` is given and no transition brackets it, returns an
    empty list -- a legitimate, honest answer ("no transition observed
    around this window"), never an error."""
    try:
        from reporting import model_loader, transition_events as transition_events_module
        from reporting.trajectory_service import TestSplitLockedError
    except ImportError as e:
        raise ToolUnavailableError(
            f"get_transition_events: Reporting API dependency unavailable ({e})."
        ) from e
    try:
        events = transition_events_module.transition_events(video_id, split, window)
    except TestSplitLockedError as e:
        raise ToolInputError(str(e)) from e
    except model_loader.ModelIncompatibleError as e:
        raise ToolUnavailableError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolUnavailableError(str(e)) from e
    except ValueError as e:
        raise ToolInputError(str(e)) from e
    except KeyError as e:
        raise ToolNotFoundError(str(e)) from e
    return {
        "video_id": video_id, "split": split, "window": window,
        "n_transitions": len(events),
        "transitions": [e.to_dict() for e in events],
    }


# ---------------------------------------------------------------------------
# get_inference_history  (multi-window / G1)
# ---------------------------------------------------------------------------

def get_inference_history(
    video_id: str, center_window: int, before: int = 5, after: int = 5, split: str = "val",
    top_k_probabilities: Optional[int] = None,
) -> Dict:
    """Tool: get_inference_history(video_id, center_window, before, after).
    Wraps reporting.inference_service.infer_history() -- the MULTI-WINDOW
    capability get_current_inference (single window) could not provide.
    Returns, for each real window in [center-before, center+after] of THIS
    video, temporally ordered: window_start / times, and a `model_derived`
    block (current_phase, full phase_probabilities, entropy,
    next_phase_distribution) kept PHYSICALLY SEPARATE from an `observed`
    block (ground_truth_phase, consistency_flag) -- annotation and model
    prediction never merged.

    Source of truth: the frozen Semi-HMM via the Reporting API. Never
    recomputes -- each window is a slice of the same InferenceRecord
    get_current_inference already produces, and the forward pass is cached
    once per (video, split). Clamps `before`/`after` to the video's real
    trajectory bounds (warning added), never spills into another video,
    never fabricates a window. Deterministic given (video_id,
    center_window, before, after, split) and the frozen model version.

    top_k_probabilities (ETAPE 4, context compactness): keep only the k
    highest-probability entries of each per-window distribution, verbatim.
    None (default) = full 15-entry distributions, unchanged. entropy /
    phase_probability / next_phase_probability are computed upstream from
    the FULL distribution and are never affected; a warning discloses the
    truncation and the largest omitted probability mass."""
    try:
        from reporting import inference_service, model_loader
        from reporting.inference_service import WindowNotFoundError
        from reporting.trajectory_service import TestSplitLockedError
    except ImportError as e:
        raise ToolUnavailableError(
            f"get_inference_history: Reporting API dependency unavailable ({e}). "
            f"Requires the embryo_env conda environment (torch) and the real frozen "
            f"model/embeddings artifacts -- see CLAUDE.md."
        ) from e
    try:
        record = inference_service.infer_history(video_id, split, center_window, before, after,
                                                 top_k_probabilities=top_k_probabilities)
    except TestSplitLockedError as e:
        raise ToolInputError(str(e)) from e
    except model_loader.ModelIncompatibleError as e:
        raise ToolUnavailableError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolUnavailableError(str(e)) from e
    except WindowNotFoundError as e:
        raise ToolNotFoundError(str(e)) from e
    except ValueError as e:
        raise ToolInputError(str(e)) from e
    except KeyError as e:
        raise ToolNotFoundError(str(e)) from e
    return record.to_dict()


# ---------------------------------------------------------------------------
# get_model_metadata
# ---------------------------------------------------------------------------

def get_model_metadata() -> Dict:
    """Tool: get_model_metadata(). Wraps reporting.model_loader --
    identical shape to `GET /model` (docs/REPORTING_API.md). No
    parameters, no side effects, deterministic for the lifetime of the
    serving process (the frozen model is loaded once and cached; a
    version change requires restarting the process, never a live
    swap)."""
    try:
        from reporting import model_loader
    except ImportError as e:
        raise ToolUnavailableError(
            f"get_model_metadata: Reporting API dependency unavailable ({e})."
        ) from e
    try:
        model = model_loader.get_model()
    except model_loader.ModelIncompatibleError as e:
        raise ToolUnavailableError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolUnavailableError(str(e)) from e
    return {
        "model_name": "semi_hmm",
        "model_version": model_loader.model_version_string(model),
        "model_configuration": model_loader.model_configuration(model),
        "model_source": str(model_loader.REFERENCE_MODEL_PATH),
    }


# ---------------------------------------------------------------------------
# retrieve_documents
# ---------------------------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
    status: Optional[str] = None,
    authoritative_only: bool = False,
    persist_dir: Optional[Path] = None,
) -> List[Dict]:
    """Tool: retrieve_documents(query, ...). Wraps rag.retrieval.retrieve()
    -- semantic search over the static documentary corpus (Chroma,
    docs/RAG_ARCHITECTURE.md). Source of truth: the ingested docs/*.md
    corpus (RagIndex/), never live model output (structurally -- see
    Training/rag/inventory.py's docstring). Read-only. NOT deterministic
    in the strict bit-for-bit sense across different corpus states (a
    re-ingestion changes results), but deterministic for a fixed
    RagIndex/ snapshot and a fixed query."""
    try:
        from rag import retrieval

        results = retrieval.retrieve(
            query, top_k=top_k, document_type=document_type, status=status,
            authoritative_only=authoritative_only, persist_dir=persist_dir,
        )
    except ImportError as e:
        raise ToolUnavailableError(
            f"retrieve_documents: RAG dependency unavailable ({e}). "
            f"Requires chromadb + sentence-transformers."
        ) from e
    except ValueError as e:
        raise ToolInputError(str(e)) from e
    return results


# ---------------------------------------------------------------------------
# get_phase_information
# ---------------------------------------------------------------------------

def get_phase_information(phase_id: str, top_k: int = 3, persist_dir: Optional[Path] = None) -> Dict:
    """Tool: get_phase_information(phase_id). NOT a Reporting API
    endpoint: phase definitions/methodology are static documentary
    knowledge, RAG's domain (docs/RAG_ARCHITECTURE.md sec 1), never a
    live lookup. Implemented as a constrained retrieve_documents() call,
    NOT a dedicated phase-metadata index, because
    Training/rag/schemas.py's Chunk carries no `phase` metadata field
    today -- a real, already-documented content gap
    (docs/RAG_ARCHITECTURE.md sec 1: "no phase-level biological glossary
    exists yet anywhere in this repo"), surfaced here as an explicit
    warning rather than hidden behind a best-effort text match that looks
    more authoritative than it is."""
    query = f"phase {phase_id} developpement embryonnaire definition"
    results = retrieve_documents(query, top_k=top_k, persist_dir=persist_dir)
    return {
        "phase_id": phase_id,
        "results": results,
        "warnings": [
            "get_phase_information has no dedicated phase-metadata index -- this is a "
            "best-effort text search over the general documentation corpus, not an exact "
            "per-phase glossary lookup (docs/RAG_ARCHITECTURE.md sec 1, known content gap)."
        ],
    }

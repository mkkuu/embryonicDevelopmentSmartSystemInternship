"""
Curated source inventory for the RAG corpus (Product Roadmap Phase 3 /
"Phase 0-2" of the ingestion-foundation task).

This is a DELIBERATE, HAND-REVIEWED list, not a directory scan. The task
this module implements is explicit that the RAG "must not blindly ingest
the whole repository" -- every docs/*.md file was inspected (title, first
lines, mtime, cross-references) this session before being included or
excluded below, and every exclusion carries a stated reason so the
decision is auditable, not silent.

Scope of this pass: `docs/*.md` only. `Results/evaluation/*/REPORT.md`
files (the "EXPERIMENTAL KNOWLEDGE" layer per docs/RAG_DATA_MODEL.md) are
a real, legitimate future source -- already synthesized into
docs/SCIENTIFIC_REPORT.md and docs/GLOBAL_MODEL_COMPARISON.md for this
pass -- but pulling ~24 remote, gitignored files into the corpus is its
own scoped follow-up, not attempted here (see docs/RAG_PHASE_3_REPORT.md
§16 Next step).

Never lists anything under Results/evaluation/*/analysis.json,
Embeddings/, Cache/, or any per-window/per-video artifact -- see
Training/rag/__init__.py's module docstring and docs/RAG_ARCHITECTURE.md's
"dynamic data never in the vector DB" rule. This is the structural
guarantee that Phase 10's data/RAG separation holds: the RAG literally
has no code path capable of reading a prediction file, the same
discipline `Training/reporting/trajectory_service.py` uses for the Test
split (a hard-coded allowlist, not a runtime filter trusted to catch
everything).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DOCS_ROOT = Path(__file__).resolve().parent.parent.parent / "docs"


@dataclass
class InventoryEntry:
    source: str                # "docs/FOO.md", relative to repo root
    document_type: str          # scientific | project | methodological
    status: str                  # authoritative | current | historical | experimental
    authoritative: bool
    supersedes: Optional[str] = None
    reason: str = ""              # why this classification -- always filled in


# ---------------------------------------------------------------------------
# INCLUDED -- 30 documents, hand-classified.
# ---------------------------------------------------------------------------
INCLUDED: List[InventoryEntry] = [
    InventoryEntry("docs/HANDOFF.md", "scientific", "authoritative", True,
                    reason="Stable, unchanging mathematical formulation (hidden state, dynamics, observation model) -- the project's own root reference."),
    InventoryEntry("docs/RESEARCH_BLUEPRINT.md", "scientific", "authoritative", True,
                    reason="Crystallized hypothesis programme (RQ1/H1 etc.) research decisions are made against."),
    InventoryEntry("docs/BLUEPRINT_SUMMARY.md", "scientific", "current", False,
                    reason="Short daily-reference distillation of RESEARCH_BLUEPRINT.md, companion not replacement."),
    InventoryEntry("docs/HMM_RESEARCH_PLAN.md", "methodological", "current", False,
                    reason="Full HMM/Semi-HMM math/design -- plan fully executed but the design content remains accurate and citable."),
    InventoryEntry("docs/HMM_PHASE0_EXECUTION_PLAN.md", "methodological", "historical", False,
                    reason="Phase 0 execution plan, fully executed -- historical record of the concrete first-step reasoning."),
    InventoryEntry("docs/SCIENTIFIC_REPORT.md", "project", "authoritative", True, supersedes="scientific_results",
                    reason="Primary consolidated scientific reference, 2026-08-24 -- explicitly supersedes SCIENTIFIC_RESULTS.md as the main entry point."),
    InventoryEntry("docs/SCIENTIFIC_RESULTS.md", "project", "historical", False,
                    reason="Prior consolidation pass -- content still accurate per SCIENTIFIC_REPORT.md's own text ('not deleted, not contradicted'), but no longer the primary entry point."),
    InventoryEntry("docs/MODEL_COMPARISON.md", "methodological", "current", False,
                    reason="Per-model comparison table + decision narrative for every model actually built."),
    InventoryEntry("docs/GLOBAL_MODEL_COMPARISON.md", "project", "current", False,
                    reason="Full cross-architecture comparison including PR-AUC -- read-only synthesis of real results."),
    InventoryEntry("docs/REPRODUCIBILITY.md", "project", "current", False,
                    reason="How to reproduce/extend the SciML experiments."),
    InventoryEntry("docs/DATA_LINEAGE.md", "project", "current", False,
                    reason="Full pipeline RAW DATA -> RAG/Web App as it actually exists."),
    InventoryEntry("docs/INFERENCE_SCHEMA.md", "methodological", "current", False,
                    reason="Canonical inference-record shape, derived from real model outputs."),
    InventoryEntry("docs/REPORTING_API.md", "project", "authoritative", True,
                    reason="The Reporting API is implemented and this is its own documentation, written from the real code."),
    InventoryEntry("docs/REPORTING_API_IMPLEMENTATION_REPORT.md", "project", "historical", False,
                    reason="Build-time implementation report for a now-complete, stable phase."),
    InventoryEntry("docs/REPORTING_CACHE.md", "project", "authoritative", True,
                    reason="The Reporting Cache is implemented and this is its own documentation."),
    InventoryEntry("docs/REPORTING_CACHE_IMPLEMENTATION_REPORT.md", "project", "historical", False,
                    reason="Build-time implementation report for a now-complete, stable phase."),
    InventoryEntry("docs/SEMI_HMM_PHASE_1_6_REPORT.md", "methodological", "current", False,
                    reason="Real, measured optimization report for a specific Semi-HMM method, current and complete."),
    InventoryEntry("docs/GRU_IDENTITY_ANALYSIS.md", "project", "authoritative", True,
                    reason="Full, reviewed GRU vs Identity Dynamics analysis -- the authoritative reference for this comparison."),
    InventoryEntry("docs/GRU_IDENTITY_EXECUTIVE_SUMMARY.md", "project", "current", False,
                    reason="~1-page distillation of GRU_IDENTITY_ANALYSIS.md, companion not replacement."),
    InventoryEntry("docs/PRODUCT_ARCHITECTURE.md", "project", "authoritative", True,
                    reason="Living architecture reference, updated each phase against real verified code state."),
    InventoryEntry("docs/PRODUCT_ROADMAP.md", "project", "authoritative", True,
                    reason="The phase-by-phase plan this very task is executing against."),
    InventoryEntry("docs/RAG_ARCHITECTURE.md", "project", "authoritative", True,
                    reason="The design this ingestion pipeline itself implements."),
    InventoryEntry("docs/RAG_DATA_MODEL.md", "project", "current", False,
                    reason="What information a RAG needs and where it lives -- actively used as this pass's own design reference."),
    InventoryEntry("docs/LLM_ORCHESTRATION.md", "project", "experimental", False,
                    reason="Planning only, no code written, explicitly out of scope for this phase (no LLM implementation) -- still valid design content, tagged experimental so retrieval doesn't imply it's built."),
    InventoryEntry("docs/WEBAPP_ARCHITECTURE.md", "project", "experimental", False,
                    reason="Planning only, not implemented."),
    InventoryEntry("docs/WEBAPP_DATA_REQUIREMENTS.md", "project", "experimental", False,
                    reason="Requirements list only, not implemented."),
    InventoryEntry("docs/HANDOFF_SEMI_HMM.md", "methodological", "current", False,
                    reason="Stable Semi-HMM design/architecture reference, explicitly 'should not need updating every session'."),
    InventoryEntry("docs/HANDOFF_EMISSION_BALANCED.md", "project", "historical", False,
                    reason="Closed experiment branch (class-reweighting, Case D) -- real negative result, kept as history."),
    InventoryEntry("docs/HANDOFF_EMISSION_WEIGHTED.md", "project", "historical", False,
                    reason="Closed experiment branch (weighted sweep, alpha=0.25 closing run) -- real negative result, kept as history."),
    InventoryEntry("docs/PROJECT_TO_PRODUCT.md", "project", "historical", False,
                    reason="Transition-assessment snapshot (2026-08-24) -- narrative value retained even though PRODUCT_ARCHITECTURE.md is now the living reference it fed into."),
]

# ---------------------------------------------------------------------------
# EXCLUDED -- every docs/*.md file NOT in INCLUDED, with a stated reason.
# Never delete anything (docs/ files stay on disk exactly as they are) --
# this is purely "the RAG corpus does not ingest these," a retrieval-scope
# decision, not a repository-cleanup one (see
# docs/archive/sessions/CLEANUP_PLAN.md and
# docs/archive/sessions/REPOSITORY_INVENTORY.md for that separate question).
#
# SCOPE (2026-09-16): this dict lists ONLY paths matching `docs/*.md` -- the
# exact glob verify_inventory_matches_disk() compares against. The nine session
# logs and hygiene meta-documents previously listed here (SESSION_STATE,
# SPRINT0_EXECUTION_LOG, POINT_STAGE_2026-07-23, PROJECT_STATE, CLEANUP_PLAN,
# REPOSITORY_INVENTORY and the three PROJECT_CHECKPOINT* notes) were moved into
# docs/archive/sessions/ by the 2026-09-15 docs restructure. That put them
# outside this glob entirely, so they could never match again and were reported
# forever as `listed_but_missing`. They are still on disk, untouched, under
# docs/archive/sessions/, and still not ingested -- nothing changes for them:
# ingest.py only ever reads INCLUDED, never this dict.
#
# The six entries below are the successor guides written by that same
# restructure. They are classified here for the first time; INCLUDED is
# unchanged at 30 documents, so the ingested corpus and the vector-index
# identity are exactly what they were.
# ---------------------------------------------------------------------------
EXCLUDED = {
    "docs/README.md": "Navigation index for docs/ (which guide lives where) -- repository wayfinding, not scientific/project/methodological knowledge a user would query.",
    "docs/ARCHITECTURE.md": "2026-09-15 successor guide: code-layout overview distilled from PRODUCT_ARCHITECTURE.md, which is already ingested as authoritative -- ingesting both would duplicate the same content under two ranks.",
    "docs/SCIENTIFIC_BACKGROUND.md": "2026-09-15 successor guide: entry-level distillation of HANDOFF.md and RESEARCH_BLUEPRINT.md, both already ingested as authoritative.",
    "docs/REPRODUCIBILITY_GUIDE.md": "2026-09-15 successor guide: operational how-to (paths, hashes, snapshot locations) distilled from REPRODUCIBILITY.md, which is already ingested.",
    "docs/BENCHMARK.md": "2026-09-15 successor guide: how the benchmark is run and where its artefacts live -- procedure, not documentary knowledge; the measured results are in SCIENTIFIC_REPORT.md and GLOBAL_MODEL_COMPARISON.md, both already ingested.",
    "docs/DEVELOPMENT.md": "2026-09-15 successor guide: how to run the tests and the local servers -- developer procedure, same reasoning as the repository-hygiene meta-documents.",
}


def verify_inventory_matches_disk() -> dict:
    """Cross-check the hand-curated lists above against what's actually on
    disk in docs/ right now -- catches drift (a new doc added since this
    list was written, or a listed doc that no longer exists) rather than
    silently ingesting an unreviewed file or crashing on a missing one."""
    on_disk = {f"docs/{p.name}" for p in DOCS_ROOT.glob("*.md")}
    included_paths = {e.source for e in INCLUDED}
    excluded_paths = set(EXCLUDED.keys())
    accounted_for = included_paths | excluded_paths
    return {
        "on_disk_count": len(on_disk),
        "included_count": len(included_paths),
        "excluded_count": len(excluded_paths),
        "unaccounted_for": sorted(on_disk - accounted_for),   # exists on disk, not in either list -- needs review
        "listed_but_missing": sorted(accounted_for - on_disk),  # in a list, but not on disk -- stale entry
        "duplicate_in_both_lists": sorted(included_paths & excluded_paths),
    }

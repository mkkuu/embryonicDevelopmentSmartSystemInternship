"""
Tests for rag.inventory -- the curated docs/ source list. Skips cleanly
if docs/ isn't present on this machine (it's gitignored/local-only, see
CLAUDE.md -- not guaranteed to exist in every checkout, same discipline
Tests/reporting/test_integration_real_val.py already uses for
Results/Embeddings/).
"""

import pytest

from rag import inventory

pytestmark = pytest.mark.skipif(
    not inventory.DOCS_ROOT.exists(),
    reason="docs/ is gitignored/local-only (CLAUDE.md) -- not guaranteed present on this machine.",
)


def test_inventory_matches_disk_exactly():
    report = inventory.verify_inventory_matches_disk()
    assert report["unaccounted_for"] == [], (
        f"docs/*.md files present on disk but not classified in inventory.py: {report['unaccounted_for']}"
    )
    assert report["listed_but_missing"] == [], (
        f"inventory.py lists a path that no longer exists on disk: {report['listed_but_missing']}"
    )
    assert report["duplicate_in_both_lists"] == []


def test_every_included_entry_has_valid_document_type():
    from rag.schemas import DOCUMENT_TYPE_VALUES

    for entry in inventory.INCLUDED:
        assert entry.document_type in DOCUMENT_TYPE_VALUES, entry.source


def test_every_included_entry_has_valid_status():
    from rag.schemas import DOCUMENT_STATUS_VALUES

    for entry in inventory.INCLUDED:
        assert entry.status in DOCUMENT_STATUS_VALUES, entry.source


def test_every_included_entry_has_a_reason():
    for entry in inventory.INCLUDED:
        assert entry.reason, f"{entry.source} has no classification reason recorded"


def test_every_excluded_entry_has_a_reason():
    for source, reason in inventory.EXCLUDED.items():
        assert reason, f"{source} is excluded with no reason recorded"


def test_no_document_appears_twice_in_included():
    sources = [e.source for e in inventory.INCLUDED]
    assert len(sources) == len(set(sources))


def test_authoritative_entries_have_status_authoritative_or_current():
    # authoritative=True should never coexist with a "historical"/"experimental"
    # status -- an authoritative-but-stale document is a contradiction retrieval
    # ranking should never have to resolve silently.
    for entry in inventory.INCLUDED:
        if entry.authoritative:
            assert entry.status in ("authoritative", "current"), (
                f"{entry.source} is marked authoritative=True but status={entry.status!r}"
            )


def test_supersedes_reference_resolves_to_an_included_document():
    included_ids = set()
    from rag.schemas import build_document_id

    for entry in inventory.INCLUDED:
        included_ids.add(build_document_id(entry.source))
    for entry in inventory.INCLUDED:
        if entry.supersedes is not None:
            assert entry.supersedes in included_ids, (
                f"{entry.source} claims to supersede {entry.supersedes!r}, which is not an included document_id"
            )

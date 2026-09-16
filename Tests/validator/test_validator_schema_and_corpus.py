"""Schema invariants and the corpus reader."""

import pytest

from validator import corpus as corpus_module
from validator.schema import (Claim, COMBINED, MISSING_DATA, MODEL_DERIVED,
                              NOT_ASSESSABLE, SUPPORTED, ValidationReport)


def test_not_assessable_requires_a_reason_code():
    with pytest.raises(ValueError, match="reason_code"):
        Claim(claim="x", claim_kind="data_statement", provenance=MODEL_DERIVED,
              status=NOT_ASSESSABLE)


def test_not_assessable_with_a_reason_code_is_valid():
    claim = Claim(claim="x", claim_kind="data_statement", provenance=MODEL_DERIVED,
                  status=NOT_ASSESSABLE, reason_code=MISSING_DATA)
    assert claim.reason_code == MISSING_DATA


def test_combined_requires_components():
    with pytest.raises(ValueError, match="components"):
        Claim(claim="x", claim_kind="data_statement", provenance=COMBINED,
              status=SUPPORTED)


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status"):
        Claim(claim="x", claim_kind="data_statement", provenance=MODEL_DERIVED,
              status="PROBABLY_FINE")


def test_unknown_provenance_is_rejected():
    with pytest.raises(ValueError, match="provenance"):
        Claim(claim="x", claim_kind="data_statement", provenance="VIBES",
              status=SUPPORTED)


def test_not_assessable_is_not_a_violation():
    """An honest limit is the CORRECT outcome, never a failure."""
    report = ValidationReport(claims=[
        Claim(claim="x", claim_kind="definition", provenance=MODEL_DERIVED,
              status=NOT_ASSESSABLE, reason_code=MISSING_DATA)])
    assert report.has_violation is False


# --- corpus ---------------------------------------------------------------

def test_corpus_parses_every_block_shape(corpus):
    if not corpus.available:
        pytest.skip("docs/corpus is gitignored/local-only")
    ids = {b.block_id for b in corpus.blocks}
    assert "IC2025-STATUS-01" in ids      # shape A
    assert "IC2025-AC-B01" in ids         # shape B
    assert "IC2025-KG-01" in ids          # shape C


def test_corpus_preserves_type_page_and_table(corpus):
    if not corpus.available:
        pytest.skip("docs/corpus is gitignored/local-only")
    table5 = corpus.get("IC2025-T5")
    assert table5 is not None
    assert "Table 5" in (table5.table or "")
    assert "21" in (table5.page or "")
    assert "recommendation" in (table5.evidence_type or "")


def test_corpus_exposes_limitations_and_knowledge_gaps(corpus):
    if not corpus.available:
        pytest.skip("docs/corpus is gitignored/local-only")
    assert len(corpus.knowledge_gaps()) >= 10
    assert len(corpus.limitations()) >= 15


def test_corpus_keeps_the_absent_definition_block(corpus):
    if not corpus.available:
        pytest.skip("docs/corpus is gitignored/local-only")
    block = corpus.get("IC2025-AC-B01")
    assert "No operational definition" in block.text


def test_corpus_never_invents_a_field(corpus):
    """A block whose source line prints no table must not acquire one."""
    if not corpus.available:
        pytest.skip("docs/corpus is gitignored/local-only")
    for block in corpus.blocks:
        if block.table:
            assert "Table" in block.table or "Figure" in block.table


def test_missing_corpus_is_reported_not_faked(tmp_path):
    empty = corpus_module.Corpus(tmp_path / "nope.md")
    assert empty.available is False
    assert empty.blocks == []

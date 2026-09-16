"""[SCIENTIFIC] block -- Istanbul Consensus blocks for the LLM context.

Skips (never fails) when the corpus file is absent from the checkout: docs/
is local-only and the corpus lives in docs/corpus/.
"""

from pathlib import Path

import pytest

from orchestrator import scientific_reference as sr
from validator.corpus import Corpus

CLEAVAGE_Q = "Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division chaotique ?"
TIMING_Q = "Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques disponibles ?"
MODEL_ONLY_Q = "Quelle est la deuxième phase la plus probable autour de cette transition ?"

corpus_present = pytest.mark.skipif(not Corpus().available, reason="Istanbul corpus not on this machine")


@corpus_present
def test_every_limitation_id_in_the_table_exists_in_the_corpus():
    corpus = Corpus()
    for marker, ids in sr.LIMITATIONS_BY_MARKER.items():
        for block_id in ids:
            assert corpus.get(block_id) is not None, (marker, block_id)


def test_a_question_without_scientific_topic_gets_no_block(monkeypatch):
    out = sr.retrieve_scientific_reference(MODEL_ONLY_Q)
    assert out["blocks"] == [] and out["limitations"] == []
    assert out["scientific_markers"] == []
    assert any("aucun sujet couvert" in w for w in out["warnings"])


def test_a_timing_only_question_about_the_model_lag_gets_no_block():
    """Q7: 'avec quel decalage ?' is a model-vs-annotation quantity; Table 1's
    reference timings would be a lexical distraction, not a reference."""
    out = sr.retrieve_scientific_reference(
        "Le modèle détecte-t-il la transition avant ou après l'annotation, et avec quel décalage ?")
    assert "timing" in out["scientific_markers"]
    assert out["blocks"] == [] and out["limitations"] == []


def test_uncertainty_question_gets_no_block():
    out = sr.retrieve_scientific_reference(
        "Le niveau d'incertitude est-il plus élevé autour de cette transition que pendant les périodes stables ?")
    assert out["blocks"] == [] and out["limitations"] == []


@corpus_present
def test_cleavage_question_gets_the_absent_definition_limitation():
    out = sr.retrieve_scientific_reference(CLEAVAGE_Q)
    ids = {b["block_id"] for b in out["blocks"]} | {b["block_id"] for b in out["limitations"]}
    assert "IC2025-AC-B01" in ids and "IC2025-ABSENT-01" in ids
    assert all(b["role"] == "evidence" for b in out["blocks"])
    assert all(b["role"] == "limitation" for b in out["limitations"])
    assert all(b["source"] == "ISTANBUL_CONSENSUS_2025" for b in out["blocks"] + out["limitations"])


@corpus_present
def test_timing_question_carries_the_hpi_and_median_limitations():
    out = sr.retrieve_scientific_reference(TIMING_Q)
    ids = {b["block_id"] for b in out["blocks"]} | {b["block_id"] for b in out["limitations"]}
    assert {"IC2025-CP1-01", "IC2025-LIM-09"} <= ids


@corpus_present
def test_block_text_is_verbatim_and_fields_are_only_what_the_corpus_prints():
    corpus = Corpus()
    out = sr.retrieve_scientific_reference(CLEAVAGE_Q)
    for entry in out["blocks"] + out["limitations"]:
        block = corpus.get(entry["block_id"])
        assert entry["text"] == block.text
        assert entry["evidence_grade"] == block.evidence_grade
        assert entry["page"] == block.page


@corpus_present
def test_no_duplicate_between_evidence_and_limitations():
    out = sr.retrieve_scientific_reference(CLEAVAGE_Q)
    ids = [b["block_id"] for b in out["blocks"] + out["limitations"]]
    assert len(ids) == len(set(ids))


def test_missing_corpus_is_reported_never_reconstructed(tmp_path):
    empty = Corpus(path=tmp_path / "absent.md")
    out = sr.retrieve_scientific_reference(CLEAVAGE_Q, corpus=empty)
    assert out["available"] is False and out["blocks"] == [] and out["limitations"] == []
    assert any("indisponible" in w for w in out["warnings"])


@corpus_present
def test_scientific_context_text_exposes_ids_and_bodies_for_grounding():
    out = sr.retrieve_scientific_reference(TIMING_Q)
    text = sr.scientific_context_text(out)
    assert "IC2025-CP1-01" in text
    assert sr.scientific_context_text(None) == ""

"""The pairing the VALIDATOR OFF vs ON experiment rests on: both arms derive
from ONE raw answer, OFF keeps it verbatim, ON appends the qualification, and
the two arms never disagree on anything but the validator's output."""

from orchestrator.run_validator_experiment import CONDITION_OFF, CONDITION_ON, arms_from_raw
from test_prompt_compact_format import _full_context


def test_both_arms_share_the_raw_answer_and_differ_only_by_the_qualification():
    ctx = _full_context("Quelle etait la phase dominante juste avant la transition ?")
    raw = "L'embryon était en t7."
    arms = arms_from_raw(raw, ctx)
    assert arms[CONDITION_OFF]["final_answer"] == raw
    assert arms[CONDITION_ON]["final_answer"].startswith(raw)
    assert arms[CONDITION_ON]["final_answer"] != raw
    assert arms[CONDITION_OFF]["validation"]["enabled"] is False
    assert arms[CONDITION_ON]["validation"]["enabled"] is True
    assert arms[CONDITION_ON]["validation"]["provenance_valid"] is False


def test_arms_are_deterministic():
    ctx = _full_context()
    raw = "Le modèle prédit t7 ; l'annotation dit t4."
    assert arms_from_raw(raw, ctx) == arms_from_raw(raw, ctx)
    assert arms_from_raw(raw, ctx)[CONDITION_ON]["final_answer"] == raw   # nothing to qualify

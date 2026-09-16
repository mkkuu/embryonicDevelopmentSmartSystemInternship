"""POST /chat with the Scientific Validator in the loop: the shown answer is
the FINAL (qualified) answer, the raw answer is returned alongside, and the
validation summary is additive -- the legacy body (no validation keys) is
unchanged, as Tests/webapp_api/test_app.py keeps pinning."""

import pytest

pytest.importorskip("flask")

from test_app import _fake_answer_question_result, client  # noqa: E402,F401


def _with_validation(raw, final, enabled=True):
    out = _fake_answer_question_result(text=raw)
    out["final_answer"] = final
    out["context_policy"] = {"context_format": "compact_v2", "curated_retrieval": True,
                             "scientific_reference": True, "validator_enabled": enabled}
    out["validation"] = {
        "enabled": enabled, "report": {"claims": []}, "provenance_valid": False,
        "claim_validation": {"n_claims": 1, "by_status": {"NOT_SUPPORTED": 1}, "has_violation": True,
                             "rules_fired": {"R2_prediction_is_not_observation": 1}},
        "scientific_validation": {}, "qualification_appended": final != raw,
        "raw_answer_preserved": True, "final_answer": final, "error": None,
    }
    return out


def test_chat_returns_the_qualified_answer_and_keeps_the_raw_one(monkeypatch, client):
    raw = "L'embryon était en t7."
    final = raw + "\n\n--- QUALIFICATION SCIENTIFIQUE ---\n- « L'embryon était en t7. » -> NOT_SUPPORTED : ..."
    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question",
                        lambda question, video_id=None, window=None, split="val", llm=None, **kw:
                        _with_validation(raw, final))
    body = client.post("/chat", json={"question": "Quelle phase ?"}).get_json()
    assert body["answer"] == final
    assert body["raw_answer"] == raw
    assert body["validation"]["enabled"] is True
    assert body["validation"]["provenance_valid"] is False
    assert body["validation"]["has_violation"] is True
    assert body["validation"]["by_status"] == {"NOT_SUPPORTED": 1}
    assert body["validation"]["qualification_appended"] is True
    assert body["context_format"] == "compact_v2"
    assert body["grounded"] is True          # untouched presence check


def test_chat_with_validator_off_shows_the_raw_answer(monkeypatch, client):
    raw = "L'embryon était en t7."
    out = _with_validation(raw, raw, enabled=False)
    out["validation"].update({"report": None, "provenance_valid": None, "claim_validation": None,
                              "qualification_appended": False})
    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question",
                        lambda question, video_id=None, window=None, split="val", llm=None, **kw: out)
    body = client.post("/chat", json={"question": "Quelle phase ?"}).get_json()
    assert body["answer"] == raw and body["raw_answer"] == raw
    assert body["validation"]["enabled"] is False and body["validation"]["n_claims"] is None


def test_chat_legacy_result_without_validation_keeps_the_legacy_body(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question",
                        lambda question, video_id=None, window=None, split="val", llm=None, **kw:
                        _fake_answer_question_result(text="La phase est t6."))
    body = client.post("/chat", json={"question": "Quelle phase ?"}).get_json()
    assert body["answer"] == "La phase est t6."
    assert "raw_answer" not in body and "validation" not in body

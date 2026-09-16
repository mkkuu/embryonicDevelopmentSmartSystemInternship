"""The frozen benchmark questions Q1-Q15 as served by the Application API
(Training/webapp_api/app.py): one source of truth, verbatim wording,
question-by-id chat, and the static markup the frontend needs.

Flask test client + monkeypatch, same convention as test_app.py -- no real
data, model or Ollama server. The wording below is the FROZEN reference
(docs/RAG_FIXED_QUESTION_BENCHMARK.md, 2026-09-01): it is repeated here,
in a test, precisely so that a change to either the benchmark module or the
UI feed fails loudly. It is not a second implementation -- the app serves
`orchestrator.fixed_question_benchmark.FIXED_QUESTIONS` itself.
"""

import re
from pathlib import Path

import pytest

from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from webapp_api.app import app

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_HTML = ROOT / "Training" / "webapp_api" / "templates" / "index.html"
APP_JS = ROOT / "Training" / "webapp_api" / "static" / "app.js"
STYLE_CSS = ROOT / "Training" / "webapp_api" / "static" / "style.css"
APP_PY = ROOT / "Training" / "webapp_api" / "app.py"

# The 15 frozen formulations, byte for byte (U+2019 apostrophes included).
FROZEN_WORDING = [
    "Quelle a été la dernière transition détectée et à quel moment a-t-elle eu lieu ?",
    "Depuis combien de temps l’embryon se trouve-t-il dans la phase actuelle ?",
    "Quelle était la phase dominante juste avant la transition ?",
    "Quelle est la deuxième phase la plus probable autour de cette transition ?",
    "À quel moment les probabilités des deux phases commencent-elles à se rapprocher ?",
    "Comment les probabilités évoluent-elles dans les fenêtres précédant et suivant la transition ?",
    "Le modèle détecte-t-il la transition avant ou après l’annotation, et avec quel décalage ?",
    "La prédiction est-elle stable après la transition ou observe-t-on des retours vers la phase précédente ?",
    "Y a-t-il un saut de phase ou une régression dans la séquence prédite ?",
    "Le niveau d’incertitude est-il plus élevé autour de cette transition que pendant les périodes stables ?",
    "Cette transition est-elle compatible avec l’ordre attendu du développement embryonnaire ?",
    "Quelle est la prochaine étape développementale attendue selon le référentiel ?",
    "Le moment observé pour cette transition est-il compatible avec les repères morphocinétiques disponibles ?",
    "Le décalage observé doit-il être considéré comme atypique ou peut-il relever de la variabilité biologique connue ?",
    "Le motif observé peut-il correspondre à un direct cleavage, un reverse cleavage ou une division chaotique ?",
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fake_result(text="Réponse.", category="DYNAMIC_DATA"):
    return {
        "question": "q", "route": {"category": category},
        "tool_plan": {"called": ["get_current_inference"], "not_called": []},
        "context": {"document_context": [], "dynamic_context": {}, "warnings": []},
        "response": {"text": text, "answer": text, "provider": "ollama", "grounded": True,
                     "warnings": [], "sources": [], "dynamic_data": [],
                     "used_tools": ["get_current_inference"], "confidence": "medium"},
    }


# --- the benchmark module itself is the frozen reference -----------------

def test_benchmark_holds_exactly_the_fifteen_frozen_formulations():
    assert [q.question for q in FIXED_QUESTIONS] == FROZEN_WORDING
    assert [q.question_id for q in FIXED_QUESTIONS] == [f"Q{i}" for i in range(1, 16)]


# --- GET /questions -----------------------------------------------------

def test_questions_endpoint_serves_the_benchmark_questions_verbatim_and_in_order(client):
    resp = client.get("/questions")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_questions"] == 15
    assert len(body["questions"]) == 15
    assert [q["question"] for q in body["questions"]] == FROZEN_WORDING
    assert [q["id"] for q in body["questions"]] == [q.question_id for q in FIXED_QUESTIONS]
    assert [q["number"] for q in body["questions"]] == list(range(1, 16))
    assert [q["category"] for q in body["questions"]] == [q.category for q in FIXED_QUESTIONS]


def test_questions_endpoint_never_leaks_grading_material(client):
    """The UI must never show the expected answer next to the question."""
    body = client.get("/questions").get_json()
    for q in body["questions"]:
        assert set(q) == {"id", "number", "question", "category"}, sorted(q)


def test_questions_endpoint_is_read_only(client):
    assert client.post("/questions", json={}).status_code == 405


def test_the_app_holds_no_second_copy_of_any_question():
    """The wording lives in orchestrator/fixed_question_benchmark.py ONLY.
    Neither the app, the template nor the frontend may spell out a question."""
    for path in (APP_PY, INDEX_HTML, APP_JS):
        source = path.read_text(encoding="utf-8")
        for wording in FROZEN_WORDING:
            assert wording not in source, f"{path.name} duplicates a frozen question: {wording[:40]}..."
    app_source = APP_PY.read_text(encoding="utf-8")
    assert "fixed_question_benchmark.FIXED_QUESTIONS" in app_source


# --- POST /chat by question id -------------------------------------------

def test_chat_by_question_id_sends_the_frozen_wording_to_the_pipeline(monkeypatch, client):
    captured = {}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        captured.update(question=question, video_id=video_id, window=window, split=split,
                        llm_is_set=llm is not None)
        return _fake_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={"question_id": "Q7", "video_id": "Patient_319", "window": 156, "split": "val"})
    assert resp.status_code == 200
    assert captured == {"question": FROZEN_WORDING[6], "video_id": "Patient_319", "window": 156,
                        "split": "val", "llm_is_set": True}
    body = resp.get_json()
    assert body["question_id"] == "Q7"
    assert body["question"] == FROZEN_WORDING[6]
    assert body["answer"] == "Réponse."


@pytest.mark.parametrize("question_id", [q.question_id for q in FIXED_QUESTIONS])
def test_every_frozen_question_id_resolves_to_its_own_wording(monkeypatch, client, question_id):
    seen = {}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None, **kwargs):
        seen["question"] = question
        return _fake_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={"question_id": question_id, "video_id": "Patient_1", "window": 0})
    assert resp.status_code == 200
    expected = next(q.question for q in FIXED_QUESTIONS if q.question_id == question_id)
    assert seen["question"] == expected


def test_chat_unknown_question_id_returns_400(monkeypatch, client):
    called = {"n": 0}

    def fake_answer_question(*args, **kwargs):
        called["n"] += 1
        return _fake_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    resp = client.post("/chat", json={"question_id": "Q99", "video_id": "Patient_1", "window": 0})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_request"
    assert called["n"] == 0, "the pipeline must not be reached for an unknown id"


def test_chat_rejects_both_question_id_and_free_text(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question",
                        lambda *a, **k: _fake_result())
    resp = client.post("/chat", json={"question_id": "Q1", "question": "autre chose"})
    assert resp.status_code == 400


def test_chat_by_question_id_still_rejects_test_split(client):
    resp = client.post("/chat", json={"question_id": "Q1", "split": "test"})
    assert resp.status_code == 403


def test_chat_free_text_is_unchanged_and_reports_no_question_id(monkeypatch, client):
    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question",
                        lambda question, video_id=None, window=None, split="val", llm=None, **k: _fake_result())
    resp = client.post("/chat", json={"question": "Quelle est la phase actuelle ?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["question_id"] is None
    assert body["question"] == "Quelle est la phase actuelle ?"


def test_chat_passes_the_configured_timeout_to_the_pipeline(monkeypatch, client):
    from webapp_api import app as app_module
    captured = {}

    def fake_answer_question(question, video_id=None, window=None, split="val", llm=None,
                             llm_timeout_seconds=None):
        captured["timeout"] = llm_timeout_seconds
        captured["provider_timeout"] = llm.request_timeout_seconds
        return _fake_result()

    monkeypatch.setattr("webapp_api.app.orchestrator_module.answer_question", fake_answer_question)
    client.post("/chat", json={"question_id": "Q1"})
    assert captured["timeout"] == app_module._CHAT_LLM_TIMEOUT_SECONDS
    assert captured["timeout"] == captured["provider_timeout"], (
        "the orchestrator budget and the HTTP request timeout must be the same number"
    )
    assert captured["timeout"] > 30, "30 s cut real benchmark answers (max 33.1 s) -- the UI needs more"


# --- static markup / frontend contract -----------------------------------

def test_fixed_questions_section_exists_and_follows_the_viewer():
    source = INDEX_HTML.read_text(encoding="utf-8")
    for element_id in ("fixed-questions-section", "fq-context", "fq-form", "fq-list", "fq-submit",
                       "fq-status", "fq-answer-block", "fq-answer-question", "fq-sent-context",
                       "fq-answer", "fq-grounding", "fq-provenance", "fq-sources", "fq-warnings"):
        assert f'id="{element_id}"' in source, f"missing element id={element_id!r}"
    viewer_at = source.index('id="viewer-section"')
    fixed_at = source.index('id="fixed-questions-section"')
    assert viewer_at < fixed_at, "layout: viewer, then fixed questions"


def test_the_free_question_panel_is_gone():
    """The free-text question feature was removed on request (UI only). The
    fixed-question panel is now the only way to reach the LLM from the page;
    POST /chat itself stays, because that is what a fixed question uses."""
    source = INDEX_HTML.read_text(encoding="utf-8")
    for fragment in ('id="chat-section"', 'id="chat-form"', 'id="chat-question"',
                     'id="chat-answer-block"', 'id="chat-meta"', 'id="chat-context"',
                     "Poser une question libre"):
        assert fragment not in source, f"the removed free-question panel is back: {fragment!r}"
    js = APP_JS.read_text(encoding="utf-8")
    for fragment in ('getElementById("chat-form")', 'getElementById("chat-question")',
                     'getElementById("chat-answer-block")'):
        assert fragment not in js, f"dead free-question handler left behind: {fragment!r}"
    css = STYLE_CSS.read_text(encoding="utf-8")
    for selector in ("#chat-form", "#chat-question", "#chat-answer-block", "#chat-meta",
                     "#chat-sources", "#chat-warnings", "#chat-sent-context", "#chat-context "):
        assert selector not in css, f"dead free-question style left behind: {selector}"


def test_fixed_questions_ui_strings_are_in_french():
    source = INDEX_HTML.read_text(encoding="utf-8")
    for expected in ("Questions fixes", "Poser la question", "Réponse", "Détails techniques"):
        assert expected in source, f"missing French UI string: {expected!r}"


def test_the_list_is_empty_in_the_template_and_filled_from_the_endpoint():
    source = INDEX_HTML.read_text(encoding="utf-8")
    assert re.search(r'<ol id="fq-list"[^>]*></ol>', source), "the template must not pre-render any question"
    js = APP_JS.read_text(encoding="utf-8")
    assert "fetchJSON(`/questions`)" in js
    assert "function loadFixedQuestions(" in js
    assert "function renderFixedQuestionList(" in js


def test_frontend_sends_the_question_by_id_only():
    code = "\n".join(line for line in APP_JS.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("//"))
    assert "question_id: asked.id" in code
    assert "function askFixedQuestion(" in code
    assert "function invalidateFixedQuestionAnswerIfStale(" in code, (
        "an answer must be tied to its (patient, window) and removed when the selection changes"
    )


def test_fixed_questions_have_styles():
    css = STYLE_CSS.read_text(encoding="utf-8")
    for selector in (".fq-list", ".fq-item", "#fq-submit", ".fq-status", ".answer-block"):
        assert selector in css, f"missing style rule for {selector}"


def test_no_model_name_or_technical_route_is_shown_in_the_fixed_question_panel():
    """The user-facing lines carry plain-language provenance; raw tool names
    and the router category only appear inside the collapsed technical block."""
    js = APP_JS.read_text(encoding="utf-8")
    assert "TOOL_PROVENANCE_LABELS" in js
    assert "function toolProvenanceLabel(" in js
    index = INDEX_HTML.read_text(encoding="utf-8")
    assert "<details" in index and "Détails techniques" in index

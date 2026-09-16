"""Static-source regression guard for docs/WEBAPP_SECURITY.md -- Phase 7's
frontend (Training/webapp_api/static/app.js, templates/index.html) must
never reintroduce `.innerHTML =`/`insertAdjacentHTML`/`.outerHTML =`/
`document.write` for dynamic (backend/LLM/user-derived) content. Pure
source-text checks, no Flask/torch/browser needed -- this is the
permanent, automated form of this session's manual "audit final" grep
(docs/WEBAPP_SECURITY.md sec on the diagnostic table), wired into the
normal `pytest Tests/` suite so a future regression is caught
automatically rather than relying on someone re-running the grep by
hand. The runtime/behavioral proof (malicious payloads actually render
as inert text) lives in the companion, dependency-free Node test:
`node Tests/webapp_api/test_chat_dom_safety.mjs` (a different ecosystem,
not collected by pytest)."""

from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent.parent / "Training" / "webapp_api" / "static" / "app.js"
INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "Training" / "webapp_api" / "templates" / "index.html"

UNSAFE_PATTERNS = ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write")


def test_app_js_exists():
    assert APP_JS.is_file(), f"expected {APP_JS} to exist"


def test_app_js_contains_no_unsafe_dom_sink():
    source = APP_JS.read_text(encoding="utf-8")
    found = [p for p in UNSAFE_PATTERNS if p in source]
    assert not found, (
        f"Training/webapp_api/static/app.js contains unsafe DOM sink(s) {found} -- "
        f"docs/WEBAPP_SECURITY.md requires all dynamic (backend/LLM/user-derived) "
        f"content to be inserted via textContent/createElement, never HTML strings."
    )


def test_index_html_contains_no_inline_unsafe_dom_sink():
    # index.html has no inline <script> logic at all (script.src points at
    # static/app.js) -- this just guards against one ever being added with
    # an unsafe sink inside it.
    source = INDEX_HTML.read_text(encoding="utf-8")
    found = [p for p in UNSAFE_PATTERNS if p in source]
    assert not found, f"templates/index.html contains unsafe DOM sink(s) {found}"


def test_safe_dom_helpers_are_present():
    # Positive-presence check: guards against the fix being silently
    # reverted/renamed without test_app_js_contains_no_unsafe_dom_sink
    # catching it (e.g. someone re-inlining the old template-literal
    # pattern under a different, still-unsafe method name).
    source = APP_JS.read_text(encoding="utf-8")
    for helper in ("function textSpan(", "function renderWarningsList(", "function renderSources("):
        assert helper in source, f"expected safe-DOM helper {helper!r} in app.js"


def test_answer_panel_elements_use_replace_children_or_text_content():
    # The answer-panel sinks named in docs/WEBAPP_SECURITY.md must be driven
    # by replaceChildren/textContent, never a raw HTML string assignment --
    # this is the same guarantee as test_app_js_contains_no_unsafe_dom_sink,
    # restated per-element for a more specific failure message if one
    # regresses. The free-question panel (chat-meta, chat-sent-context,
    # chat-context) was removed; the surviving sinks are the fixed-question
    # ones, whose element ids are asserted below.
    source = APP_JS.read_text(encoding="utf-8")
    for var_name in ("groundingEl", "provenanceEl", "sourcesEl", "warningsEl"):
        assert f"{var_name}.innerHTML" not in source, f"{var_name} must never use innerHTML"
    for element_id in ("fq-context", "fq-answer", "fq-sent-context", "fq-answer-question"):
        assert f'getElementById("{element_id}")' in source, (
            f"expected the {element_id!r} sink to still be written from JS"
        )

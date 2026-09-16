"""
Tests for OllamaLLMProvider -- NO real Ollama server required anywhere.
The happy-path test mocks `urllib.request.urlopen` (never touches the
network); the failure-path test connects to a real, deliberately-closed
local port (no Ollama is installed anywhere in this project as of this
phase, docs/LLM_PROVIDER_DECISION.md) to confirm the failure is caught
cleanly, not that a fake success is simulated.
"""

import json

from orchestrator.context_builder import build_context
from orchestrator.llm_provider import OllamaLLMProvider, OllamaUnavailableError
from orchestrator.router import classify


def _context():
    from orchestrator.context_builder import SOURCE_DOCUMENT, ToolCallResult

    results = {
        "retrieve_documents": ToolCallResult(
            tool_name="retrieve_documents", source_type=SOURCE_DOCUMENT, ok=True,
            value=[{"chunk_id": "a::0001", "content": "Le Semi-HMM ajoute un modele de duree explicite.",
                    "score": 0.9,
                    "metadata": {"source": "docs/HANDOFF_SEMI_HMM.md", "section": "1",
                                 "status": "authoritative", "authoritative": True}}],
        ),
    }
    return build_context("Qu'est-ce que le Semi-HMM ?", classify("Qu'est-ce que le Semi-HMM ?"), results,
                          tools_called=["retrieve_documents"])


def test_ollama_provider_never_reaches_a_host_other_than_base_url(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Le Semi-HMM ajoute un modele de duree explicite au HMM."})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaLLMProvider(model="mistral", base_url="http://localhost:11434")
    response = provider.generate(_context())

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == "mistral"
    assert captured["body"]["stream"] is False
    assert "system" in captured["body"] and "prompt" in captured["body"]
    assert response.provider == "ollama"
    assert response.grounded is True  # the fake text only restates real context content
    assert response.sources
    # num_ctx is ALWAYS sent (2026-09-02); seed/temperature remain opt-in, so
    # with neither set the options dict must contain nothing but num_ctx.
    from orchestrator.llm_provider import DEFAULT_NUM_CTX
    assert captured["body"]["options"] == {"num_ctx": DEFAULT_NUM_CTX}


def test_ollama_provider_passes_seed_through_when_set(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Le Semi-HMM ajoute un modele de duree explicite au HMM."})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaLLMProvider(model="mistral", base_url="http://localhost:11434", seed=42)
    provider.generate(_context())

    from orchestrator.llm_provider import DEFAULT_NUM_CTX
    assert captured["body"]["options"] == {"num_ctx": DEFAULT_NUM_CTX, "seed": 42}


def test_ollama_provider_passes_temperature_through_when_set(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Le Semi-HMM ajoute un modele de duree explicite au HMM."})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaLLMProvider(model="mistral", base_url="http://localhost:11434", temperature=0.1)
    provider.generate(_context())

    from orchestrator.llm_provider import DEFAULT_NUM_CTX
    assert captured["body"]["options"] == {"num_ctx": DEFAULT_NUM_CTX, "temperature": 0.1}


def test_ollama_provider_passes_seed_and_temperature_together(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Le Semi-HMM ajoute un modele de duree explicite au HMM."})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaLLMProvider(model="mistral", base_url="http://localhost:11434", seed=42, temperature=0.1)
    provider.generate(_context())

    from orchestrator.llm_provider import DEFAULT_NUM_CTX
    assert captured["body"]["options"] == {"num_ctx": DEFAULT_NUM_CTX, "seed": 42, "temperature": 0.1}


def test_ollama_provider_raises_a_typed_error_when_server_unreachable():
    # Port 1 is a real, always-closed local port -- no Ollama server is
    # installed anywhere in this project (docs/LLM_PROVIDER_DECISION.md);
    # this deliberately exercises the real failure path, not a mock.
    provider = OllamaLLMProvider(model="mistral", base_url="http://127.0.0.1:1", request_timeout_seconds=2.0)
    try:
        provider.generate(_context())
        assert False, "expected OllamaUnavailableError"
    except OllamaUnavailableError as e:
        assert "ollama serve" in str(e)
        assert "mistral" in str(e)


def test_ollama_provider_failure_is_caught_by_the_orchestrator_safety_net(monkeypatch):
    from orchestrator import tools
    from orchestrator.orchestrator import answer_question

    def fake_retrieve_documents(query, top_k=5, **kwargs):
        return [{"chunk_id": "a::0001", "content": "x", "score": 0.9,
                 "metadata": {"source": "docs/X.md", "section": "1", "status": "current",
                              "authoritative": False}}]

    monkeypatch.setattr(tools, "retrieve_documents", fake_retrieve_documents)

    provider = OllamaLLMProvider(model="mistral", base_url="http://127.0.0.1:1", request_timeout_seconds=1.0)
    out = answer_question("Qu'est-ce que le Semi-HMM ?", llm=provider)
    assert out["response"]["grounded"] is False
    assert "OllamaUnavailableError" in out["response"]["text"]


# ---------------------------------------------------------------------------
# num_ctx -- the context window actually sent to Ollama.
#
# Why these exist: with no `num_ctx` in options, Ollama 0.7.0 silently
# truncated a ~21k-token prompt to prompt_eval_count=4096 (measured on the
# real server, 2026-09-02). Every benchmark prompt above 4096 tokens was
# therefore being scored on a prompt the model had only partly seen. These
# tests pin the fix so it cannot regress into silence again.
# ---------------------------------------------------------------------------

import pytest

from orchestrator import llm_provider as _llm


def _capture_body(monkeypatch, provider):
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "ok"})

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider.generate(_context())
    return captured["body"]


def test_num_ctx_is_always_sent_even_with_no_seed_or_temperature(monkeypatch):
    body = _capture_body(monkeypatch, OllamaLLMProvider(model="llama3.2:latest"))
    assert "options" in body, "options must always be present now"
    assert body["options"]["num_ctx"] == _llm.DEFAULT_NUM_CTX


def test_default_num_ctx_is_the_documented_value():
    assert _llm.DEFAULT_NUM_CTX == 16384
    assert OllamaLLMProvider(model="llama3.2:latest").num_ctx == 16384


def test_explicit_num_ctx_overrides_the_default(monkeypatch):
    body = _capture_body(monkeypatch, OllamaLLMProvider(model="llama3.2:latest", num_ctx=8192))
    assert body["options"]["num_ctx"] == 8192


def test_env_var_overrides_the_default(monkeypatch):
    monkeypatch.setenv(_llm.NUM_CTX_ENV_VAR, "32768")
    assert OllamaLLMProvider(model="llama3.2:latest").num_ctx == 32768


def test_explicit_argument_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(_llm.NUM_CTX_ENV_VAR, "32768")
    assert OllamaLLMProvider(model="llama3.2:latest", num_ctx=8192).num_ctx == 8192


def test_blank_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(_llm.NUM_CTX_ENV_VAR, "   ")
    assert OllamaLLMProvider(model="llama3.2:latest").num_ctx == _llm.DEFAULT_NUM_CTX


@pytest.mark.parametrize("bad", [0, -1, "big", 3.5, True])
def test_invalid_explicit_num_ctx_raises_at_construction(bad):
    # fail loudly at construction, never silently per-request -- a silently
    # wrong context window is the exact bug this parameter exists to end.
    with pytest.raises(ValueError):
        OllamaLLMProvider(model="llama3.2:latest", num_ctx=bad)


@pytest.mark.parametrize("bad", ["nope", "0", "-5"])
def test_invalid_env_var_raises_at_construction(monkeypatch, bad):
    monkeypatch.setenv(_llm.NUM_CTX_ENV_VAR, bad)
    with pytest.raises(ValueError):
        OllamaLLMProvider(model="llama3.2:latest")


def test_num_ctx_resolution_is_frozen_at_construction(monkeypatch):
    # a provider instance's window must not change mid-run if the env moves.
    provider = OllamaLLMProvider(model="llama3.2:latest")
    monkeypatch.setenv(_llm.NUM_CTX_ENV_VAR, "2048")
    body = _capture_body(monkeypatch, provider)
    assert body["options"]["num_ctx"] == _llm.DEFAULT_NUM_CTX


def test_num_ctx_coexists_with_seed_and_temperature(monkeypatch):
    # conservation guard: the pre-existing opt-in options still work.
    body = _capture_body(
        monkeypatch, OllamaLLMProvider(model="llama3.2:latest", seed=7, temperature=0.0))
    assert body["options"] == {"num_ctx": _llm.DEFAULT_NUM_CTX, "seed": 7, "temperature": 0.0}


def test_default_num_ctx_exceeds_the_largest_measured_prompt():
    # measured 2026-09-02 after the top_k compaction: Q8 was the largest
    # assembled prompt at ~9,767 estimated tokens. The window must leave
    # real room for the answer on top of that.
    largest_measured_prompt_tokens = 9767
    assert _llm.DEFAULT_NUM_CTX > largest_measured_prompt_tokens
    assert _llm.DEFAULT_NUM_CTX - largest_measured_prompt_tokens >= 4096

"""
Runner for the RAG Fixed-Question Functional Benchmark
(`Training/orchestrator/fixed_question_benchmark.py`,
`docs/RAG_FIXED_QUESTION_BENCHMARK.md`) -- the project's new, durable
primary functional benchmark. Reuses the existing evaluation
infrastructure exactly (never bypasses the real pipeline): the real,
unmodified `orchestrator.answer_question()`
(QUESTION -> Router -> Tools -> Context Builder -> LLM -> Grounding ->
Response), the real `grounding_check.check_grounding()` for full detail,
and `prompt.build_system_prompt()`/`build_user_prompt()` to capture the
exact LLM input text -- same pattern as
`Training/orchestrator/run_llm_model_ab_test.py`/
`run_rag_context_failure_diagnostic.py`, not a new, parallel mechanism.

Deliberately does NOT auto-classify SUCCESS/PARTIAL/FAIL/HALLUCINATION/
HONEST_LIMITATION -- that judgment is made by reading each real answer
against the question's own `expected_information`/`expected_behavior`
(same manual-rubric discipline as every prior LLM-quality evaluation in
this project, `docs/RAG_LLM_QUALITY_REPORT.md`'s own "automating this
would reintroduce NLP complexe" principle) and recorded in
`docs/RAG_FIXED_QUESTION_BASELINE.md`, not by this script.

Comparability (docs/PROJECT_CHECKPOINT.md 2026-09-03b, "Lacune de
traçabilité"): the artefact records every generation parameter
(`provider`: model, num_ctx, seed, temperature, timeout -- via
`llm_config.describe_provider()`) plus, when the server answers, the
Ollama version and the model digest, so two runs can be compared without
reconstructing their settings from code and environment. The artefact is
also rewritten after EVERY question (same path), so a crash at Q7 no
longer destroys Q1-Q6 (the 2026-09-02h OOM lost a full run that way).

Model selection goes through the same resolver as the Web App
(`orchestrator/llm_config.py`): `--model` > `$LLM_MODEL` > the production
default. A comparison run is therefore always explicit about its model,
and only the generative model differs between runs.

Usage (from Training/, GPU server):
    python -m orchestrator.run_fixed_question_benchmark --model llama3.2:latest
    python -m orchestrator.run_fixed_question_benchmark --model mistral-nemo:12b --timeout 240 --run-label rep2
"""

from __future__ import annotations

import argparse
import json
import os
import time

from orchestrator import grounding_check, llm_config, prompt as prompt_module, scientific_validation
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.llm_provider import OllamaLLMProvider, OllamaUnavailableError
from orchestrator.orchestrator import answer_question


def ollama_server_metadata(base_url: str, model: str) -> dict:
    """Best-effort, read-only: Ollama's version and the model's digest, for
    the artefact header. Never fails the run -- an unreachable server is
    reported as None here and surfaces per question through the normal
    OllamaUnavailableError path."""
    import json as _json
    import urllib.error
    import urllib.request

    meta = {"ollama_version": None, "model_digest": None, "model_details": None}
    try:
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=5) as r:
            meta["ollama_version"] = _json.loads(r.read().decode("utf-8")).get("version")
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return meta
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as r:
            for entry in _json.loads(r.read().decode("utf-8")).get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    meta["model_digest"] = entry.get("digest")
                    meta["model_details"] = entry.get("details")
                    break
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        pass
    return meta


def ollama_placement(base_url: str, model: str) -> dict:
    """Best-effort, read-only: how Ollama placed the loaded model (VRAM vs
    RAM, i.e. whether layers were offloaded to CPU because GPU0 was
    shared). Recorded after the first question so a run made under GPU
    contention can never be silently compared with a 100 %-GPU run."""
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5) as r:
            for entry in _json.loads(r.read().decode("utf-8")).get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    size = entry.get("size") or 0
                    vram = entry.get("size_vram") or 0
                    return {"size_bytes": size, "size_vram_bytes": vram,
                            "gpu_fraction": (vram / size) if size else None}
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        pass
    return {"size_bytes": None, "size_vram_bytes": None, "gpu_fraction": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                         help="Ollama model tag. Default: $LLM_MODEL, else the production default "
                              f"({llm_config.DEFAULT_LLM_MODEL}) -- orchestrator/llm_config.py.")
    parser.add_argument("--timeout", type=float, default=60.0,
                         help="Per-question timeout. Production /chat's own default is 30s "
                              "(Training/webapp_api/app.py); a modest margin above that is used "
                              "here so a real but slightly slow answer isn't cut off mid-baseline, "
                              "without approving mistral-small3.1:24b-scale latencies.")
    parser.add_argument("--run-label", default="baseline")
    parser.add_argument("--context-format", default=None,
                        help="Context rendering sent to the LLM: v2 (historical default), compact, "
                             "compact_v2. Default: $EMBRYO_LLM_CONTEXT_FORMAT, else v2.")
    parser.add_argument("--validator", default=None, choices=["on", "off"],
                        help="Scientific Validator as post-generation control layer. Default: "
                             "$EMBRYO_SCIENTIFIC_VALIDATOR, else on. The raw answer is recorded "
                             "either way; only `final_answer` and `validation` change.")
    args = parser.parse_args()
    validate = None if args.validator is None else (args.validator == "on")

    if args.context_format is not None:
        # The documented selection mechanism is the environment variable
        # (prompt.resolve_context_format: explicit > env > v2); the CLI flag
        # sets it for this process so the provider is still built through the
        # ONE shared constructor call below, unchanged.
        os.environ[prompt_module.CONTEXT_FORMAT_ENV_VAR] = args.context_format
    provider: OllamaLLMProvider = llm_config.build_ollama_provider(args.model, args.timeout)
    provider_meta = llm_config.describe_provider(provider)
    provider_meta["validator_enabled"] = scientific_validation.resolve_validator_enabled(validate)
    provider_meta.update(ollama_server_metadata(provider.base_url, provider.model))
    model_slug = provider.model.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = (f"../Results/evaluation/event_anomaly_rag_inventory/"
                    f"fixed_question_benchmark_{model_slug}_{args.run_label}.json")
    print(f"model={provider.model} timeout={provider.request_timeout_seconds} "
          f"num_ctx={provider.num_ctx} seed={provider.seed} temperature={provider.temperature} "
          f"ollama={provider_meta.get('ollama_version')} digest={provider_meta.get('model_digest')}")

    rows = []

    def persist(complete: bool) -> None:
        # Rewritten after every question: a partial artefact is explicitly
        # marked as such and never mistaken for a full run.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"model": provider.model, "timeout": provider.request_timeout_seconds,
                       "run_label": args.run_label, "provider": provider_meta,
                       "n_questions": len(FIXED_QUESTIONS), "n_completed": len(rows),
                       "complete": complete, "rows": rows}, f, ensure_ascii=False, indent=2)

    for q in FIXED_QUESTIONS:
        t0 = time.time()
        try:
            out = answer_question(q.question, video_id=q.video_id, window=q.window, split=q.split,
                                   llm=provider, llm_timeout_seconds=args.timeout, validate=validate)
        except OllamaUnavailableError as e:
            elapsed = time.time() - t0
            print(f"[{q.question_id}] FAILED after {elapsed:.1f}s: {e}")
            rows.append({
                "question_id": q.question_id, "question": q.question, "category": q.category,
                "type": q.type, "current_support": q.current_support, "gaps": q.gaps,
                "is_historical": q.is_historical, "is_gap_test": q.is_gap_test,
                "error": str(e), "elapsed_seconds": elapsed,
            })
            persist(complete=False)
            continue
        elapsed = time.time() - t0
        context = out["context"]
        response = out["response"]

        system_prompt = prompt_module.build_system_prompt()
        user_prompt = prompt_module.build_user_prompt(context, context_format=provider.context_format)
        grounding_detail = grounding_check.check_grounding(response["text"], context)

        rows.append({
            "question_id": q.question_id, "question": q.question, "category": q.category,
            "type": q.type, "expected_tools": q.expected_tools, "expected_sources": q.expected_sources,
            "expected_information": q.expected_information, "expected_behavior": q.expected_behavior,
            "success_criterion": q.success_criterion, "behavior_if_absent": q.behavior_if_absent,
            "current_support": q.current_support, "gaps": q.gaps,
            "is_historical": q.is_historical, "is_gap_test": q.is_gap_test,
            "video_id": q.video_id, "window": q.window, "split": q.split,
            "elapsed_seconds": elapsed,
            "route": out["route"],
            "tool_plan": out["tool_plan"],
            "dynamic_context": context["dynamic_context"],
            "document_context": context["document_context"],
            "context_warnings": context["warnings"],
            "scientific_context": context.get("scientific_context"),
            "retrieval_audit": context.get("retrieval_audit"),
            "context_policy": out.get("context_policy"),
            "llm_input_user_prompt_length": len(user_prompt),
            "llm_input_user_prompt": user_prompt,
            "answer": response["text"],                 # RAW LLM answer, never rewritten
            "final_answer": out.get("final_answer"),    # raw + validator qualification (or raw)
            "validation": out.get("validation"),
            "llm_self_reported_grounded": response["grounded"],
            "grounding_detail": grounding_detail.to_dict(),
            "final_confidence": response["confidence"],
            "final_warnings": response["warnings"],
        })
        print(f"[{q.question_id}] elapsed={elapsed:.1f}s route={out['route']['category']} "
              f"tools={out['tool_plan']['called']} grounded={grounding_detail.grounded}")
        if "placement" not in provider_meta:
            provider_meta["placement"] = ollama_placement(provider.base_url, provider.model)
            print(f"placement={provider_meta['placement']}")
        persist(complete=False)

    persist(complete=True)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

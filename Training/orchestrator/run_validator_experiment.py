"""
Paired VALIDATOR OFF vs ON experiment over the frozen Q1-Q15 benchmark,
context format COMPACT_V2.

DESIGN -- why ONE generation feeds BOTH arms
--------------------------------------------
The Scientific Validator is a deterministic control layer applied AFTER
generation (`orchestrator/scientific_validation.py`): it never touches the
prompt, the tools, the context or the LLM call. The only way the two arms can
receive "exactly the same context and the same raw answer" is therefore to
generate ONCE and derive both arms from that single raw answer:

    for each question (real pipeline, unmodified, executed ONCE):
        route, plan, results, context  (policy compact_v2)
        for each replication:
            raw = LLM(context)                       <- one stochastic call
            OFF = validate_response(raw, ctx, enabled=False)   -> final = raw
            ON  = validate_response(raw, ctx, enabled=True)    -> final = raw + qualification

Running the LLM twice would re-introduce its own run-to-run spread (sigma 0.75
points over 15 questions, 2026-09-07c) into a comparison whose single variable
is a deterministic post-processing step. The pairing here is exact by
construction, and `validate_response` is the very function
`answer_question()` calls in production.

What is recorded per replication: the raw answer, both final answers, the
full validation report (claims, provenance, statuses, evidence, limitations),
the independent grounding detail (identical in both arms, by construction),
latency, and the provider placement. Per question: the context dict, the
exact compact_v2 prompt, the retrieval audit, the scientific block, the
transition context. In the header: every hash needed to prove the conditions
(system prompt, questions, validator modules, corpus, renderer, Ollama
version/digest).

Nothing is modified: no dataset, annotation, model, corpus, index, question or
scoring criterion; one NEW artefact is written, never over an existing one.
Scoring stays manual (frozen `success_criterion`), by protocol.

Usage (from Training/, GPU server, embedder on CPU):
    CUDA_VISIBLE_DEVICES='' python -m orchestrator.run_validator_experiment \
        --model mistral-nemo:12b --timeout 240 --replications 3 --run-label validator_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from orchestrator import grounding_check, llm_config, prompt as prompt_module, scientific_validation
from orchestrator.compact_v2_render import CONTEXT_FORMAT_COMPACT_V2
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.llm_provider import OllamaUnavailableError
from orchestrator.orchestrator import execute_plan, generate_grounded_response, plan_tools
from orchestrator.router import classify
from orchestrator.run_fixed_question_benchmark import ollama_placement, ollama_server_metadata
from validator.corpus import DEFAULT_CORPUS_PATH

OUTPUT_DIR = "../Results/evaluation/validator_experiment"
CONDITION_OFF = "VALIDATOR_OFF"
CONDITION_ON = "VALIDATOR_ON"

_TRAINING = Path(__file__).resolve().parent.parent
_HASHED_MODULES = (
    "orchestrator/prompt.py", "orchestrator/compact_v2_render.py", "orchestrator/compact_context.py",
    "orchestrator/scientific_reference.py", "orchestrator/scientific_validation.py",
    "orchestrator/orchestrator.py", "orchestrator/temporal_context.py", "orchestrator/grounding_check.py",
    "validator/validator.py", "validator/rules.py", "validator/extraction.py",
    "validator/provenance.py", "validator/retrieval.py", "validator/corpus.py", "validator/schema.py",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def arms_from_raw(raw_answer: str, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Both arms from ONE raw answer. Pure and deterministic: the pairing the
    whole experiment rests on. Exposed for Tests/orchestrator."""
    off = scientific_validation.validate_response(raw_answer, context, enabled=False)
    on = scientific_validation.validate_response(raw_answer, context, enabled=True)
    return {
        CONDITION_OFF: {"final_answer": off["final_answer"], "validation": off},
        CONDITION_ON: {"final_answer": on["final_answer"], "validation": on},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Ollama model tag (reference: mistral-nemo:12b).")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--run-label", default="validator_v1")
    parser.add_argument("--questions", default=None, help="Comma-separated ids for a smoke run.")
    args = parser.parse_args()

    selected = {q.strip() for q in args.questions.split(",")} if args.questions else None
    questions = [q for q in FIXED_QUESTIONS if selected is None or q.question_id in selected]

    provider = llm_config.build_ollama_provider(args.model, args.timeout,
                                                context_format=CONTEXT_FORMAT_COMPACT_V2)
    provider_meta = llm_config.describe_provider(provider)
    provider_meta.update(ollama_server_metadata(provider.base_url, provider.model))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_slug = provider.model.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = f"{OUTPUT_DIR}/paired_validator_off_vs_on_{model_slug}_{args.run_label}.json"
    if os.path.exists(output_path):
        raise SystemExit(f"{output_path} already exists -- refusing to overwrite a prior artefact. "
                         f"Use a different --run-label.")

    header = {
        "artifact": "paired_validator_off_vs_on",
        "run_label": args.run_label,
        "conditions_compared": [CONDITION_OFF, CONDITION_ON],
        "design": ("paired: ONE tool execution and ONE context per question (policy compact_v2); "
                   "per replication ONE LLM generation, whose raw answer feeds BOTH arms; "
                   "OFF = raw answer, ON = raw answer + Scientific Validator qualification"),
        "single_variable": ("scientific_validation.validate_response(enabled) -- a deterministic "
                            "post-generation layer. Context, prompt, tools, retrieval, scientific "
                            "models, LLM call, raw answer and grounding are identical in both arms."),
        "context_format": CONTEXT_FORMAT_COMPACT_V2,
        "replications": args.replications,
        "provider": provider_meta,
        "system_prompt_sha256": _sha256(prompt_module.build_system_prompt()),
        "questions_sha256": _sha256(_canonical([q.question for q in FIXED_QUESTIONS])),
        "module_sha256": {m: _sha256_file(_TRAINING / m) for m in _HASHED_MODULES},
        "istanbul_corpus_sha256": _sha256_file(DEFAULT_CORPUS_PATH),
        "validator_engine": scientific_validation.VALIDATOR_ENGINE,
        "n_questions": len(questions),
        "benchmark_module": "orchestrator.fixed_question_benchmark (frozen, unmodified)",
        "scoring": "manual, frozen success_criterion, PASS=1 / PARTIAL=0.5 / FAIL=0 -- graded after "
                   "this run, separately for raw_answer and for each arm's final_answer",
    }
    print(f"model={provider.model} num_ctx={provider.num_ctx} seed={provider.seed} "
          f"temperature={provider.temperature} timeout={provider.request_timeout_seconds} "
          f"context_format={provider.context_format} ollama={provider_meta.get('ollama_version')} "
          f"reps={args.replications}", flush=True)

    rows: List[Dict[str, Any]] = []

    def persist(complete: bool) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({**header, "complete": complete, "n_completed": len(rows), "rows": rows},
                      f, ensure_ascii=False, indent=2)

    for q in questions:
        route = classify(q.question)
        plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
        results = execute_plan(plan, q.question, q.video_id, q.window, q.split,
                               context_policy=CONTEXT_FORMAT_COMPACT_V2)
        ctx = build_context(q.question, route, results, tools_called=plan.called)
        user_prompt = prompt_module.build_user_prompt(ctx, CONTEXT_FORMAT_COMPACT_V2)

        row: Dict[str, Any] = {
            "question_id": q.question_id, "question": q.question, "category": q.category,
            "type": q.type, "current_support": q.current_support, "is_gap_test": q.is_gap_test,
            "success_criterion": q.success_criterion,
            "expected_information": q.expected_information,
            "expected_behavior": q.expected_behavior,
            "video_id": q.video_id, "window": q.window, "split": q.split,
            "route": route.to_dict(), "tool_plan": plan.to_dict(),
            "dynamic_context": ctx["dynamic_context"],
            "document_context": ctx["document_context"],
            "scientific_context": ctx.get("scientific_context"),
            "retrieval_audit": ctx.get("retrieval_audit"),
            "context_warnings": ctx["warnings"],
            "dynamic_context_sha256": _sha256(_canonical(ctx["dynamic_context"])),
            "document_context_sha256": _sha256(_canonical(ctx["document_context"])),
            "scientific_context_sha256": _sha256(_canonical(ctx.get("scientific_context"))),
            "llm_input_user_prompt": user_prompt,
            "llm_input_user_prompt_chars": len(user_prompt),
            "replications": [],
        }
        print(f"[{q.question_id}] route={route.category} tools={plan.called} "
              f"docs={len(ctx['document_context'])} "
              f"sci={[b['block_id'] for b in (ctx.get('scientific_context') or {}).get('blocks', [])]} "
              f"prompt={len(user_prompt)} chars", flush=True)
        rows.append(row)
        persist(complete=False)

        for replication in range(1, args.replications + 1):
            t0 = time.time()
            try:
                response = generate_grounded_response(provider, ctx, timeout_seconds=args.timeout)
            except OllamaUnavailableError as e:
                row["replications"].append({"replication": replication, "error": str(e),
                                            "elapsed_seconds": round(time.time() - t0, 3)})
                print(f"  rep{replication}: FAILED {e}", flush=True)
                persist(complete=False)
                continue
            elapsed = time.time() - t0
            raw = response.text
            detail = grounding_check.check_grounding(raw, ctx)
            arms = arms_from_raw(raw, ctx)
            on = arms[CONDITION_ON]["validation"]
            entry = {
                "replication": replication,
                "raw_answer": raw,
                "raw_answer_sha256": _sha256(raw),
                "grounded": response.grounded,
                "confidence": response.confidence,
                "response_warnings": list(response.warnings),
                "grounding_detail": detail.to_dict(),
                "llm_elapsed_seconds": round(elapsed, 3),
                CONDITION_OFF: arms[CONDITION_OFF],
                CONDITION_ON: arms[CONDITION_ON],
                "claims": (on.get("report") or {}).get("claims"),
                "available_provenance": (on.get("report") or {}).get("available_provenance"),
                "provenance_valid": on.get("provenance_valid"),
                "has_violation": (on.get("claim_validation") or {}).get("has_violation"),
                "qualification_appended": on.get("qualification_appended"),
            }
            row["replications"].append(entry)
            print(f"  rep{replication}: {elapsed:.1f}s grounded={response.grounded} "
                  f"len={len(raw)} claims={len(entry['claims'] or [])} "
                  f"provenance_valid={entry['provenance_valid']} "
                  f"qualified={entry['qualification_appended']}", flush=True)
            persist(complete=False)

        if "placement" not in provider_meta:
            provider_meta["placement"] = ollama_placement(provider.base_url, provider.model)
            header["provider"] = provider_meta
            print(f"placement={provider_meta['placement']}", flush=True)
        persist(complete=False)

    persist(complete=True)
    print(f"\nWrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

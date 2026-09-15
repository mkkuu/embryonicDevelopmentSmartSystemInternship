"""
Paired FULL vs OBSERVED-ONLY context-ablation experiment over the frozen
Q1-Q15 benchmark.

DESIGN
------
For each of the 15 frozen questions, at the question's own frozen anchor
(Patient_319 / val / window 156):

    route   = router.classify(q)                    ] the real production
    plan    = orchestrator.plan_tools(...)          ] pipeline, unmodified,
    results = orchestrator.execute_plan(...)        ] executed ONCE
    ctx_full = context_builder.build_context(...)   ]

    ctx_observed_only = observed_only.filter_context(ctx_full)

    for each replication:
        answer_full     = generate_grounded_response(llm, ctx_full)
        answer_observed = generate_grounded_response(llm, ctx_observed_only)

The two arms therefore share ONE tool execution: the same cached forward pass,
the same annotation, the same RAG chunks, the same `inference_timestamp`. The
only difference between the two prompts is the model-derived content removed by
`observed_only.filter_context()`. This is a stronger pairing than re-running the
pipeline per arm, where the contexts would differ by their timestamps.

Every function called here is the real, unmodified production one --
`classify`, `plan_tools`, `execute_plan`, `build_context`,
`generate_grounded_response` (which re-runs `grounding_check` exactly as
production does), `prompt.build_system_prompt`/`build_user_prompt`. The
experiment adds a filter between context and generation; it changes no step.

`answer_question()` itself is not called, because it fuses context building and
generation and would rebuild the context per arm. `Tests/orchestrator/
test_observed_only.py` pins that this runner's FULL arm reproduces
`answer_question()`'s own context for the same inputs.

SCORING is deliberately NOT automated: the project's rubric is the frozen
`success_criterion` of each question, read by a human against the real answer
(docs/RAG_LLM_QUALITY_REPORT.md's "automating this would reintroduce NLP
complexe"). This runner produces the auditable artefact; the grades are added
afterwards, exactly like every prior benchmark run.

Usage (from Training/, on the GPU server):
    python -m experiments.context_ablations.run_observed_only_experiment \
        --model mistral-nemo:12b --timeout 240 --replications 3 --run-label obs_only_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List

from orchestrator import grounding_check, llm_config, prompt as prompt_module
from experiments.context_ablations import observed_only
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.llm_provider import OllamaUnavailableError
from orchestrator.orchestrator import (
    execute_plan,
    generate_grounded_response,
    plan_tools,
)
from orchestrator.router import classify
from orchestrator.run_fixed_question_benchmark import ollama_placement, ollama_server_metadata

OUTPUT_DIR = "../Results/evaluation/observed_only_experiment"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _generate(llm, context: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """One arm, one replication. Returns the row fragment for this answer."""
    t0 = time.time()
    response = generate_grounded_response(llm, context, timeout_seconds=timeout)
    elapsed = time.time() - t0
    detail = grounding_check.check_grounding(response.text, context)
    return {
        "answer": response.text,
        "grounded": response.grounded,
        "confidence": response.confidence,
        "warnings": list(response.warnings),
        "grounding_detail": detail.to_dict(),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Ollama model tag. Default: $LLM_MODEL, else "
                             f"{llm_config.DEFAULT_LLM_MODEL}. The reference benchmark condition "
                             "is mistral-nemo:12b -- pass it explicitly to reproduce it.")
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Per-answer timeout, seconds (reference runs used 240).")
    parser.add_argument("--replications", type=int, default=3,
                        help="Replications of EACH condition (the LLM is stochastic: seed is not "
                             "set in this project, by protocol).")
    parser.add_argument("--run-label", default="observed_only")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question ids for a smoke run (e.g. Q1,Q4). "
                             "Default: all 15, in their frozen order.")
    args = parser.parse_args()

    selected = None
    if args.questions:
        selected = {q.strip() for q in args.questions.split(",") if q.strip()}
    questions = [q for q in FIXED_QUESTIONS if selected is None or q.question_id in selected]

    provider = llm_config.build_ollama_provider(args.model, args.timeout)
    provider_meta = llm_config.describe_provider(provider)
    provider_meta.update(ollama_server_metadata(provider.base_url, provider.model))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_slug = provider.model.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = f"{OUTPUT_DIR}/paired_{model_slug}_{args.run_label}.json"
    if os.path.exists(output_path):
        raise SystemExit(f"{output_path} already exists -- refusing to overwrite a prior "
                         f"experiment artefact. Use a different --run-label.")

    system_prompt = prompt_module.build_system_prompt()
    header = {
        "artifact": "paired_full_vs_observed_only_context_ablation",
        "run_label": args.run_label,
        "conditions_compared": [observed_only.CONDITION_FULL, observed_only.CONDITION_OBSERVED_ONLY],
        "design": "paired: ONE tool execution per question, two contexts derived from it, "
                  "both answered by the same provider, replications interleaved FULL then "
                  "OBSERVED-ONLY back to back",
        "replications": args.replications,
        "provider": provider_meta,
        "system_prompt_sha256": _sha256(system_prompt),
        "questions_sha256": _sha256(_canonical([q.question for q in FIXED_QUESTIONS])),
        "n_questions": len(questions),
        "benchmark_module": "orchestrator.fixed_question_benchmark (frozen, unmodified)",
        "scoring": "manual, frozen success_criterion, PASS=1 / PARTIAL=0.5 / FAIL=0 -- graded "
                   "after this run, not by this script",
    }
    print(f"model={provider.model} num_ctx={provider.num_ctx} seed={provider.seed} "
          f"temperature={provider.temperature} timeout={provider.request_timeout_seconds} "
          f"ollama={provider_meta.get('ollama_version')} reps={args.replications}")

    rows: List[Dict[str, Any]] = []

    def persist(complete: bool) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({**header, "complete": complete, "n_completed": len(rows), "rows": rows},
                      f, ensure_ascii=False, indent=2)

    for q in questions:
        route = classify(q.question)
        plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
        results = execute_plan(plan, q.question, q.video_id, q.window, q.split)
        ctx_full = build_context(q.question, route, results, tools_called=plan.called)
        ctx_observed = observed_only.filter_context(ctx_full)

        # Fail fast, before any token is generated: a leak must stop the
        # experiment, never silently contaminate the OBSERVED-ONLY arm.
        observed_only.assert_no_model_derived(ctx_observed)

        # Proof, recorded per question, that the annotation is the SAME in both
        # arms -- computed on both contexts by the same projection function.
        observed_full = observed_only.observed_payload(ctx_full)
        observed_filtered = observed_only.observed_payload(ctx_observed)
        observed_identical = _canonical(observed_full) == _canonical(observed_filtered)

        prompt_full = prompt_module.build_user_prompt(ctx_full)
        prompt_observed = prompt_module.build_user_prompt(ctx_observed)

        row: Dict[str, Any] = {
            "question_id": q.question_id,
            "question": q.question,
            "category": q.category,
            "type": q.type,
            "current_support": q.current_support,
            "is_gap_test": q.is_gap_test,
            "success_criterion": q.success_criterion,
            "expected_information": q.expected_information,
            "expected_behavior": q.expected_behavior,
            "video_id": q.video_id, "window": q.window, "split": q.split,
            "route": route.to_dict(),
            "tool_plan": plan.to_dict(),
            "observed_only_computability": observed_only.computability_of(q.type),
            "observed_only_removed_paths": ctx_observed["observed_only_removed_paths"],
            "n_removed_paths": len(ctx_observed["observed_only_removed_paths"]),
            "observed_data_identical_between_conditions": observed_identical,
            "observed_payload_sha256": _sha256(_canonical(observed_full)),
            "model_derived_keys_found_in_observed_only": observed_only.find_model_derived_keys(
                ctx_observed["dynamic_context"]),
            "document_context_sha256_full": _sha256(_canonical(ctx_full["document_context"])),
            "document_context_sha256_observed_only": _sha256(
                _canonical(ctx_observed["document_context"])),
            "full_dynamic_context": ctx_full["dynamic_context"],
            "observed_only_dynamic_context": ctx_observed["dynamic_context"],
            "document_context": ctx_full["document_context"],
            "context_warnings": ctx_full["warnings"],
            "full_user_prompt": prompt_full,
            "observed_only_user_prompt": prompt_observed,
            "full_user_prompt_chars": len(prompt_full),
            "observed_only_user_prompt_chars": len(prompt_observed),
            "replications": [],
        }

        print(f"[{q.question_id}] route={route.category} tools={plan.called} "
              f"removed={row['n_removed_paths']} prompt {len(prompt_full)} -> "
              f"{len(prompt_observed)} chars  computability={row['observed_only_computability']}")

        # Appended before the first answer so a crash mid-question still leaves
        # the question's contexts and removal evidence on disk.
        rows.append(row)
        persist(complete=False)

        for replication in range(1, args.replications + 1):
            entry: Dict[str, Any] = {"replication": replication}
            for condition, context in ((observed_only.CONDITION_FULL, ctx_full),
                                       (observed_only.CONDITION_OBSERVED_ONLY, ctx_observed)):
                try:
                    entry[condition] = _generate(provider, context, args.timeout)
                except OllamaUnavailableError as e:
                    entry[condition] = {"error": str(e)}
                    print(f"  rep{replication} {condition}: FAILED {e}")
                    continue
                print(f"  rep{replication} {condition}: {entry[condition]['elapsed_seconds']}s "
                      f"grounded={entry[condition]['grounded']} "
                      f"len={len(entry[condition]['answer'])}")
            row["replications"].append(entry)
            persist(complete=False)

        if "placement" not in provider_meta:
            provider_meta["placement"] = ollama_placement(provider.base_url, provider.model)
            header["provider"] = provider_meta
            print(f"placement={provider_meta['placement']}")
        persist(complete=False)

    persist(complete=True)
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

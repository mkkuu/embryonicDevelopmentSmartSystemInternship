"""
PREDICTION-ONLY experiment over the frozen Q1-Q15 benchmark.

DESIGN
------
For each of the 15 frozen questions, at the question's own frozen anchor
(Patient_319 / val / window 156):

    route   = router.classify(q)                    ] the real production
    plan    = orchestrator.plan_tools(...)          ] pipeline, unmodified,
    results = orchestrator.execute_plan(...)        ] executed ONCE
    ctx_full = context_builder.build_context(...)   ]

    ctx_pred = prediction_only.filter_context(ctx_full)

    for each replication:
        answer = generate_grounded_response(llm, ctx_pred)

Only ONE condition is generated here: PREDICTION_ONLY. The FULL context is built
because the production pipeline builds it -- it is the input the filter subtracts
from -- but it is NEVER sent to the provider. Its annotation is projected into an
explicitly-named `evaluator_ground_truth` block so the later analysis can compare
MODEL-DERIVED against the annotation; that block is evaluator metadata and is
never rendered into a prompt.

Every function called here is the real, unmodified production one --
`classify`, `plan_tools`, `execute_plan`, `build_context`,
`generate_grounded_response` (which re-runs `grounding_check` exactly as
production does), `prompt.build_system_prompt`/`build_user_prompt`. The
experiment adds a filter between context and generation; it changes no step, and
it modifies no renderer.

FAIL-FAST ON LEAK
-----------------
Before a single token is generated, every question's filtered context is run
through four independent checks (`prediction_only.assert_no_observed`): forbidden
key names at any depth, annotation provenance markers in any string value,
forbidden substrings in the RENDERED dynamic-context section, and survival by
value of the annotation payloads taken from the FULL context. A leak aborts the
whole run.

SCORING is deliberately NOT automated: the project's rubric is the frozen
`success_criterion` of each question, read by a human against the real answer.
This runner produces the auditable artefact; the grades are added afterwards,
exactly like every prior benchmark run.

Usage (from Training/, on the GPU server):
    python -m experiments.context_ablations.run_prediction_only_experiment --dry-run
    python -m experiments.context_ablations.run_prediction_only_experiment \
        --model mistral-nemo:12b --timeout 240 --replications 3 \
        --run-label prediction_only_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List

from orchestrator import grounding_check, llm_config, prompt as prompt_module
from experiments.context_ablations import prediction_only
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.llm_provider import OllamaUnavailableError
from orchestrator.orchestrator import execute_plan, generate_grounded_response, plan_tools
from orchestrator.prompt import _format_dynamic_context
from orchestrator.router import classify
from orchestrator.run_fixed_question_benchmark import ollama_placement, ollama_server_metadata

OUTPUT_DIR = "../Results/evaluation/prediction_only_experiment"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _generate(llm, context: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """One replication of the single condition."""
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


def build_rows(questions) -> List[Dict[str, Any]]:
    """Executes the production pipeline once per question, filters, and runs the
    leak checks. Returns the per-question rows WITHOUT any answer."""
    rows: List[Dict[str, Any]] = []
    for q in questions:
        route = classify(q.question)
        plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
        results = execute_plan(plan, q.question, q.video_id, q.window, q.split)
        ctx_full = build_context(q.question, route, results, tools_called=plan.called)
        ctx_pred = prediction_only.filter_context(ctx_full)

        rendered_dynamic = _format_dynamic_context(ctx_pred["dynamic_context"])
        # Fail fast, before any token is generated.
        prediction_only.assert_no_observed(ctx_pred, full_context=ctx_full,
                                           rendered_dynamic_section=rendered_dynamic)

        prompt_pred = prompt_module.build_user_prompt(ctx_pred)

        rows.append({
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
            "condition": prediction_only.CONDITION_PREDICTION_ONLY,
            "route": route.to_dict(),
            "tool_plan": plan.to_dict(),
            "prediction_only_computability": prediction_only.computability_of(q.type),
            "prediction_only_removed_paths": ctx_pred["prediction_only_removed_paths"],
            "n_removed_paths": len(ctx_pred["prediction_only_removed_paths"]),
            # leak evidence, recorded per question
            "observed_keys_found_in_prediction_only":
                prediction_only.find_observed_keys(ctx_pred["dynamic_context"]),
            "observed_value_markers_found":
                prediction_only.find_observed_value_markers(ctx_pred["dynamic_context"]),
            "observed_markers_in_rendered_context":
                prediction_only.find_rendered_markers(rendered_dynamic),
            "ground_truth_fingerprints_found":
                prediction_only.find_ground_truth_fingerprints(ctx_pred, ctx_full),
            # context identity
            "prediction_only_dynamic_context": ctx_pred["dynamic_context"],
            "document_context": ctx_pred["document_context"],
            "document_context_sha256": _sha256(_canonical(ctx_pred["document_context"])),
            "dynamic_context_sha256": _sha256(_canonical(ctx_pred["dynamic_context"])),
            "context_warnings": ctx_pred["warnings"],
            "prediction_only_user_prompt": prompt_pred,
            "prediction_only_user_prompt_sha256": _sha256(prompt_pred),
            "prediction_only_user_prompt_chars": len(prompt_pred),
            "full_user_prompt_chars": len(prompt_module.build_user_prompt(ctx_full)),
            # EVALUATOR-ONLY -- never sent to the provider
            "evaluator_ground_truth": prediction_only.evaluator_ground_truth(ctx_full),
            "replications": [],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Ollama model tag. Default: $LLM_MODEL, else "
                             f"{llm_config.DEFAULT_LLM_MODEL}.")
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Per-question wall-clock budget, seconds.")
    parser.add_argument("--replications", type=int, default=3,
                        help="Replications per question.")
    parser.add_argument("--run-label", default="prediction_only")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question ids for a smoke run (e.g. Q1,Q4). "
                             "Default: all 15, in their frozen order.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and verify every context, write the preflight artefact, "
                             "and exit WITHOUT contacting the LLM.")
    args = parser.parse_args()

    selected = None
    if args.questions:
        selected = {q.strip() for q in args.questions.split(",") if q.strip()}
    questions = [q for q in FIXED_QUESTIONS if selected is None or q.question_id in selected]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    system_prompt = prompt_module.build_system_prompt()

    print(f"building {len(questions)} context(s) and running the leak checks…")
    rows = build_rows(questions)
    total_leaks = sum(len(r["observed_keys_found_in_prediction_only"])
                      + len(r["observed_value_markers_found"])
                      + len(r["observed_markers_in_rendered_context"])
                      + len(r["ground_truth_fingerprints_found"]) for r in rows)
    print(f"leak checks: {total_leaks} finding(s) across {len(rows)} question(s)")
    for r in rows:
        print(f"  {r['question_id']:4} removed={r['n_removed_paths']:<3} "
              f"prompt {r['full_user_prompt_chars']} -> {r['prediction_only_user_prompt_chars']} chars "
              f"computability={r['prediction_only_computability']}")

    if args.dry_run:
        path = f"{OUTPUT_DIR}/preflight_{args.run_label}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"artifact": "prediction_only_preflight",
                       "run_label": args.run_label,
                       "system_prompt_sha256": _sha256(system_prompt),
                       "questions_sha256": _sha256(_canonical([q.question for q in FIXED_QUESTIONS])),
                       "n_questions": len(rows),
                       "total_leak_findings": total_leaks,
                       "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nDRY RUN — wrote {path}; no LLM was contacted.")
        return 0 if total_leaks == 0 else 1

    provider = llm_config.build_ollama_provider(args.model, args.timeout)
    provider_meta = llm_config.describe_provider(provider)
    provider_meta.update(ollama_server_metadata(provider.base_url, provider.model))

    model_slug = provider.model.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = f"{OUTPUT_DIR}/paired_{model_slug}_{args.run_label}.json"
    if os.path.exists(output_path):
        raise SystemExit(f"{output_path} already exists -- refusing to overwrite a prior "
                         f"experiment artefact. Use a different --run-label.")

    header = {
        "artifact": "prediction_only_context_condition",
        "run_label": args.run_label,
        "condition": prediction_only.CONDITION_PREDICTION_ONLY,
        "design": "single condition: ONE tool execution per question, the FULL context is filtered "
                  "to MODEL-DERIVED + DERIVED-from-MODEL + RAG, and only that filtered context is "
                  "sent to the provider. The annotation is kept EVALUATOR-SIDE ONLY, under "
                  "rows[].evaluator_ground_truth, and is never rendered into a prompt.",
        "replications": args.replications,
        "provider": provider_meta,
        "system_prompt_sha256": _sha256(system_prompt),
        "questions_sha256": _sha256(_canonical([q.question for q in FIXED_QUESTIONS])),
        "n_questions": len(questions),
        "benchmark_module": "orchestrator.fixed_question_benchmark (frozen, unmodified)",
        "filter_module": "orchestrator.prediction_only (additive; modifies no production module)",
        "total_leak_findings": total_leaks,
        "scoring": "manual, frozen success_criterion, PASS=1 / PARTIAL=0.5 / FAIL=0 -- graded "
                   "after this run, not by this script",
    }

    def persist(complete: bool) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({**header, "complete": complete, "n_completed":
                       sum(1 for r in rows if len(r["replications"]) == args.replications),
                       "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"\nmodel={provider.model} num_ctx={provider.num_ctx} seed={provider.seed} "
          f"temperature={provider.temperature} timeout={provider.request_timeout_seconds} "
          f"ollama={provider_meta.get('ollama_version')} reps={args.replications}")
    persist(complete=False)

    for row in rows:
        ctx_pred = {"question": row["question"],
                    "route": row["route"],
                    "dynamic_context": row["prediction_only_dynamic_context"],
                    "document_context": row["document_context"],
                    "warnings": row["context_warnings"],
                    "model_metadata": None,
                    "condition": prediction_only.CONDITION_PREDICTION_ONLY}
        for replication in range(1, args.replications + 1):
            entry: Dict[str, Any] = {"replication": replication,
                                     "condition": prediction_only.CONDITION_PREDICTION_ONLY}
            try:
                entry["answer_record"] = _generate(provider, ctx_pred, args.timeout)
                a = entry["answer_record"]
                print(f"  {row['question_id']:4} rep{replication}: {a['elapsed_seconds']}s "
                      f"grounded={a['grounded']} len={len(a['answer'])}")
            except OllamaUnavailableError as e:
                entry["answer_record"] = {"error": str(e)}
                print(f"  {row['question_id']:4} rep{replication}: FAILED {e}")
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

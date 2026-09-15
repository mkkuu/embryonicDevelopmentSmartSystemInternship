"""
Paired V2 vs COMPACT_CONTEXT_V1 rendering experiment over the frozen Q1-Q15
benchmark.

HYPOTHESIS UNDER TEST
---------------------
"A shorter, structured, factual context improves the LLM's answers by removing
project-narrative distractors." The intervention is `compact_context.py` plus
`prompt.build_user_prompt(..., context_format="compact")`; nothing else in the
pipeline changes.

DESIGN -- the same pairing discipline as `run_observed_only_experiment.py`
-------------------------------------------------------------------------
For each of the 15 frozen questions, at the question's own frozen anchor
(Patient_319 / val / window 156):

    route   = router.classify(q)                    ] the real production
    plan    = orchestrator.plan_tools(...)          ] pipeline, unmodified,
    results = orchestrator.execute_plan(...)        ] executed ONCE
    ctx     = context_builder.build_context(...)    ]

    for each replication:
        answer_v2      = generate_grounded_response(llm_v2,      ctx)
        answer_compact = generate_grounded_response(llm_compact, ctx)

The two arms share ONE context dict -- the same cached forward pass, the same
annotation, the same retrieved chunks, the same `inference_timestamp`. They
differ by ONE thing only: which renderer turns that dict into the prompt
string. This is the strongest pairing available: the scientific input is not
merely equivalent between arms, it is the same object.

The two providers are built from the same `llm_config.build_ollama_provider()`
call arguments and differ only in `context_format`; both are asserted
identical on every other field before a token is generated.

WHAT THIS RUNNER DOES NOT DO
----------------------------
It does not modify, refit or re-rank anything: no dataset, no annotation, no
Semi-HMM, no embeddings, no Chroma index, no corpus file, no router, no
grounding checker, no question wording, no scoring criterion. It reads the
frozen questions from `orchestrator.fixed_question_benchmark` and writes ONE
new JSON artefact; it never overwrites an existing one.

SCORING is deliberately NOT automated: the project's rubric is the frozen
`success_criterion` of each question, read by a human against the real answer.
This runner produces the auditable artefact; grades are added afterwards,
exactly like every prior benchmark run.

Usage (from Training/, on the GPU server):
    python -m experiments.context_ablations.run_compact_context_experiment \
        --model mistral-nemo:12b --timeout 240 --replications 3 \
        --run-label compact_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List

from orchestrator import compact_context, grounding_check, llm_config, prompt as prompt_module
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.llm_provider import OllamaUnavailableError
from orchestrator.orchestrator import execute_plan, generate_grounded_response, plan_tools
from orchestrator.router import classify
from orchestrator.run_fixed_question_benchmark import ollama_placement, ollama_server_metadata

OUTPUT_DIR = "../Results/evaluation/compact_context_experiment"

CONDITION_V2 = "V2"
CONDITION_COMPACT = "COMPACT"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _generate(llm, context: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """One arm, one replication."""
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
                        help="Ollama model tag. The reference benchmark condition is "
                             "mistral-nemo:12b -- pass it explicitly to reproduce it.")
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Per-answer timeout, seconds (reference runs used 240).")
    parser.add_argument("--replications", type=int, default=3,
                        help="Replications of EACH condition (the LLM is stochastic; seed is "
                             "not set in this project, by protocol).")
    parser.add_argument("--run-label", default="compact_v1")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question ids for a smoke run (e.g. Q11,Q12). "
                             "Default: all 15, in their frozen order.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build both prompts for every question, write the artefact with "
                             "the measurements, and generate NOTHING. No LLM call.")
    args = parser.parse_args()

    selected = None
    if args.questions:
        selected = {q.strip() for q in args.questions.split(",") if q.strip()}
    questions = [q for q in FIXED_QUESTIONS if selected is None or q.question_id in selected]

    provider_v2 = llm_config.build_ollama_provider(
        args.model, args.timeout, context_format=prompt_module.CONTEXT_FORMAT_V2)
    provider_compact = llm_config.build_ollama_provider(
        args.model, args.timeout, context_format=prompt_module.CONTEXT_FORMAT_COMPACT)

    meta_v2 = llm_config.describe_provider(provider_v2)
    meta_compact = llm_config.describe_provider(provider_compact)
    differing = {k for k in meta_v2 if meta_v2[k] != meta_compact.get(k)}
    if differing != {"context_format"}:
        raise SystemExit(f"The two arms must differ by context_format ONLY; they differ by "
                         f"{sorted(differing)}.")
    provider_meta = dict(meta_v2)
    provider_meta.pop("context_format", None)
    provider_meta.update(ollama_server_metadata(provider_v2.base_url, provider_v2.model))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_slug = provider_v2.model.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = f"{OUTPUT_DIR}/paired_{model_slug}_{args.run_label}.json"
    if os.path.exists(output_path):
        raise SystemExit(f"{output_path} already exists -- refusing to overwrite a prior "
                         f"experiment artefact. Use a different --run-label.")

    system_prompt = prompt_module.build_system_prompt()
    header = {
        "artifact": "paired_v2_vs_compact_context_rendering",
        "run_label": args.run_label,
        "conditions_compared": [CONDITION_V2, CONDITION_COMPACT],
        "design": "paired: ONE tool execution and ONE context dict per question, rendered by "
                  "two renderers, both answered by the same model; replications interleaved "
                  "V2 then COMPACT back to back",
        "single_variable": "prompt.build_user_prompt context_format (v2 | compact). The system "
                           "prompt, the questions, the retrieval, the tools, the scientific "
                           "models and the grounding check are identical in both arms.",
        "replications": args.replications,
        "provider": provider_meta,
        "system_prompt_sha256": _sha256(system_prompt),
        "questions_sha256": _sha256(_canonical([q.question for q in FIXED_QUESTIONS])),
        "compact_rules_sha256": _sha256(_canonical([
            compact_context.SECTION_RULES, compact_context.LINE_RULES,
            compact_context.KEEP_RULES, compact_context.MARKER_TEMPLATES])),
        "n_questions": len(questions),
        "benchmark_module": "orchestrator.fixed_question_benchmark (frozen, unmodified)",
        "scoring": "manual, frozen success_criterion, PASS=1 / PARTIAL=0.5 / FAIL=0 -- graded "
                   "after this run, not by this script",
        "dry_run": bool(args.dry_run),
    }
    print(f"model={provider_v2.model} num_ctx={provider_v2.num_ctx} seed={provider_v2.seed} "
          f"temperature={provider_v2.temperature} timeout={provider_v2.request_timeout_seconds} "
          f"ollama={provider_meta.get('ollama_version')} reps={args.replications} "
          f"dry_run={args.dry_run}")

    rows: List[Dict[str, Any]] = []

    def persist(complete: bool) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({**header, "complete": complete, "n_completed": len(rows), "rows": rows},
                      f, ensure_ascii=False, indent=2)

    for q in questions:
        route = classify(q.question)
        plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
        results = execute_plan(plan, q.question, q.video_id, q.window, q.split)
        ctx = build_context(q.question, route, results, tools_called=plan.called)

        prompt_v2 = prompt_module.build_user_prompt(ctx, prompt_module.CONTEXT_FORMAT_V2)
        prompt_compact = prompt_module.build_user_prompt(ctx, prompt_module.CONTEXT_FORMAT_COMPACT)
        compacted = compact_context.compact_document_context(ctx["document_context"])
        stats = compact_context.compaction_stats(compacted)

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
            # The scientific input, identical for both arms by construction --
            # recorded once, with its hash, so the pairing is auditable.
            "dynamic_context": ctx["dynamic_context"],
            "document_context": ctx["document_context"],
            "context_warnings": ctx["warnings"],
            "document_context_sha256": _sha256(_canonical(ctx["document_context"])),
            "dynamic_context_sha256": _sha256(_canonical(ctx["dynamic_context"])),
            # What the filter did, per chunk and in total.
            "compaction_stats": stats,
            "compacted_document_context": [
                {k: v for k, v in c.items() if k.startswith("compact_")
                 or k in ("source", "section", "status", "score")}
                for c in compacted
            ],
            "v2_user_prompt": prompt_v2,
            "compact_user_prompt": prompt_compact,
            "v2_user_prompt_chars": len(prompt_v2),
            "compact_user_prompt_chars": len(prompt_compact),
            "prompt_delta_chars": len(prompt_compact) - len(prompt_v2),
            "prompt_delta_fraction": round(
                (len(prompt_compact) - len(prompt_v2)) / len(prompt_v2), 4) if prompt_v2 else 0.0,
            "replications": [],
        }

        print(f"[{q.question_id}] route={route.category} tools={plan.called} "
              f"rag_chunks={stats['n_chunks']} dropped_chunks={stats['n_chunks_dropped']} "
              f"dropped_lines={stats['n_lines_dropped']} prompt {len(prompt_v2)} -> "
              f"{len(prompt_compact)} chars ({row['prompt_delta_fraction']:+.1%})")

        rows.append(row)
        persist(complete=False)

        if args.dry_run:
            continue

        for replication in range(1, args.replications + 1):
            entry: Dict[str, Any] = {"replication": replication}
            for condition, llm in ((CONDITION_V2, provider_v2),
                                   (CONDITION_COMPACT, provider_compact)):
                try:
                    entry[condition] = _generate(llm, ctx, args.timeout)
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
            provider_meta["placement"] = ollama_placement(provider_v2.base_url, provider_v2.model)
            header["provider"] = provider_meta
            print(f"placement={provider_meta['placement']}")
        persist(complete=False)

    persist(complete=True)
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

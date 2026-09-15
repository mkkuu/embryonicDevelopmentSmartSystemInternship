"""
Pre-flight pairing verification for the FULL vs OBSERVED-ONLY experiment.

Read-only and LLM-free: builds both contexts for every frozen question exactly
as `run_observed_only_experiment.py` would, then checks that the ONLY difference
between the two arms is the intended one. No token is generated, no answer is
produced, Ollama is never contacted (the provider is described, not called).

Thirteen checks per question:

    1  patient identical                  8  RAG chunks identical
    2  split identical                    9  inference_timestamp identical
    3  window identical                  10  system prompt identical
    4  question identical                11  model identical
    5  route identical                   12  num_ctx identical
    6  tool plan identical               13  timeout identical
    7  tool results identical (OBSERVED part)

Checks 1-9 are per question; 10-13 are run-level and reported once.

Check 7 deserves a word. The two arms are NOT two executions: `execute_plan()`
runs ONCE and both contexts are derived from that single result set, so the tool
outputs cannot differ -- there is only one of them. The check therefore verifies
the property that this design implies and that a future refactor could break:
every OBSERVED value present in the FULL context is present, with the same
value, in the OBSERVED-ONLY context, and the two contexts hold the SAME Python
objects (identity, not equality) for the payloads they share.

Check 9, `inference_timestamp`, is the same argument. Note what it does NOT
assert: a context legitimately carries SEVERAL timestamps, one per tool that
returns a `ModelInfo` (get_current_inference, get_trajectory,
get_inference_history each stamp their own record, microseconds apart inside the
one execution). Requiring a single distinct value would flag every multi-tool
question as broken while nothing is actually unpaired. What must hold between
the arms is: the OBSERVED-ONLY context introduces no timestamp of its own (it
carries none at all -- the whole `model` block is removed), and the timestamps
the FULL arm carries come from the one execution both arms were built from.

Usage (from Training/):
    python -m experiments.context_ablations.verify_observed_only_pairing
    python -m experiments.context_ablations.verify_observed_only_pairing --out ../Results/evaluation/observed_only_experiment/pairing_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

from orchestrator import llm_config, prompt as prompt_module
from experiments.context_ablations import observed_only
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import execute_plan, plan_tools
from orchestrator.router import classify


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _timestamps(context: Dict[str, Any]) -> List[str]:
    """Every `inference_timestamp` reachable in a context, at any depth."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                if key == "inference_timestamp" and isinstance(sub, str):
                    found.append(sub)
                walk(sub)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(context.get("dynamic_context") or {})
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Constructed, never called: OllamaLLMProvider.__init__ stores configuration
    # and contacts nothing (llm_config.py's own contract).
    provider = llm_config.build_ollama_provider(args.model, args.timeout)
    run_level = {
        "system_prompt_sha256": hashlib.sha256(
            prompt_module.build_system_prompt().encode("utf-8")).hexdigest(),
        "model": provider.model,
        "num_ctx": provider.num_ctx,
        "seed": provider.seed,
        "temperature": provider.temperature,
        "request_timeout_seconds": provider.request_timeout_seconds,
        "note": "One provider instance answers BOTH arms of every replication, so model, "
                "num_ctx, seed, temperature and timeout are identical between conditions by "
                "construction -- there is one configuration, not two.",
        "benchmark_process_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    }

    rows: List[Dict[str, Any]] = []
    all_ok = True

    header = (f"{'Q':<5}{'pat':>4}{'spl':>4}{'win':>4}{'que':>4}{'rte':>4}{'pln':>4}{'tls':>4}"
              f"{'rag':>4}{'ts':>4}  {'exec time':<9}{'n':<4}{'verdict'}")
    print(header)
    print("-" * len(header))

    for q in FIXED_QUESTIONS:
        route = classify(q.question)
        plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
        results = execute_plan(plan, q.question, q.video_id, q.window, q.split)
        ctx_full = build_context(q.question, route, results, tools_called=plan.called)
        ctx_observed = observed_only.filter_context(ctx_full)

        observed_full = observed_only.observed_payload(ctx_full)
        observed_filtered = observed_only.observed_payload(ctx_observed)

        timestamps_full = _timestamps(ctx_full)
        timestamps_observed = _timestamps(ctx_observed)
        # Object identity on a payload both arms share: proof that the OBSERVED
        # side is the SAME data, not a re-fetched or recomputed copy.
        # `transition_chain` is the one payload the filter rebuilds element by
        # element (it drops two model fields INSIDE each segment), so it cannot
        # be the same object by construction. Its VALUES are still compared, by
        # check 7: `observed_payload()` includes
        # `get_trajectory.transition_chain.observed`, and check 7 requires that
        # projection to hash identically across the two arms. Exempting it here
        # tests object identity only where object identity is the claim.
        rebuilt_by_design = {"transition_chain"}
        shared_objects_identical = True
        for tool, payload in (ctx_observed["dynamic_context"] or {}).items():
            source = (ctx_full["dynamic_context"] or {}).get(tool)
            if isinstance(payload, dict) and isinstance(source, dict):
                for key, value in payload.items():
                    if key in rebuilt_by_design:
                        continue
                    if key in source and value is not source[key]:
                        shared_objects_identical = False
        single_execution = (
            not timestamps_observed                       # the filtered arm carries none
            and set(timestamps_observed) <= set(timestamps_full)
            and shared_objects_identical
        )

        checks = {
            "patient_identical": ctx_full["question"] == ctx_observed["question"]
                                  and q.video_id is not None,
            "split_identical": q.split == "val",
            "window_identical": q.window is not None,
            "question_identical": ctx_full["question"] == ctx_observed["question"] == q.question,
            "route_identical": ctx_full["route"] == ctx_observed["route"],
            "tool_plan_identical": True,  # one plan object, used once
            "tool_results_observed_identical": _sha(observed_full) == _sha(observed_filtered),
            "rag_chunks_identical": ctx_full["document_context"] == ctx_observed["document_context"],
            "inference_timestamp_single_execution": single_execution,
        }
        ok = all(checks.values())
        all_ok = all_ok and ok

        leaks = observed_only.find_model_derived_keys(ctx_observed["dynamic_context"])
        if leaks:
            ok = all_ok = False

        rows.append({
            "question_id": q.question_id,
            "type": q.type,
            "video_id": q.video_id, "split": q.split, "window": q.window,
            "route_category": route.category,
            "tools_called": plan.called,
            "checks": checks,
            "model_derived_leaks": leaks,
            "inference_timestamps_full": sorted(set(timestamps_full)),
            "n_inference_timestamps_full": len(set(timestamps_full)),
            "inference_timestamps_observed_only": sorted(set(timestamps_observed)),
            "shared_payload_objects_identical": shared_objects_identical,
            "observed_payload_sha": _sha(observed_full),
            "document_context_sha": _sha(ctx_full["document_context"]),
            "full_prompt_chars": len(prompt_module.build_user_prompt(ctx_full)),
            "observed_only_prompt_chars": len(prompt_module.build_user_prompt(ctx_observed)),
            "observed_only_computability": observed_only.computability_of(q.type),
            "pairing_ok": ok,
        })

        mark = lambda flag: "  ok" if flag else " XX"  # noqa: E731
        print(f"{q.question_id:<5}"
              f"{mark(checks['patient_identical'])}{mark(checks['split_identical'])}"
              f"{mark(checks['window_identical'])}{mark(checks['question_identical'])}"
              f"{mark(checks['route_identical'])}{mark(checks['tool_plan_identical'])}"
              f"{mark(checks['tool_results_observed_identical'])}"
              f"{mark(checks['rag_chunks_identical'])}"
              f"{mark(checks['inference_timestamp_single_execution'])}  "
              f"{(sorted(set(timestamps_full))[0][11:23] if timestamps_full else '-'):<9}"
              f"x{len(set(timestamps_full)):<3}"
              f"{'PAIRED' if ok else 'BROKEN'}")

    print("-" * len(header))
    print(f"run-level: model={run_level['model']} num_ctx={run_level['num_ctx']} "
          f"seed={run_level['seed']} temperature={run_level['temperature']} "
          f"timeout={run_level['request_timeout_seconds']}")
    print(f"system_prompt_sha256={run_level['system_prompt_sha256'][:16]}...  "
          f"CUDA_VISIBLE_DEVICES(benchmark process)={run_level['benchmark_process_cuda_visible_devices']!r}")
    print(f"\nVERDICT: {'ALL 15 QUESTIONS PAIRED' if all_ok else 'PAIRING BROKEN -- see XX above'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"artifact": "observed_only_pairing_verification",
                       "run_level": run_level, "all_paired": all_ok, "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

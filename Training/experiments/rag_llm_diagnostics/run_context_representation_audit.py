"""
Audit-only script for docs/TEMPORAL_CONTEXT_RESTRUCTURING.md -- NOT a
benchmark, NOT a scoring run, NO LLM call. It builds the REAL context for
a chosen subset of the 15 fixed benchmark questions (same router, same
tool plan, same prompt builder as production) and prints, per question:

  QUESTION -> ROUTE -> TOOLS PLANNED -> the exact CONTEXTE DYNAMIQUE
  block that would be sent to the LLM, plus the raw `series_analysis`
  dict as JSON.

Its purpose is to make the context representation itself inspectable --
the 2026-09-02e restructuring was designed against exactly this output,
and re-running it on the GPU server (where the real Reporting API,
embeddings cache and frozen Semi-HMM are available) is the one-command
confirmation that the window-blocked rendering and the deterministic
series analysis behave identically on live data.

Nothing here writes anything, calls no LLM, and touches no scientific
artifact. `--split test` is refused, as everywhere else in this layer.

    cd Training
    python -m experiments.rag_llm_diagnostics.run_context_representation_audit \
        --questions Q5 Q6 Q8 Q10 --video-id Patient_319 --window 156
"""

from __future__ import annotations

import argparse
import json

from orchestrator import prompt as prompt_module
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import answer_question

DEFAULT_QUESTION_IDS = ["Q1", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTION_IDS,
                        help="question ids from fixed_question_benchmark.py (default: the "
                             "eight audited in docs/TEMPORAL_CONTEXT_RESTRUCTURING.md)")
    parser.add_argument("--video-id", default=None,
                        help="override each question's own video_id (default: use it)")
    parser.add_argument("--window", type=int, default=None,
                        help="override each question's own centre window (default: use it)")
    parser.add_argument("--split", default="val",
                        help="split to query; 'test' is LOCKED and refused")
    parser.add_argument("--json", action="store_true",
                        help="also dump the raw series_analysis dict")
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit("split=test is LOCKED (final pre-registered reporting only) -- refused.")

    wanted = {qid.upper() for qid in args.questions}
    selected = [q for q in FIXED_QUESTIONS if q.question_id in wanted]
    missing = wanted - {q.question_id for q in selected}
    if missing:
        raise SystemExit(f"unknown question id(s): {sorted(missing)}")

    for question in selected:
        video_id = args.video_id if args.video_id is not None else question.video_id
        window = args.window if args.window is not None else question.window
        # NullLLMProvider (answer_question's default) -- the context is what is
        # being audited; generating prose would only add a variable.
        result = answer_question(question.question, video_id=video_id, window=window,
                                 split=args.split)
        context = result["context"]
        dynamic = context.get("dynamic_context", {}) or {}

        print("=" * 100)
        print(f"### {question.question_id}  [{context['route']['category']}]  "
              f"video={video_id} window={window} split={args.split}")
        print(f"### {question.question}")
        print(f"### tools called   : {result['tool_plan']['called']}")
        print(f"### tools skipped  : {[d['tool'] for d in result['tool_plan']['not_called']]}")
        body = prompt_module._format_dynamic_context(dynamic)
        print(f"### rendered lines : {len(body.splitlines())}")
        print(body)
        for warning in context.get("warnings") or []:
            print(f"AVERTISSEMENT : {warning}")
        if args.json and "series_analysis" in dynamic:
            print("--- raw series_analysis ---")
            print(json.dumps(dynamic["series_analysis"], indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()

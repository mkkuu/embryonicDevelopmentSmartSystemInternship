"""
P2.4 -- controlled A/B check for the P1(b) LLM under-extraction DECISION
(docs/RAG_LLM_QUALITY_REPORT.md sec 8-9, docs/PROJECT_CHECKPOINT.md's
2026-08-29 "Next planned action"). Tests ONE candidate fix (a lower
Ollama `temperature`) against the exact 5 real questions the P2.1
benchmark found under-extracted despite correct/sufficient context
(A6/B4/B5/B6/C4, `Results/evaluation/rag_llm_quality/benchmark.json`),
using the same single-variable-experiment discipline already established
in this project (docs/LLM_EVALUATION.md sec 0a: same seed both runs, only
the one parameter under test differs).

Not a re-run of the full 28-question benchmark (deliberately -- the task's
own "don't re-run unless necessary" instruction, `docs/RAG_LLM_QUALITY_REPORT.md`
DO NOT REDO). Reuses `orchestrator.answer_question()` unmodified; the only
new code anywhere is `OllamaLLMProvider`'s additive `temperature` parameter
(`Training/orchestrator/llm_provider.py`).

Usage (from Training/, GPU server, real Ollama required):
    python -m experiments.llm_regression.run_temperature_ab_check --seed 7 --temperature 0.1
"""

from __future__ import annotations

import argparse
import json
import sys

from orchestrator.llm_provider import OllamaLLMProvider, OllamaUnavailableError
from orchestrator.orchestrator import answer_question
from experiments.llm_regression.rag_llm_quality_questions import QUALITY_QUESTIONS

# The 5 real, reproducible under-extraction cases from the P2.1 benchmark
# (docs/RAG_LLM_QUALITY_REPORT.md sec 8-9) -- not the full 28-question set.
TARGET_QIDS = ("A6", "B4", "B5", "B6", "C4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=7,
                         help="Same seed for both runs -- isolates temperature as the only variable.")
    parser.add_argument("--temperature", type=float, default=0.1,
                         help="Candidate low temperature to compare against Ollama's own default.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="../Results/evaluation/rag_llm_quality/temperature_ab_check.json")
    args = parser.parse_args()

    questions = [q for q in QUALITY_QUESTIONS if q.qid in TARGET_QIDS]
    if len(questions) != len(TARGET_QIDS):
        found = {q.qid for q in questions}
        missing = set(TARGET_QIDS) - found
        print(f"ERROR: could not find question(s) {missing} in rag_llm_quality_questions.py", file=sys.stderr)
        return 1

    baseline_provider = OllamaLLMProvider(model=args.model, base_url=args.base_url,
                                           request_timeout_seconds=args.timeout, seed=args.seed)
    candidate_provider = OllamaLLMProvider(model=args.model, base_url=args.base_url,
                                            request_timeout_seconds=args.timeout, seed=args.seed,
                                            temperature=args.temperature)

    rows = []
    for q in questions:
        row = {"qid": q.qid, "question": q.question, "video_id": q.video_id,
               "window": q.window, "split": q.split}
        for label, provider in (("baseline_default_temperature", baseline_provider),
                                 (f"candidate_temperature_{args.temperature}", candidate_provider)):
            try:
                out = answer_question(q.question, video_id=q.video_id, window=q.window,
                                       split=q.split, llm=provider, llm_timeout_seconds=args.timeout)
                response = out["response"]
                row[label] = {
                    "text": response["text"], "grounded": response["grounded"],
                    "warnings": response["warnings"], "confidence": response["confidence"],
                }
            except OllamaUnavailableError as e:
                row[label] = {"error": str(e)}
            print(f"[{q.qid}] {label} done.")
        rows.append(row)

    result = {"model": args.model, "seed": args.seed, "temperature_tested": args.temperature,
              "target_qids": list(TARGET_QIDS), "rows": rows}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

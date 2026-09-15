"""
Runs the canonical evaluation question set (eval_questions.py) through the
REAL orchestrator -- real Reporting API tools, real RAG retrieval, the
real frozen Semi-HMM, the real ingested docs/*.md corpus -- and writes a
JSON + Markdown report.

Requires: torch (Reporting API), chromadb + sentence-transformers (RAG),
the frozen model checkpoint (Results/evaluation/semi_hmm_weekend_phaseF/),
real Val embeddings (Embeddings/resnet18/val/), and an already-ingested
RagIndex/ (see docs/RAG_INGESTION.md). None of this is guaranteed present
in a bare local checkout (CLAUDE.md) -- run this on the GPU server, from
Training/:

    cd Training && python -m experiments.llm_regression.run_orchestration_evaluation

Does NOT call any LLM (NullLLMProvider throughout, matching this
project's "no LLM call anywhere in this repo yet" state,
docs/LLM_ORCHESTRATION.md). Writes:
    Results/evaluation/orchestration_foundation_eval/report.json
    Results/evaluation/orchestration_foundation_eval/REPORT.md

This script has NOT been executed as of the code being written -- do not
treat its own past invocation as a fact; docs/RAG_ORCHESTRATION_PHASE_REPORT.md
states explicitly whether a real run's output exists yet. Read-only with
respect to every scientific artifact (frozen model, embeddings, RagIndex/)
-- it only ever calls existing get_*/retrieve_* tools, never fits/writes
to any of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.llm_regression.eval_questions import EVAL_QUESTIONS  # noqa: E402
from orchestrator.orchestrator import answer_question  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / \
    "Results" / "evaluation" / "orchestration_foundation_eval"


def run(split: str = "val") -> dict:
    rows = []
    n_match = 0
    for eq in EVAL_QUESTIONS:
        out = answer_question(eq.question, video_id=eq.video_id, window=eq.window, split=split)
        got_category = out["route"]["category"]
        matched = got_category == eq.expected_category
        n_match += int(matched)
        rows.append({
            "question": eq.question,
            "expected_category": eq.expected_category,
            "route_selected": got_category,
            "category_match": matched,
            "note": eq.note,
            "tools_called": out["tool_plan"]["called"],
            "tools_not_called": [d["tool"] for d in out["tool_plan"]["not_called"]],
            "n_documents_retrieved": len(out["context"]["document_context"]),
            "n_dynamic_results_retrieved": len(out["context"]["dynamic_context"]),
            "provenance": out["context"]["provenance"],
            "warnings": out["context"]["warnings"],
        })
    return {
        "split": split,
        "n_questions": len(EVAL_QUESTIONS),
        "n_category_matches": n_match,
        "category_accuracy": n_match / len(EVAL_QUESTIONS),
        "rows": rows,
    }


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# RAG / LLM Orchestration Foundation -- Evaluation Run",
        "",
        f"Split: `{report['split']}`. "
        f"Category accuracy: {report['n_category_matches']}/{report['n_questions']} "
        f"({report['category_accuracy']:.1%}).",
        "",
        "No LLM was called (NullLLMProvider) -- this evaluates routing + tool selection + "
        "context assembly only, matching the task's own 'evaluation avant LLM' requirement.",
        "",
        "| # | Question | Expected | Got | Match | Tools called | Tools not called | Docs | Dynamic |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for i, row in enumerate(report["rows"], start=1):
        lines.append(
            f"| {i} | {row['question']} | {row['expected_category']} | {row['route_selected']} | "
            f"{'✓' if row['category_match'] else '✗'} | {', '.join(row['tools_called']) or '—'} | "
            f"{', '.join(row['tools_not_called']) or '—'} | {row['n_documents_retrieved']} | "
            f"{row['n_dynamic_results_retrieved']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run(split=args.split)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, output_dir / "REPORT.md")

    print(f"{report['n_category_matches']}/{report['n_questions']} category matches "
          f"({report['category_accuracy']:.1%}). Written to {output_dir}/")


if __name__ == "__main__":
    main()

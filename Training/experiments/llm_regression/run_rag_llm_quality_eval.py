"""
P2.1 RAG/LLM Quality Evaluation -- docs/RAG_LLM_QUALITY_REPORT.md.

Runs `rag_llm_quality_questions.QUALITY_QUESTIONS` (28 questions across
5 categories) through the REAL end-to-end pipeline (Router -> Tools ->
RAG/Reporting API -> Context Builder -> Ollama llama3.2:latest ->
Grounding), exactly as `POST /chat` does (same `orchestrator.answer_question()`
entry point `Training/webapp_api/app.py` calls), plus a repeated-question
stability subset (4 representative questions x 3 runs each, unseeded --
Ollama's own default sampling, same as the real `/chat` endpoint's
`_CHAT_PROVIDER`).

Distinct from `run_llm_benchmark.py` (the older, 18-question ROUTER-
validation benchmark, unchanged, still authoritative for router-only
regression) -- this script targets END-TO-END ANSWER QUALITY, not just
routing correctness, and captures full raw output (dynamic_context,
document_context, warnings, timing) for a separate, manual scoring pass
against docs/RAG_LLM_QUALITY_REPORT.md's rubric. This script does NOT
itself compute factuality/completeness/usefulness scores -- those are
inherently a judgment call made by reading the raw output against the
question's own `note` and the real tool output, not automatable without
reintroducing exactly the kind of "NLP complexe" this project's own
discipline avoids (router.py's docstring, same principle applied here).

Requires torch (Reporting API) + chromadb/sentence-transformers (RAG) +
a running local Ollama with llama3.2:latest pulled -- GPU server only,
from Training/:

    cd Training && python -m experiments.llm_regression.run_rag_llm_quality_eval

Writes (new, additive directory -- never overwrites a prior benchmark):
    Results/evaluation/rag_llm_quality/benchmark.json
    Results/evaluation/rag_llm_quality/benchmark.csv
    Results/evaluation/rag_llm_quality/repeated_questions.csv
    Results/evaluation/rag_llm_quality/latency.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from orchestrator.llm_provider import OllamaLLMProvider  # noqa: E402
from orchestrator.orchestrator import answer_question  # noqa: E402
from experiments.llm_regression.rag_llm_quality_questions import QUALITY_QUESTIONS  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "Results" / "evaluation" / "rag_llm_quality"

OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_TIMEOUT_SECONDS = 60.0

# The 4-question repeated-stability subset (task sec 8): one representative
# per DOCUMENTARY/DYNAMIC_DATA/HYBRID/TRAP category. Same provider config
# as the real `/chat` endpoint -- unseeded, Ollama's own default sampling
# -- so the measured variance is exactly what a real user would see across
# two identical questions asked minutes apart.
REPEATED_QIDS = ("A2", "B1", "C1", "E3")
REPEATED_RUNS = 3


def _run_one(qq, provider) -> Dict[str, Any]:
    start = time.monotonic()
    out = answer_question(qq.question, video_id=qq.video_id, window=qq.window, split=qq.split, llm=provider)
    elapsed = time.monotonic() - start
    response = out["response"]
    context = out["context"]
    return {
        "qid": qq.qid,
        "category": qq.category,
        "question": qq.question,
        "note": qq.note,
        "video_id": qq.video_id,
        "window": qq.window,
        "split": qq.split,
        "elapsed_seconds": round(elapsed, 3),
        "route_selected": out["route"]["category"],
        "tools_called": out["tool_plan"]["called"],
        "tools_not_called": out["tool_plan"]["not_called"],
        "answer": response["text"],
        "grounded": response["grounded"],
        "confidence": response["confidence"],
        "warnings": response["warnings"],
        "sources": [
            {"source": c.get("source"), "section": c.get("section"), "status": c.get("status")}
            for c in (context.get("document_context") or [])
        ],
        "dynamic_context": context.get("dynamic_context") or {},
        "n_sources": len(context.get("document_context") or []),
        "n_dynamic_results": len(context.get("dynamic_context") or {}),
    }


def run_main_benchmark(provider) -> List[Dict[str, Any]]:
    rows = []
    for qq in QUALITY_QUESTIONS:
        print(f"  [{qq.qid}] {qq.category:12s} {qq.question}", flush=True)
        row = _run_one(qq, provider)
        print(f"      -> route={row['route_selected']} grounded={row['grounded']} "
              f"elapsed={row['elapsed_seconds']}s n_sources={row['n_sources']} "
              f"n_dynamic={row['n_dynamic_results']}", flush=True)
        rows.append(row)
    return rows


def run_repeated_questions(provider) -> List[Dict[str, Any]]:
    by_qid = {qq.qid: qq for qq in QUALITY_QUESTIONS}
    rows = []
    for qid in REPEATED_QIDS:
        qq = by_qid[qid]
        for run_index in range(1, REPEATED_RUNS + 1):
            print(f"  [{qid} run {run_index}/{REPEATED_RUNS}] {qq.question}", flush=True)
            row = _run_one(qq, provider)
            row["run_index"] = run_index
            print(f"      -> route={row['route_selected']} grounded={row['grounded']} "
                  f"elapsed={row['elapsed_seconds']}s", flush=True)
            rows.append(row)
    return rows


def _write_benchmark_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "qid", "category", "question", "video_id", "window", "split",
        "route_selected", "grounded", "confidence", "elapsed_seconds",
        "n_sources", "n_dynamic_results", "n_warnings", "tools_called", "answer_excerpt",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "qid": row["qid"], "category": row["category"], "question": row["question"],
                "video_id": row["video_id"] or "", "window": row["window"] if row["window"] is not None else "",
                "split": row["split"], "route_selected": row["route_selected"], "grounded": row["grounded"],
                "confidence": row["confidence"], "elapsed_seconds": row["elapsed_seconds"],
                "n_sources": row["n_sources"], "n_dynamic_results": row["n_dynamic_results"],
                "n_warnings": len(row["warnings"]), "tools_called": ";".join(row["tools_called"]),
                "answer_excerpt": (row["answer"] or "")[:200].replace("\n", " "),
            })


def _write_repeated_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "qid", "run_index", "question", "route_selected", "grounded", "confidence",
        "elapsed_seconds", "n_sources", "n_dynamic_results", "answer_excerpt",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "qid": row["qid"], "run_index": row["run_index"], "question": row["question"],
                "route_selected": row["route_selected"], "grounded": row["grounded"],
                "confidence": row["confidence"], "elapsed_seconds": row["elapsed_seconds"],
                "n_sources": row["n_sources"], "n_dynamic_results": row["n_dynamic_results"],
                "answer_excerpt": (row["answer"] or "")[:300].replace("\n", " "),
            })


def _write_latency_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "category", "elapsed_seconds"])
        for row in rows:
            writer.writerow([row["qid"], row["category"], row["elapsed_seconds"]])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = OllamaLLMProvider(model=OLLAMA_MODEL, request_timeout_seconds=OLLAMA_TIMEOUT_SECONDS)

    print(f"Running {len(QUALITY_QUESTIONS)} main-benchmark questions against "
          f"provider=ollama model={OLLAMA_MODEL} (unseeded, real /chat config)...")
    main_rows = run_main_benchmark(provider)

    print(f"\nRunning repeated-question stability subset "
          f"({len(REPEATED_QIDS)} questions x {REPEATED_RUNS} runs)...")
    repeated_rows = run_repeated_questions(provider)

    benchmark = {
        "provider": "ollama", "model": OLLAMA_MODEL, "seed": None,
        "n_questions": len(main_rows), "rows": main_rows,
        "repeated_questions": {"qids": list(REPEATED_QIDS), "n_runs": REPEATED_RUNS, "rows": repeated_rows},
    }
    (OUTPUT_DIR / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _write_benchmark_csv(main_rows, OUTPUT_DIR / "benchmark.csv")
    _write_repeated_csv(repeated_rows, OUTPUT_DIR / "repeated_questions.csv")
    _write_latency_csv(main_rows, OUTPUT_DIR / "latency.csv")

    print(f"\n{len(main_rows)} main questions + {len(repeated_rows)} repeated runs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

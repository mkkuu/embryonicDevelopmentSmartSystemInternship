"""
Side-by-side summary of fixed-question benchmark artefacts
(`run_fixed_question_benchmark.py` output) across models/runs.

Reports ONLY what the artefacts contain mechanically: completion, errors,
provider-failure fallbacks (timeouts), the post-hoc grounding flag, latency
and answer length. It does NOT grade answers -- PASS/PARTIAL/FAIL against
each question's `success_criterion` remains a human reading, recorded in
docs/ (docs/RAG_FIXED_QUESTION_BENCHMARK.md's own rule). The grounding
count in particular is NOT a quality measure: a refusal that cites no
number passes trivially (docs/PROJECT_CHECKPOINT.md 2026-09-03b sec 5).

It also flags runs whose generation parameters differ (num_ctx, seed,
temperature, timeout, digest), because comparing two runs with different
settings without saying so is exactly what the protocol forbids.

Usage (from Training/):
    python -m orchestrator.summarize_fixed_question_runs
    python -m orchestrator.summarize_fixed_question_runs --answers Q7 Q9
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

DEFAULT_DIR = "../Results/evaluation/event_anomaly_rag_inventory"
FALLBACK_MARKER = "] No answer was generated for this turn."  # orchestrator._safe_fallback_response


def load_runs(directory: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "fixed_question_benchmark_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = path
        runs.append(data)
    return runs


def summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    rows = run.get("rows", [])
    answered = [r for r in rows if "answer" in r]
    errors = [r for r in rows if "error" in r]
    fallbacks = [r for r in answered if FALLBACK_MARKER in (r.get("answer") or "")]
    grounded = [r for r in answered if (r.get("grounding_detail") or {}).get("grounded")]
    elapsed = [r.get("elapsed_seconds", 0.0) for r in rows]
    lengths = [len(r.get("answer") or "") for r in answered if r not in fallbacks]
    provider = run.get("provider") or {}
    return {
        "file": os.path.basename(run["_path"]),
        "model": run.get("model"),
        "run_label": run.get("run_label", "?"),
        "complete": run.get("complete", "?"),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "n_fallbacks": len(fallbacks),
        "n_grounded": len(grounded),
        "elapsed_total_s": round(sum(elapsed), 1),
        "elapsed_max_s": round(max(elapsed), 1) if elapsed else 0.0,
        "answer_len_mean": round(sum(lengths) / len(lengths)) if lengths else 0,
        "timeout": run.get("timeout"),
        "num_ctx": provider.get("num_ctx", "?"),
        "seed": provider.get("seed", "?"),
        "temperature": provider.get("temperature", "?"),
        "digest": (provider.get("model_digest") or "?")[:12],
        "ollama": provider.get("ollama_version", "?"),
    }


def print_table(summaries: List[Dict[str, Any]]) -> None:
    cols = ["model", "run_label", "complete", "n_rows", "n_errors", "n_fallbacks", "n_grounded",
            "elapsed_total_s", "elapsed_max_s", "answer_len_mean", "timeout", "num_ctx", "seed",
            "temperature", "digest", "ollama"]
    widths = {c: max(len(c), *(len(str(s[c])) for s in summaries)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for s in summaries:
        print(" | ".join(str(s[c]).ljust(widths[c]) for c in cols))


def print_comparability_warnings(summaries: List[Dict[str, Any]]) -> None:
    unrecorded = [s["file"] for s in summaries if s["num_ctx"] == "?"]
    if unrecorded:
        print(f"NOTE: {len(unrecorded)} artefact(s) predate provider metadata (num_ctx/seed/temperature/"
              f"digest unrecorded): {', '.join(unrecorded)} -- their settings were reconstructed in "
              f"docs/PROJECT_CHECKPOINT.md (2026-09-03b), not stored.")
    for key in ("timeout", "num_ctx", "seed", "temperature", "ollama"):
        values = {str(s[key]) for s in summaries if str(s[key]) != "?"}
        if len(values) > 1:
            print(f"WARNING: runs differ on {key}: {sorted(values)} -- not directly comparable "
                  f"without stating it.")


def print_per_question(runs: List[Dict[str, Any]], question_ids: List[str]) -> None:
    for qid in question_ids:
        print(f"\n=== {qid} ===")
        for run in runs:
            row = next((r for r in run.get("rows", []) if r.get("question_id") == qid), None)
            label = f"{run.get('model')} / {run.get('run_label', '?')}"
            if row is None:
                print(f"--- {label}: (absent)")
                continue
            if "error" in row:
                print(f"--- {label}: ERROR {row['error']}")
                continue
            grounded = (row.get("grounding_detail") or {}).get("grounded")
            print(f"--- {label}: elapsed={row.get('elapsed_seconds', 0):.1f}s grounded={grounded}")
            print(row.get("answer", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--answers", nargs="*", default=[],
                         help="Question ids whose raw answers to print side by side (e.g. Q7 Q9).")
    args = parser.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        print(f"No fixed_question_benchmark_*.json under {args.dir}")
        return 1
    summaries = [summarize(r) for r in runs]
    print_table(summaries)
    print_comparability_warnings(summaries)
    print("\nNOTE: n_grounded is a mechanical flag, not a quality score; grading is manual "
          "(docs/RAG_FIXED_QUESTION_BENCHMARK.md).")
    if args.answers:
        print_per_question(runs, [q.upper() for q in args.answers])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

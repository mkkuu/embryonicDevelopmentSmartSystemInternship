"""
READ-ONLY retrospective validation of a change to `grounding_check.py`.

Replays the answers ALREADY STORED in fixed-question benchmark artefacts
through the CURRENT checker and compares the verdict with the one recorded
in the artefact at run time. No LLM, no GPU, no network, no re-run: the
answers, contexts and stored verdicts are read verbatim and nothing is
written back.

What it is for: a change to the grounding mechanism must be shown to
remove the intended FALSE NEGATIVES without removing any TRUE POSITIVE.
The four `prompt_v2_reading_method*` artefacts of 2026-09-07c are the
natural corpus -- 60 real answers produced on byte-identical context.

What it is NOT for: the manual PASS/PARTIAL/FAIL score of the 15 frozen
questions. `grounded` has never been a quality measure
(docs/PROJECT_CHECKPOINT.md 2026-09-03b sec 5) and this script never
touches the grading artefact.

Usage (from Training/):
    python -m experiments.rag_llm_diagnostics.validate_grounding_change
    python -m experiments.rag_llm_diagnostics.validate_grounding_change --show Q4 Q6 Q11
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

from orchestrator import grounding_check

DEFAULT_DIR = "../Results/evaluation/event_anomaly_rag_inventory"
DEFAULT_PATTERN = "fixed_question_benchmark_mistral-nemo_12b_prompt_v2_reading_method*.json"


def _context_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuilds exactly the context the checker saw at run time. Every field
    was stored in the artefact by `run_fixed_question_benchmark.py`."""
    return {
        "question": row.get("question", ""),
        "document_context": row.get("document_context", []) or [],
        "dynamic_context": row.get("dynamic_context", {}) or {},
        "warnings": row.get("context_warnings", []) or [],
    }


def _reasons(detail: Dict[str, Any]) -> str:
    bits = []
    for key, label in (("ungrounded_numbers", "num"), ("ungrounded_phase_tokens", "phase"),
                       ("ungrounded_transition_claims", "pair"), ("skip_claim_mismatches", "skip")):
        value = detail.get(key) or []
        if value:
            if key == "ungrounded_transition_claims":
                value = [f"{c['from_phase']}->{c['to_phase']}" for c in value]
            elif key == "skip_claim_mismatches":
                value = [f"{c['from_phase']}->{c['to_phase']}" for c in value]
            bits.append(f"{label}={value}")
    return " ".join(bits) or "-"


def compare(directory: str, pattern: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path, encoding="utf-8") as f:
            artefact = json.load(f)
        label = artefact.get("run_label") or os.path.basename(path)
        for row in artefact.get("rows", []):
            if "answer" not in row:
                continue
            old = row.get("grounding_detail", {}) or {}
            new = grounding_check.check_grounding(row["answer"], _context_of(row)).to_dict()
            rows.append({
                "run": label, "question_id": row["question_id"],
                "old_grounded": bool(old.get("grounded")), "new_grounded": bool(new.get("grounded")),
                "old_reasons": _reasons(old), "new_reasons": _reasons(new),
                "provenance": new.get("transition_claim_provenance", []),
                "answer": row["answer"],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--show", nargs="*", default=[],
                        help="Question ids whose answers to print in full for inspection.")
    args = parser.parse_args()

    rows = compare(args.dir, args.pattern)
    if not rows:
        print(f"No artefact matched {args.pattern} in {args.dir}")
        return 1

    changed = [r for r in rows if r["old_grounded"] != r["new_grounded"]]
    freed = [r for r in changed if r["new_grounded"]]
    tightened = [r for r in changed if not r["new_grounded"]]
    still_flagged = [r for r in rows if not r["old_grounded"] and not r["new_grounded"]]

    print(f"answers replayed: {len(rows)}  "
          f"grounded before: {sum(r['old_grounded'] for r in rows)}  "
          f"after: {sum(r['new_grounded'] for r in rows)}")
    print(f"changed: {len(changed)}  (ungrounded -> grounded: {len(freed)}, "
          f"grounded -> ungrounded: {len(tightened)})")
    print(f"still flagged after the change: {len(still_flagged)}")

    print("\n--- verdict changes ---")
    if not changed:
        print("(none)")
    for r in sorted(changed, key=lambda r: (r["question_id"], r["run"])):
        arrow = "FLAGGED -> grounded" if r["new_grounded"] else "grounded -> FLAGGED"
        print(f"{r['question_id']:4} {r['run'][-4:]:5} {arrow}   old[{r['old_reasons']}]  new[{r['new_reasons']}]")

    print("\n--- still flagged (true positives preserved) ---")
    for r in sorted(still_flagged, key=lambda r: (r["question_id"], r["run"])):
        print(f"{r['question_id']:4} {r['run'][-4:]:5} {r['new_reasons']}")

    print("\n--- per-question summary (grounded count, out of 4 runs) ---")
    ids = sorted({r["question_id"] for r in rows}, key=lambda q: int(q[1:]))
    for qid in ids:
        subset = [r for r in rows if r["question_id"] == qid]
        before = sum(r["old_grounded"] for r in subset)
        after = sum(r["new_grounded"] for r in subset)
        mark = "  <-- changed" if before != after else ""
        print(f"{qid:4} before={before}/{len(subset)}  after={after}/{len(subset)}{mark}")

    for qid in [q.upper() for q in args.show]:
        print(f"\n===== {qid} =====")
        for r in [x for x in rows if x["question_id"] == qid]:
            print(f"--- {r['run']}  old_grounded={r['old_grounded']} new_grounded={r['new_grounded']} "
                  f"provenance={r['provenance']}")
            print(f"    old[{r['old_reasons']}] new[{r['new_reasons']}]")
            print("   ", r["answer"][:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Mechanical aggregation of the paired V2 vs COMPACT_CONTEXT_V1 experiment
(`run_compact_context_experiment.py`) once the answers have been graded by hand.

Grading stays manual, by protocol: the rubric is each question's frozen
`success_criterion`, read against the real answer (the discipline of every
prior LLM-quality evaluation in this project). This script NEVER assigns a
grade, never reads an answer's text, and never decides whether something is a
hallucination -- it reads a grades file written by the grader and does only
arithmetic and counting.

    PASS = 1.0   PARTIAL = 0.5   FAIL = 0.0
    delta = score(COMPACT) - score(V2)

Grades file shape (JSON):

    {"grader": "...", "rubric": "...",
     "grades": {
       "Q1": {
         "V2":      [{"class": "FAIL", "hallucination": true, "honest_limitation": false,
                      "provenance_error": false, "model_as_observed": false,
                      "temporal_error": false, "refusal": false, "note": "..."}, ...],
         "COMPACT": [ ... same shape, one entry per replication ... ],
         "note": "..."}, ...}}

one entry per replication, in replication order, for each condition. The
mechanical fields (grounded, latency, prompt size) are read from the run
artefact, never from the grades file, so a grader cannot move them.

Usage (from Training/):
    python -m experiments.context_ablations.summarize_compact_context_experiment \
        --artifact ../Results/evaluation/compact_context_experiment/paired_..._compact_v1.json \
        --grades   ../Results/evaluation/compact_context_experiment/grades_compact_v1.json \
        --out      ../Results/evaluation/compact_context_experiment/analysis_compact_v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict, List, Optional

from experiments.context_ablations.run_compact_context_experiment import CONDITION_COMPACT, CONDITION_V2

GRADE_VALUES = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}
CONDITIONS = (CONDITION_V2, CONDITION_COMPACT)

# Error flags the grader sets per answer. Counted, never inferred here.
ERROR_FLAGS = ("hallucination", "honest_limitation", "provenance_error",
               "model_as_observed", "temporal_error", "refusal")


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def _stats(values: List[float]) -> Dict[str, Any]:
    return {
        "per_replication": [round(v, 4) for v in values],
        "mean": _mean(values),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--grades", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.artifact, encoding="utf-8") as f:
        artifact = json.load(f)
    with open(args.grades, encoding="utf-8") as f:
        grading = json.load(f)
    grades = grading["grades"]
    n_reps = artifact["replications"]

    totals = {c: [0.0] * n_reps for c in CONDITIONS}
    graded = {c: [0] * n_reps for c in CONDITIONS}
    flags = {c: {flag: 0 for flag in ERROR_FLAGS} for c in CONDITIONS}
    classes = {c: {"PASS": 0, "PARTIAL": 0, "FAIL": 0} for c in CONDITIONS}
    grounded = {c: 0 for c in CONDITIONS}
    grounded_not_pass = {c: 0 for c in CONDITIONS}
    latency = {c: [] for c in CONDITIONS}
    answers_seen = {c: 0 for c in CONDITIONS}
    prompt_chars = {c: [] for c in CONDITIONS}
    per_question = []

    for row in artifact["rows"]:
        qid = row["question_id"]
        entry = grades.get(qid, {})
        prompt_chars[CONDITION_V2].append(row["v2_user_prompt_chars"])
        prompt_chars[CONDITION_COMPACT].append(row["compact_user_prompt_chars"])

        per_condition_scores: Dict[str, Optional[float]] = {}
        for condition in CONDITIONS:
            condition_grades = entry.get(condition, [])
            values = []
            for index, grade_entry in enumerate(condition_grades[:n_reps]):
                grade = grade_entry.get("class") if isinstance(grade_entry, dict) else grade_entry
                if grade not in GRADE_VALUES:
                    continue
                totals[condition][index] += GRADE_VALUES[grade]
                graded[condition][index] += 1
                classes[condition][grade] += 1
                values.append(GRADE_VALUES[grade])
                if isinstance(grade_entry, dict):
                    for flag in ERROR_FLAGS:
                        if grade_entry.get(flag):
                            flags[condition][flag] += 1
            per_condition_scores[condition] = _mean(values)

            # mechanical fields, straight from the run artefact
            for replication in row["replications"]:
                answer = replication.get(condition) or {}
                if "answer" not in answer:
                    continue
                answers_seen[condition] += 1
                latency[condition].append(answer.get("elapsed_seconds", 0.0))
                if answer.get("grounded"):
                    grounded[condition] += 1
                    index = replication["replication"] - 1
                    condition_grades = entry.get(condition, [])
                    if index < len(condition_grades):
                        graded_entry = condition_grades[index]
                        grade = (graded_entry.get("class") if isinstance(graded_entry, dict)
                                 else graded_entry)
                        if grade != "PASS":
                            grounded_not_pass[condition] += 1

        v2_mean = per_condition_scores[CONDITION_V2]
        compact_mean = per_condition_scores[CONDITION_COMPACT]
        delta = (round(compact_mean - v2_mean, 4)
                 if v2_mean is not None and compact_mean is not None else None)
        verdict = ("NON NOTE" if delta is None else
                   "AMELIOREE" if delta > 0 else "DEGRADEE" if delta < 0 else "INCHANGEE")

        per_question.append({
            "question_id": qid,
            "question": row["question"],
            "type": row["type"],
            "n_rag_chunks": row["compaction_stats"]["n_chunks"],
            "n_rag_chunks_dropped": row["compaction_stats"]["n_chunks_dropped"],
            "n_narrative_lines_dropped": row["compaction_stats"]["n_lines_dropped"],
            "rag_chars_v2": row["compaction_stats"]["original_content_chars"],
            "rag_chars_compact": row["compaction_stats"]["kept_content_chars"],
            "rules_applied": row["compaction_stats"]["rules_applied"],
            "v2_prompt_chars": row["v2_user_prompt_chars"],
            "compact_prompt_chars": row["compact_user_prompt_chars"],
            "prompt_delta_fraction": row["prompt_delta_fraction"],
            "grades_v2": [g.get("class") if isinstance(g, dict) else g
                          for g in entry.get(CONDITION_V2, [])],
            "grades_compact": [g.get("class") if isinstance(g, dict) else g
                               for g in entry.get(CONDITION_COMPACT, [])],
            "score_v2_mean": v2_mean,
            "score_compact_mean": compact_mean,
            "delta": delta,
            "verdict": verdict,
            "note": entry.get("note", ""),
        })

    deltas = [c - v for v, c in zip(totals[CONDITION_V2], totals[CONDITION_COMPACT])]
    n_questions = len(artifact["rows"])
    incomplete = {c: [n_questions - n for n in counts] for c, counts in graded.items()
                  if any(n < n_questions for n in counts)}

    summary = {
        "artifact": args.artifact,
        "grades_file": args.grades,
        "grader": grading.get("grader"),
        "rubric": grading.get("rubric"),
        "model": artifact["provider"]["model"],
        "single_variable": artifact["single_variable"],
        "conditions": {
            "num_ctx": artifact["provider"]["num_ctx"],
            "seed": artifact["provider"]["seed"],
            "temperature": artifact["provider"]["temperature"],
            "timeout_seconds": artifact["provider"]["request_timeout_seconds"],
            "placement": artifact["provider"].get("placement"),
            "ollama_version": artifact["provider"].get("ollama_version"),
            "model_digest": artifact["provider"].get("model_digest"),
        },
        "n_questions": n_questions,
        "replications": n_reps,
        "score_V2": _stats(totals[CONDITION_V2]),
        "score_COMPACT": _stats(totals[CONDITION_COMPACT]),
        "delta_per_replication": _stats(deltas),
        "classes": classes,
        "error_flags": flags,
        "grounding": {
            c: {"grounded_true": grounded[c], "of": answers_seen[c],
                "grounded_but_not_PASS": grounded_not_pass[c]} for c in CONDITIONS},
        "latency_seconds": {c: _stats(latency[c]) for c in CONDITIONS},
        "prompt_chars": {
            c: {"total": sum(prompt_chars[c]), "mean": _mean([float(v) for v in prompt_chars[c]]),
                "max": max(prompt_chars[c]) if prompt_chars[c] else None} for c in CONDITIONS},
        "rag_chars": {
            "v2": sum(p["rag_chars_v2"] for p in per_question),
            "compact": sum(p["rag_chars_compact"] for p in per_question),
        },
        "n_graded_per_replication": graded,
        "ungraded_questions_per_replication": incomplete or None,
        "counts": {
            "improved": sum(1 for p in per_question if p["verdict"] == "AMELIOREE"),
            "degraded": sum(1 for p in per_question if p["verdict"] == "DEGRADEE"),
            "unchanged": sum(1 for p in per_question if p["verdict"] == "INCHANGEE"),
            "not_graded": sum(1 for p in per_question if p["verdict"] == "NON NOTE"),
        },
        "questions_with_rag": [p["question_id"] for p in per_question if p["n_rag_chunks"]],
        "per_question": per_question,
        "stochasticity_warning":
            "The LLM is stochastic and no seed is set, by protocol. With this many "
            "replications a difference of the order of the run-to-run spread is NOT evidence "
            "of an effect. Read delta_per_replication's sd and min/max before concluding.",
    }

    header = f"{'Q':<5}{'type':<15}{'ragΔ':>8}{'promptΔ':>9}{'V2':>7}{'COMPACT':>9}{'delta':>8}  verdict"
    print(header)
    print("-" * len(header))
    for p in per_question:
        rag_delta = (f"{p['rag_chars_compact'] - p['rag_chars_v2']:+d}" if p["n_rag_chunks"] else "-")
        print(f"{p['question_id']:<5}{p['type']:<15}{rag_delta:>8}"
              f"{p['prompt_delta_fraction']:>8.1%}"
              f"{p['score_v2_mean'] if p['score_v2_mean'] is not None else '-':>7}"
              f"{p['score_compact_mean'] if p['score_compact_mean'] is not None else '-':>9}"
              f"{p['delta'] if p['delta'] is not None else '-':>8}  {p['verdict']}")
    print("-" * len(header))
    print(f"V2      /15 : {summary['score_V2']}")
    print(f"COMPACT /15 : {summary['score_COMPACT']}")
    print(f"DELTA       : {summary['delta_per_replication']}")
    print(f"classes     : {classes}")
    print(f"flags       : {flags}")
    print(f"grounding   : {summary['grounding']}")
    if summary["ungraded_questions_per_replication"]:
        print(f"\nATTENTION - notes manquantes : {summary['ungraded_questions_per_replication']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

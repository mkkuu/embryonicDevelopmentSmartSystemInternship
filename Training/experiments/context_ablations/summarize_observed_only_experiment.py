"""
Mechanical aggregation of the paired FULL vs OBSERVED-ONLY experiment
(`run_observed_only_experiment.py`) once the answers have been graded by hand.

Grading stays manual, by protocol: the rubric is each question's frozen
`success_criterion`, read against the real answer (the same discipline as every
prior LLM-quality evaluation in this project -- `docs/RAG_LLM_QUALITY_REPORT.md`,
"automating this would reintroduce NLP complexe"). This script NEVER assigns a
grade. It reads a grades file written by the grader and does only arithmetic:

    PASS = 1.0   PARTIAL = 0.5   FAIL = 0.0
    delta = score(OBSERVED_ONLY) - score(FULL)

Grades file shape (JSON):

    {"grader": "...", "rubric": "...",
     "grades": {"Q1": {"FULL": ["FAIL", "PARTIAL", "FAIL"],
                       "OBSERVED_ONLY": ["PARTIAL", "PARTIAL", "PASS"],
                       "note": "..."}, ...}}

one grade per replication, in replication order, for each condition.

Usage (from Training/):
    python -m experiments.context_ablations.summarize_observed_only_experiment \
        --artifact ../Results/evaluation/observed_only_experiment/paired_..._paired_v1.json \
        --grades   ../Results/evaluation/observed_only_experiment/grades_paired_v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List, Optional

from experiments.context_ablations.observed_only import (
    CONDITION_FULL,
    CONDITION_OBSERVED_ONLY,
    NOT_COMPUTABLE,
)

GRADE_VALUES = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}


def _score(grades: List[str]) -> Optional[float]:
    values = [GRADE_VALUES[g] for g in grades if g in GRADE_VALUES]
    return sum(values) if values else None


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--grades", required=True)
    parser.add_argument("--out", default=None, help="Write the summary JSON here as well.")
    args = parser.parse_args()

    with open(args.artifact, encoding="utf-8") as f:
        artifact = json.load(f)
    with open(args.grades, encoding="utf-8") as f:
        grading = json.load(f)
    grades = grading["grades"]

    n_reps = artifact["replications"]
    per_question = []
    totals: Dict[str, List[float]] = {CONDITION_FULL: [0.0] * n_reps,
                                      CONDITION_OBSERVED_ONLY: [0.0] * n_reps}
    # Counted separately from the totals so an UNGRADED question is visible as
    # ungraded instead of silently contributing 0 -- a missing grade is not a
    # FAIL, and a question that is NOT COMPUTABLE in one condition is not a FAIL
    # either. Both must be graded by the human, or reported as not graded.
    graded: Dict[str, List[int]] = {CONDITION_FULL: [0] * n_reps,
                                    CONDITION_OBSERVED_ONLY: [0] * n_reps}

    for row in artifact["rows"]:
        qid = row["question_id"]
        entry = grades.get(qid, {})
        full_grades = entry.get(CONDITION_FULL, [])
        observed_grades = entry.get(CONDITION_OBSERVED_ONLY, [])
        for condition, condition_grades in ((CONDITION_FULL, full_grades),
                                            (CONDITION_OBSERVED_ONLY, observed_grades)):
            for index, grade in enumerate(condition_grades[:n_reps]):
                if grade not in GRADE_VALUES:
                    continue
                totals[condition][index] += GRADE_VALUES[grade]
                graded[condition][index] += 1

        full_mean = _mean([GRADE_VALUES[g] for g in full_grades if g in GRADE_VALUES])
        observed_mean = _mean([GRADE_VALUES[g] for g in observed_grades if g in GRADE_VALUES])
        delta = (round(observed_mean - full_mean, 4)
                 if full_mean is not None and observed_mean is not None else None)
        if delta is None:
            verdict = "NON NOTE"
        elif delta > 0:
            verdict = "AMELIOREE"
        elif delta < 0:
            verdict = "DEGRADEE"
        else:
            verdict = "INCHANGEE"

        per_question.append({
            "question_id": qid,
            "question": row["question"],
            "type": row["type"],
            "observed_only_computability": row["observed_only_computability"],
            "n_removed_paths": row["n_removed_paths"],
            "full_prompt_chars": row["full_user_prompt_chars"],
            "observed_only_prompt_chars": row["observed_only_user_prompt_chars"],
            "grades_full": full_grades,
            "grades_observed_only": observed_grades,
            "score_full_mean": full_mean,
            "score_observed_only_mean": observed_mean,
            "delta": delta,
            "verdict": verdict,
            "note": entry.get("note", ""),
        })

    full_totals = totals[CONDITION_FULL]
    observed_totals = totals[CONDITION_OBSERVED_ONLY]
    deltas = [o - f for f, o in zip(full_totals, observed_totals)]

    def _stats(values: List[float]) -> Dict:
        return {
            "per_replication": [round(v, 4) for v in values],
            "mean": _mean(values),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    not_computable = [p["question_id"] for p in per_question
                      if p["observed_only_computability"] == NOT_COMPUTABLE]
    computable = [p for p in per_question if p["question_id"] not in not_computable]

    n_questions = len(artifact["rows"])
    incomplete = {
        condition: [n_questions - n for n in counts]
        for condition, counts in graded.items()
        if any(n < n_questions for n in counts)
    }

    summary = {
        "artifact": args.artifact,
        "grades_file": args.grades,
        "grader": grading.get("grader"),
        "rubric": grading.get("rubric"),
        "model": artifact["provider"]["model"],
        "conditions": {"num_ctx": artifact["provider"]["num_ctx"],
                       "seed": artifact["provider"]["seed"],
                       "temperature": artifact["provider"]["temperature"],
                       "timeout_seconds": artifact["provider"]["request_timeout_seconds"],
                       "placement": artifact["provider"].get("placement")},
        "n_questions": artifact["n_questions"],
        "replications": n_reps,
        # (A) COMPLETE score: over all 15 questions, the questions that are not
        # computable in OBSERVED-ONLY explicitly INCLUDED and graded as they were
        # answered. Never auto-converted to FAIL.
        "score_complete_FULL": _stats(full_totals),
        "score_complete_OBSERVED_ONLY": _stats(observed_totals),
        "delta_complete": _stats(deltas),
        "n_graded_per_replication": graded,
        "ungraded_questions_per_replication": incomplete or None,
        "scoring_note": "A question pre-declared NOT COMPUTABLE in OBSERVED-ONLY is still graded "
                        "from the answer actually produced, against the same frozen "
                        "success_criterion. It is never turned into a FAIL by this script, and a "
                        "MISSING grade is never counted as 0 -- it is reported under "
                        "ungraded_questions_per_replication instead.",
        "questions_not_computable_in_observed_only": not_computable,
        # (B) RESTRICTED score: only the questions genuinely comparable between
        # the two conditions, i.e. those whose answer does not require a model
        # prediction the OBSERVED-ONLY arm cannot have.
        "score_restricted_to_comparable_questions": {
            "question_ids": [p["question_id"] for p in computable],
            "n": len(computable),
            "score_full_sum": round(sum(p["score_full_mean"] for p in computable
                                        if p["score_full_mean"] is not None), 4),
            "score_observed_only_sum": round(sum(p["score_observed_only_mean"] for p in computable
                                                 if p["score_observed_only_mean"] is not None), 4),
            "score_full_mean_per_question": _mean([p["score_full_mean"] for p in computable
                                                   if p["score_full_mean"] is not None]),
            "score_observed_only_mean_per_question": _mean(
                [p["score_observed_only_mean"] for p in computable
                 if p["score_observed_only_mean"] is not None]),
        },
        "counts": {
            "improved": sum(1 for p in per_question if p["verdict"] == "AMELIOREE"),
            "degraded": sum(1 for p in per_question if p["verdict"] == "DEGRADEE"),
            "unchanged": sum(1 for p in per_question if p["verdict"] == "INCHANGEE"),
            "not_graded": sum(1 for p in per_question if p["verdict"] == "NON NOTE"),
        },
        "per_question": per_question,
    }

    header = f"{'Q':<5}{'type':<15}{'calc.':<26}{'FULL':>7}{'OBS':>7}{'delta':>8}  verdict"
    print(header)
    print("-" * len(header))
    for p in per_question:
        print(f"{p['question_id']:<5}{p['type']:<15}{p['observed_only_computability']:<26}"
              f"{p['score_full_mean'] if p['score_full_mean'] is not None else '-':>7}"
              f"{p['score_observed_only_mean'] if p['score_observed_only_mean'] is not None else '-':>7}"
              f"{p['delta'] if p['delta'] is not None else '-':>8}  {p['verdict']}")
    print("-" * len(header))
    print(f"(A) COMPLET /15  FULL          : {summary['score_complete_FULL']}")
    print(f"(A) COMPLET /15  OBSERVED-ONLY : {summary['score_complete_OBSERVED_ONLY']}")
    print(f"(A) DELTA                      : {summary['delta_complete']}")
    restricted = summary["score_restricted_to_comparable_questions"]
    print(f"(B) RESTREINT ({restricted['n']} questions {restricted['question_ids']})")
    print(f"    FULL={restricted['score_full_sum']}  "
          f"OBSERVED-ONLY={restricted['score_observed_only_sum']}")
    if summary["ungraded_questions_per_replication"]:
        print(f"\nATTENTION - notes manquantes : "
              f"{summary['ungraded_questions_per_replication']} "
              f"(non comptees comme FAIL ; a noter avant toute conclusion)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

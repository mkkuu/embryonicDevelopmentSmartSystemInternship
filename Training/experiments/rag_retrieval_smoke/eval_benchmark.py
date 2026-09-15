"""
Retrieval smoke benchmark (task Phase 12) -- a HAND-LABELED set of ~10
real project questions, NOT a claim of statistically robust retrieval
evaluation (the task's own explicit instruction: "ne pas prétendre à
une évaluation scientifique robuste avec seulement quelques questions").

For each question: a hand-judged set of "relevant" document_ids (the
documents a competent answer would need to cite), checked against the
real top-k retrieval result from the real, currently-ingested corpus.

Metrics (task's own minimum ask):
- Precision@k: fraction of the top-k results whose document_id is in
  the relevant set.
- Recall@k: fraction of the relevant set that appears somewhere in the
  top-k results -- meaningful here because every question's relevant
  set is small (1-3 documents) and exhaustively hand-labeled, not
  estimated.
- MRR: reciprocal rank of the FIRST relevant result (1.0 if the very
  top result is relevant, 0 if no relevant result appears in top-k).

CLI:
    cd Training && python -m experiments.rag_retrieval_smoke.eval_benchmark
"""

from __future__ import annotations

import json
from typing import Dict, List

from rag.retrieval import retrieve

TOP_K = 5

# Each entry: (question, {relevant document_ids}, note on why -- the note
# is what makes this benchmark auditable rather than a black-box pass/fail.
BENCHMARK: List[Dict] = [
    {
        "question": "Qu'est-ce que le Semi-HMM ?",
        "relevant": {"handoff_semi_hmm", "hmm_research_plan"},
        "note": "A definitional answer needs the Semi-HMM design/motivation doc and/or the HMM research plan that introduces it relative to the plain HMM.",
    },
    {
        "question": "Pourquoi utiliser un modele de duree explicite ?",
        "relevant": {"handoff_semi_hmm", "hmm_research_plan", "scientific_results", "scientific_report"},
        "note": "The duration-heterogeneity motivation is documented in the Semi-HMM handoff, the HMM research plan's own decision criterion, and the consolidated scientific narrative.",
    },
    {
        "question": "Quelle est la difference entre GRU et Identity Dynamics ?",
        "relevant": {"gru_identity_analysis", "gru_identity_executive_summary"},
        "note": "Direct, authoritative comparison -- exactly what these two documents exist to answer.",
    },
    {
        "question": "Pourquoi le GRU peut-il etre utilise comme signal pour le RAG ?",
        "relevant": {"gru_identity_analysis", "gru_identity_executive_summary", "product_architecture"},
        "note": "GRU_IDENTITY_ANALYSIS.md §10 is the primary source; PRODUCT_ARCHITECTURE.md §4a records the same conclusion in the architecture.",
    },
    {
        "question": "Pourquoi le Test est-il verrouille ?",
        "relevant": {"handoff", "scientific_report", "reproducibility"},
        "note": "The Test-lock rationale (one pre-registered final evaluation, never used for iterative development) is methodology documented in HANDOFF.md/SCIENTIFIC_REPORT.md/REPRODUCIBILITY.md, not a single dedicated document.",
    },
    {
        "question": "Quelles sont les limites du Semi-HMM ?",
        "relevant": {"scientific_report", "gru_identity_analysis", "reporting_api"},
        "note": "Known Semi-HMM limitations (calibration, tPB2/tEB censoring) are documented in SCIENTIFIC_REPORT.md and REPORTING_API.md's Known Limitations; GRU_IDENTITY_ANALYSIS.md also restates the calibration finding for context.",
    },
    {
        "question": "Qu'est-ce que la calibration ECE ?",
        "relevant": {"semi_hmm_phase_1_6_report", "gru_identity_analysis"},
        "note": "ECE is defined/used concretely in the Phase 1.6 report's metric definitions and the GRU analysis's own metric-definitions section.",
    },
    {
        "question": "Comment fonctionne le cache du Reporting API ?",
        "relevant": {"reporting_cache", "reporting_cache_implementation_report"},
        "note": "Direct match -- these two documents exist specifically to answer this.",
    },
    {
        "question": "Quel est le modele de reference actuel du projet ?",
        "relevant": {"product_architecture", "product_roadmap", "scientific_report"},
        "note": "The current reference model (Semi-HMM k=7) and its status are stated in the living architecture/roadmap docs and the scientific report's own current-best-approach section.",
    },
    {
        "question": "Pourquoi Identity Dynamics a-t-il ete reevalue sur Val ?",
        "relevant": {"gru_identity_analysis", "gru_identity_executive_summary"},
        "note": "The Test-vs-Val split-mismatch correction is a central, explicit methodological point of the GRU analysis documents specifically.",
    },
]


def precision_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    top = set(retrieved_ids[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant: set) -> float:
    for i, d in enumerate(retrieved_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def run_benchmark(top_k: int = TOP_K) -> dict:
    rows = []
    for item in BENCHMARK:
        results = retrieve(item["question"], top_k=top_k)
        retrieved_ids = [r["metadata"]["document_id"] for r in results]
        p = precision_at_k(retrieved_ids, item["relevant"], top_k)
        r = recall_at_k(retrieved_ids, item["relevant"], top_k)
        mrr = reciprocal_rank(retrieved_ids, item["relevant"])
        rows.append({
            "question": item["question"], "relevant": sorted(item["relevant"]),
            "retrieved": retrieved_ids, "precision_at_k": p, "recall_at_k": r,
            "reciprocal_rank": mrr, "note": item["note"],
        })
    mean_p = sum(r["precision_at_k"] for r in rows) / len(rows)
    mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
    mean_mrr = sum(r["reciprocal_rank"] for r in rows) / len(rows)
    return {"top_k": top_k, "n_questions": len(rows), "rows": rows,
            "mean_precision_at_k": mean_p, "mean_recall_at_k": mean_r, "mrr": mean_mrr}


def main() -> None:
    report = run_benchmark()
    for row in report["rows"]:
        print(f"Q: {row['question']}")
        print(f"   relevant={row['relevant']}")
        print(f"   retrieved={row['retrieved']}")
        print(f"   P@{report['top_k']}={row['precision_at_k']:.2f}  R@{report['top_k']}={row['recall_at_k']:.2f}  RR={row['reciprocal_rank']:.2f}")
        print()
    print(f"Mean P@{report['top_k']} = {report['mean_precision_at_k']:.3f}")
    print(f"Mean R@{report['top_k']} = {report['mean_recall_at_k']:.3f}")
    print(f"MRR = {report['mrr']:.3f}")
    with open("rag_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

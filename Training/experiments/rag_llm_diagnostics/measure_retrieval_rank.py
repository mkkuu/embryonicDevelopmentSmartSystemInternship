"""
READ-ONLY measurement of where the canonical phase-order chunk lands for a
fixed-question benchmark question, before and after the 2026-09-07e query
enrichment.

Runs the REAL retrieval path (`plan_tools` -> `execute_plan` ->
`tools.retrieve_documents`) against the REAL index, with the real Reporting
API tools supplying the turn's phase vocabulary. It never calls an LLM,
never writes to the index, and never writes an artefact: it prints a
comparison and exits.

"Before" is obtained by asking the retriever the BARE question -- exactly
what `execute_plan` sent prior to this change -- and "after" by asking it
`build_retrieval_query()`'s output for the same turn. Same corpus, same
index, same embedding model, same `top_k`.

Usage (from Training/, CPU is enough):
    CUDA_VISIBLE_DEVICES="" python -m experiments.rag_llm_diagnostics.measure_retrieval_rank
    CUDA_VISIBLE_DEVICES="" python -m experiments.rag_llm_diagnostics.measure_retrieval_rank --questions Q11 Q12 --probe-depth 25
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from orchestrator import tools
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import build_retrieval_query, execute_plan, plan_tools
from orchestrator.router import classify

PRODUCTION_TOP_K = 5
_ORDER_HEAD = ("tPB2", "tPNa", "tPNf")
_ORDER_TAIL = ("tSB", "tB", "tEB")


def carries_canonical_order(content: str) -> bool:
    """True when a chunk spells the phase taxonomy out as an ORDERED sequence:
    the three opening names in order, followed somewhere by a closing name.
    Deliberately structural -- it never matches a chunk that merely mentions a
    phase name in passing."""
    content = content or ""
    if not all(p in content for p in _ORDER_HEAD):
        return False
    start = content.index("tPB2")
    if not (content.index("tPNa") > start and content.index("tPNf") > content.index("tPNa")):
        return False
    return any(p in content[content.index("tPNf"):] for p in _ORDER_TAIL)


def rank_of_canonical_chunk(query: str, probe_depth: int) -> Optional[Dict]:
    results = tools.retrieve_documents(query, top_k=probe_depth)
    for rank, result in enumerate(results, 1):
        if carries_canonical_order(result.get("content") or ""):
            return {"rank": rank, "score": result.get("score"),
                    "source": (result.get("metadata") or {}).get("source"),
                    "section": (result.get("metadata") or {}).get("section")}
    return None


def top_k_sources(query: str, top_k: int = PRODUCTION_TOP_K) -> List[str]:
    return [f"{(r.get('metadata') or {}).get('source')} :: "
            f"{str((r.get('metadata') or {}).get('section'))[:44]}"
            f"{'  <<< CANONICAL ORDER' if carries_canonical_order(r.get('content') or '') else ''}"
            for r in tools.retrieve_documents(query, top_k=top_k)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", nargs="*", default=["Q11", "Q12"])
    parser.add_argument("--probe-depth", type=int, default=25,
                        help="How deep to look for the canonical chunk. Production top_k stays 5.")
    args = parser.parse_args()

    by_id = {q.question_id: q for q in FIXED_QUESTIONS}
    summary = []
    for qid in [q.upper() for q in args.questions]:
        question = by_id[qid]
        plan = plan_tools(question.question, classify(question.question),
                          video_id=question.video_id, window=question.window)
        # Real tool calls -- this is the turn's own context, not a fixture.
        results = execute_plan(plan, question.question, question.video_id, question.window,
                               question.split)
        results.pop("retrieve_documents", None)  # the RAG call itself is replayed below
        before_query = question.question
        after_query = build_retrieval_query(question.question, results)

        before = rank_of_canonical_chunk(before_query, args.probe_depth)
        after = rank_of_canonical_chunk(after_query, args.probe_depth)

        print(f"\n######## {qid} — {question.question}")
        print(f"  tools planned      : {plan.called}")
        print(f"  query BEFORE       : {before_query!r}")
        print(f"  query AFTER        : {after_query!r}")
        print(f"  added by enrichment: {after_query[len(before_query):].strip() or '(nothing)'}")
        for label, hit in (("BEFORE", before), ("AFTER", after)):
            if hit is None:
                print(f"  canonical rank {label:6}: NOT FOUND within {args.probe_depth}")
            else:
                print(f"  canonical rank {label:6}: {hit['rank']}  score={hit['score']:.4f}  "
                      f"{hit['source']} :: {hit['section']}  "
                      f"(top-{PRODUCTION_TOP_K}: {hit['rank'] <= PRODUCTION_TOP_K})")
        print(f"  --- top-{PRODUCTION_TOP_K} BEFORE")
        for i, line in enumerate(top_k_sources(before_query), 1):
            print(f"      {i}. {line}")
        print(f"  --- top-{PRODUCTION_TOP_K} AFTER")
        for i, line in enumerate(top_k_sources(after_query), 1):
            print(f"      {i}. {line}")
        summary.append((qid, before, after))

    print("\n==== SUMMARY (production top_k =", PRODUCTION_TOP_K, ") ====")
    print(f"{'Q':5} {'before':>8} {'after':>8}  in top-5 after")
    for qid, before, after in summary:
        b = before["rank"] if before else None
        a = after["rank"] if after else None
        print(f"{qid:5} {str(b):>8} {str(a):>8}  {a is not None and a <= PRODUCTION_TOP_K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""READ-ONLY diagnostic: current retrieval behaviour for Q11/Q12."""
import sys, json, re
sys.argv = [sys.argv[0]]; sys.path.insert(0, ".")
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from rag import retrieval

QS = {q.question_id: q.question for q in FIXED_QUESTIONS}
ORDER_HEAD = ("tPB2", "tPNa", "tPNf")          # the taxonomy's opening, in order
ORDER_TAIL = ("tSB", "tB", "tEB")

def carries_canonical_order(content: str) -> bool:
    """A chunk 'carries the canonical order' iff it spells out the phase
    taxonomy as an ordered sequence: the three opening names in order, and
    at least one closing name after them."""
    if not all(p in content for p in ORDER_HEAD): return False
    i = content.index("tPB2")
    if not (content.index("tPNa") > i and content.index("tPNf") > content.index("tPNa")): return False
    return any(p in content[content.index("tPNf"):] for p in ORDER_TAIL)

def show(qid, query, top_k=25, mark=5):
    res = retrieval.retrieve(query, top_k=top_k)
    print(f"\n=== {qid}  top_k_probe={top_k} (production top_k={mark})")
    print(f"    query sent verbatim: {query!r}")
    hits = []
    for i, r in enumerate(res, 1):
        c = r.get("content") or ""
        flag = "  <<< CANONICAL ORDER" if carries_canonical_order(c) else ""
        inside = "*" if i <= mark else " "
        if i <= 8 or flag:
            print(f"  {inside}{i:3d}  {r['score']:.4f}  {r['metadata']['source']} :: "
                  f"{r['metadata']['section'][:48]}  [{r['metadata'].get('chunk_id', '?')}]{flag}")
        if flag: hits.append((i, r))
    if hits:
        i, r = hits[0]
        print(f"    -> first canonical chunk at RANK {i} (in top-5: {i <= mark})")
        print(f"       chunk_id={r['metadata'].get('chunk_id')}  score={r['score']:.4f}")
        print(f"       excerpt: {(r['content'] or '')[:230].replace(chr(10),' ')}")
    else:
        print(f"    -> NO canonical chunk in the first {top_k}")
    return hits[0][0] if hits else None

ranks = {}
for qid in ("Q11", "Q12"):
    ranks[qid] = show(qid, QS[qid])
print("\n==== REFERENCE MEASUREMENT ====")
for qid, r in ranks.items():
    print(f"{qid}: canonical order at rank {r}  -> in top-5: {r is not None and r <= 5}")

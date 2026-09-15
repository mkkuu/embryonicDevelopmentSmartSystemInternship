import sys; sys.argv=[sys.argv[0]]; sys.path.insert(0,".")
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from rag import retrieval
qs = {q.question_id: q.question for q in FIXED_QUESTIONS}
def flag(r):
    c = r.get("content") or r.get("document") or ""
    if "tPB2" in c and "tPNa" in c: return "ORDER"
    if "orphok" in c.lower() or "orphocin" in c.lower(): return "morphok"
    return ""
for qid in ("Q11","Q12","Q13","Q15"):
    res = retrieval.retrieve(qs[qid], top_k=25)
    print(f"=== {qid}: {qs[qid]}")
    for i, r in enumerate(res, 1):
        print("  %2d %.3f %s :: %s %s" % (i, r["score"], r["metadata"]["source"], r["metadata"]["section"][:60], flag(r)))
print("=== probe queries")
for probe in ("ordre chronologique des phases tPB2 tPNa tPNf t2 t3 t4 t5 t6 t7 t8",
              "ordre attendu des phases du developpement embryonnaire",
              "liste des 15 phases developpementales dans l ordre biologique",
              "quelle phase vient apres t6"):
    res = retrieval.retrieve(probe, top_k=5)
    print("--", probe)
    for i, r in enumerate(res, 1):
        print("  %d %.3f %s :: %s %s" % (i, r["score"], r["metadata"]["source"], r["metadata"]["section"][:50], flag(r)))
print("=== chunks containing the order list")
res = retrieval.retrieve("tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB", top_k=10)
for i, r in enumerate(res, 1):
    c = r.get("content") or r.get("document") or ""
    print("  %d %.3f %s :: %s %s len=%d" % (i, r["score"], r["metadata"]["source"], r["metadata"]["section"][:50], flag(r), len(c)))
    if flag(r) == "ORDER": print("     >>", c[:300].replace("\n", " "))

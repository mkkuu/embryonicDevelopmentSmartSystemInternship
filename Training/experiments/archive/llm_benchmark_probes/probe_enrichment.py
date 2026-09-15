"""READ-ONLY probe: does enriching the query with the turn's own phase tokens
pull the canonical-order chunk into the top-5? No code is modified here."""
import sys, json
sys.argv = [sys.argv[0]]; sys.path.insert(0, ".")
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.router import extract_phase_tokens
from rag import retrieval

QS = {q.question_id: q.question for q in FIXED_QUESTIONS}
ART = ("/path/to/embryonicDevelopmentSciMLExtension/Results/evaluation/"
       "event_anomaly_rag_inventory/fixed_question_benchmark_mistral-nemo_12b_"
       "prompt_v2_reading_method.json")

def carries_order(c):
    if not all(p in c for p in ("tPB2", "tPNa", "tPNf")): return False
    i = c.index("tPB2")
    return c.index("tPNa") > i and c.index("tPNf") > c.index("tPNa")

def collect_strings(v, out):
    if isinstance(v, dict):
        for x in v.values(): collect_strings(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v: collect_strings(x, out)
    elif isinstance(v, str): out.append(v)

rows = {r["question_id"]: r for r in json.load(open(ART))["rows"]}

def rank_of_order(query, top_k=25):
    res = retrieval.retrieve(query, top_k=top_k)
    for i, r in enumerate(res, 1):
        if carries_order(r.get("content") or ""):
            return i, r["score"], f"{r['metadata']['source']} :: {r['metadata']['section'][:40]}"
    return None, None, None

for qid in ("Q11", "Q12"):
    q = QS[qid]
    parts = []; collect_strings(rows[qid]["dynamic_context"], parts)
    phases = extract_phase_tokens(" ".join(parts))
    print(f"\n######## {qid}")
    print(f"  phases present in the turn's context (from tool output VALUES): {phases}")
    variants = {
        "0. question verbatim (current behaviour)": q,
        "1. question + bare phase tokens": f"{q} {' '.join(phases)}",
        "2. question + phases, labelled (fr)": f"{q} phases : {', '.join(phases)}",
        "3. question + 'ordre chronologique des phases' + phases":
            f"{q} ordre chronologique des phases : {', '.join(phases)}",
        "4. phases only": " ".join(phases),
    }
    for label, query in variants.items():
        rank, score, where = rank_of_order(query)
        top5 = rank is not None and rank <= 5
        print(f"  {label:52} -> rank={str(rank):>5} score={score if score is None else round(score,4)} "
              f"top5={top5}  {where or ''}")

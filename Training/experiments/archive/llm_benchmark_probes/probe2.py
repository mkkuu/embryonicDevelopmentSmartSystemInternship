"""READ-ONLY probe 2: generic enrichment variants, incl. distribution keys."""
import sys, json
sys.argv = [sys.argv[0]]; sys.path.insert(0, ".")
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.router import extract_phase_tokens, PHASE_TOKEN_PATTERN
from rag import retrieval

QS = {q.question_id: q.question for q in FIXED_QUESTIONS}
ART = ("/path/to/embryonicDevelopmentSciMLExtension/Results/evaluation/"
       "event_anomaly_rag_inventory/fixed_question_benchmark_mistral-nemo_12b_"
       "prompt_v2_reading_method.json")
rows = {r["question_id"]: r for r in json.load(open(ART))["rows"]}

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

def distribution_keys(v, out, top_n=None):
    """Phase names reachable as KEYS of a phase-distribution dict."""
    if isinstance(v, dict):
        if v and all(isinstance(k, str) and PHASE_TOKEN_PATTERN.fullmatch(k) for k in v) and \
           all(not isinstance(x, bool) and isinstance(x, (int, float)) for x in v.values()):
            keys = list(v.keys()) if top_n is None else \
                [k for k, _ in sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
            for k in keys:
                if k.lower() not in out: out.append(k.lower())
            return
        for x in v.values(): distribution_keys(x, out, top_n)
    elif isinstance(v, (list, tuple)):
        for x in v: distribution_keys(x, out, top_n)

def rank_of_order(query, top_k=25):
    for i, r in enumerate(retrieval.retrieve(query, top_k=top_k), 1):
        if carries_order(r.get("content") or ""):
            return i, round(r["score"], 4), f"{r['metadata']['source']} :: {r['metadata']['section'][:36]}"
    return None, None, None

for qid in ("Q11", "Q12"):
    q, dyn = QS[qid], rows[qid]["dynamic_context"]
    parts = []; collect_strings(dyn, parts)
    values = extract_phase_tokens(" ".join(parts))
    top2, allk = [], []
    distribution_keys(dyn, top2, top_n=2); distribution_keys(dyn, allk, top_n=None)
    union_top2 = values + [p for p in top2 if p not in values]
    union_all = values + [p for p in allk if p not in values]
    print(f"\n######## {qid}")
    print(f"  values only        : {values}")
    print(f"  + top-2 dist. keys : {union_top2}")
    print(f"  + ALL dist. keys   : {union_all}")
    for label, query in {
        "A. values, bare":                    f"{q} {' '.join(values)}",
        "B. values + top-2 keys, bare":       f"{q} {' '.join(union_top2)}",
        "C. ALL distribution keys, bare":     f"{q} {' '.join(union_all)}",
        "D. values, labelled 'phases :'":     f"{q} phases : {', '.join(values)}",
        "E. values+top2, labelled 'phases :'": f"{q} phases : {', '.join(union_top2)}",
        "F. ALL keys, labelled 'phases :'":   f"{q} phases : {', '.join(union_all)}",
    }.items():
        rank, score, where = rank_of_order(query)
        print(f"  {label:38} -> rank={str(rank):>5} score={score} top5={rank is not None and rank<=5}  {where or ''}")

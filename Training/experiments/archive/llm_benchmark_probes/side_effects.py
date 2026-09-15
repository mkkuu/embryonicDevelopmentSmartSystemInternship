"""READ-ONLY: does any question other than Q11/Q12 get a different
document_context than the one stored in the four v2 artefacts?"""
import sys, json, glob
sys.argv = [sys.argv[0]]; sys.path.insert(0, ".")
from orchestrator import tools
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import build_retrieval_query, execute_plan, plan_tools
from orchestrator.router import classify

ART_DIR = ("/path/to/embryonicDevelopmentSciMLExtension/Results/evaluation/"
           "event_anomaly_rag_inventory/")
stored = {}
for p in sorted(glob.glob(ART_DIR + "fixed_question_benchmark_mistral-nemo_12b_prompt_v2_reading_method*.json")):
    for r in json.load(open(p))["rows"]:
        stored.setdefault(r["question_id"], []).append(
            [(c.get("source"), c.get("section")) for c in r["document_context"]])

by_id = {q.question_id: q for q in FIXED_QUESTIONS}
print(f"{'Q':5} {'enriched':>9} {'stored ctx identical across 4 runs':>34} {'now == stored':>14}")
for qid in [f"Q{i}" for i in range(1, 16)]:
    if qid not in stored or not stored[qid][0]:
        print(f"{qid:5} {'-':>9} {'(no document_context)':>34} {'-':>14}")
        continue
    q = by_id[qid]
    plan = plan_tools(q.question, classify(q.question), video_id=q.video_id, window=q.window)
    if "retrieve_documents" not in plan.called:
        print(f"{qid:5} {'-':>9} {'(RAG not planned)':>34} {'-':>14}")
        continue
    results = execute_plan(plan, q.question, q.video_id, q.window, q.split)
    now = [(c.get("metadata", {}).get("source"), c.get("metadata", {}).get("section"))
           for c in (results["retrieve_documents"].value or [])]
    results_wo = dict(results); results_wo.pop("retrieve_documents", None)
    enriched = build_retrieval_query(q.question, results_wo) != q.question
    all_same = all(s == stored[qid][0] for s in stored[qid])
    print(f"{qid:5} {str(enriched):>9} {str(all_same):>34} {str(now == stored[qid][0]):>14}")

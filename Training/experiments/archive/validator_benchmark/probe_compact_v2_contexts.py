"""Read-only probe: the REAL compact_v2 context for the 15 frozen questions
(real Reporting API, real Chroma index, real Istanbul corpus), rendered in
both v2 and compact_v2. No LLM call. Writes /tmp/compact_v2_probe.json.
Run from Training/ with CUDA_VISIBLE_DEVICES=''."""
import json, sys, time
sys.argv = sys.argv[:1]
from orchestrator import prompt
from orchestrator.compact_v2_render import CONTEXT_FORMAT_COMPACT_V2
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import execute_plan, plan_tools
from orchestrator.router import classify

rows = []
for q in FIXED_QUESTIONS:
    t0 = time.time()
    route = classify(q.question)
    plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
    results_v2 = execute_plan(plan, q.question, q.video_id, q.window, q.split, context_policy="v2")
    results_c2 = execute_plan(plan, q.question, q.video_id, q.window, q.split, context_policy=CONTEXT_FORMAT_COMPACT_V2)
    ctx_v2 = build_context(q.question, route, results_v2, tools_called=plan.called)
    ctx_c2 = build_context(q.question, route, results_c2, tools_called=plan.called)
    p_v2 = prompt.build_user_prompt(ctx_v2, "v2")
    p_c1 = prompt.build_user_prompt(ctx_v2, "compact")
    p_c2 = prompt.build_user_prompt(ctx_c2, CONTEXT_FORMAT_COMPACT_V2)
    sci = ctx_c2.get("scientific_context") or {}
    row = {
        "question_id": q.question_id, "question": q.question, "route": route.category,
        "tools": plan.called,
        "v2_chars": len(p_v2), "compact_v1_chars": len(p_c1), "compact_v2_chars": len(p_c2),
        "v2_docs": [(c["source"], c["section"]) for c in ctx_v2["document_context"]],
        "c2_docs": [(c["source"], c["section"]) for c in ctx_c2["document_context"]],
        "retrieval_audit": ctx_c2.get("retrieval_audit"),
        "scientific_blocks": [b["block_id"] for b in sci.get("blocks", [])],
        "scientific_limitations": [b["block_id"] for b in sci.get("limitations", [])],
        "scientific_warnings": sci.get("warnings"),
        "transition_context": ctx_c2["dynamic_context"].get("transition_context"),
        "warnings_c2": ctx_c2["warnings"],
        "prompt_v2": p_v2, "prompt_compact_v2": p_c2,
        "elapsed": round(time.time() - t0, 1),
    }
    rows.append(row)
    print(f"[{q.question_id}] {route.category} v2={len(p_v2)} c1={len(p_c1)} c2={len(p_c2)} "
          f"docs v2={len(row['v2_docs'])} c2={len(row['c2_docs'])} sci={row['scientific_blocks']} "
          f"tc={'yes' if row['transition_context'] else 'no'} ({row['elapsed']}s)", flush=True)
json.dump(rows, open("/tmp/compact_v2_probe.json", "w"), ensure_ascii=False, indent=1)
print("wrote /tmp/compact_v2_probe.json")

import json, sys
from orchestrator.compact_v2_render import CONTEXT_FORMAT_COMPACT_V2
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import execute_plan, plan_tools
from orchestrator.router import classify
v1 = {r["question_id"]: r for r in json.load(open("../Results/evaluation/validator_experiment/paired_validator_off_vs_on_mistral-nemo_12b_validator_v1.json"))["rows"]}
def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k, "<absent>"), b.get(k, "<absent>"), path + "." + str(k), out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, str(a)[:80], str(b)[:80]))
for q in FIXED_QUESTIONS:
    if q.question_id not in sys.argv[1].split(","): continue
    route = classify(q.question); plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
    results = execute_plan(plan, q.question, q.video_id, q.window, q.split, context_policy=CONTEXT_FORMAT_COMPACT_V2)
    ctx = build_context(q.question, route, results, tools_called=plan.called)
    out = []; walk(ctx["dynamic_context"], v1[q.question_id]["dynamic_context"], "dyn", out)
    print(f"== {q.question_id}: {len(out)} differing leaf paths")
    for p, a, b in out[:12]: print(f"  {p}\n     now={a}\n     v1 ={b}")
    pr = __import__("orchestrator.prompt", fromlist=["x"]).build_user_prompt(ctx, CONTEXT_FORMAT_COMPACT_V2)
    old = v1[q.question_id]["llm_input_user_prompt"]
    import difflib
    d = [l for l in difflib.unified_diff(old.splitlines(), pr.splitlines(), lineterm="", n=0) if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    print(f"  prompt diff lines: {len(d)}"); [print("   ", l[:140]) for l in d[:8]]

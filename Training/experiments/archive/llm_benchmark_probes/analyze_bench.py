"""Read-only comparison of the two mistral-nemo benchmark artifacts."""
import json, sys, re

BASE = "/path/to/embryonicDevelopmentSciMLExtension/Results/evaluation/event_anomaly_rag_inventory"
OLD = f"{BASE}/fixed_question_benchmark_mistral-nemo_12b_num_ctx16384.json"
NEW = f"{BASE}/fixed_question_benchmark_mistral-nemo_12b_post_temporal_restructuring.json"

old = {r["question_id"]: r for r in json.load(open(OLD))["rows"]}
newd = json.load(open(NEW))
new = {r["question_id"]: r for r in newd["rows"]}

print(f"model={newd['model']} timeout={newd['timeout']} n={newd['n_questions']}")
els = [r["elapsed_seconds"] for r in new.values()]
print(f"total={sum(els):.1f}s mean={sum(els)/len(els):.1f}s max={max(els):.1f}s "
      f"errors={sum(1 for r in new.values() if 'error' in r)}")
print()
for qid in [f"Q{i}" for i in range(1, 16)]:
    n, o = new[qid], old[qid]
    print("=" * 100)
    print(f"### {qid}  {n['question']}")
    print(f"  tools : {n['tool_plan']['called']}")
    print(f"  prompt_chars : {o['llm_input_user_prompt_length']} -> {n['llm_input_user_prompt_length']}")
    print(f"  elapsed : {o['elapsed_seconds']:.1f}s -> {n['elapsed_seconds']:.1f}s")
    print(f"  grounded : {o['grounding_detail']['grounded']} -> {n['grounding_detail']['grounded']}"
          f"  ungrounded_numbers={n['grounding_detail'].get('ungrounded_numbers')}")
    print(f"--- OLD ANSWER ---\n{o['answer'].strip()}")
    print(f"--- NEW ANSWER ---\n{n['answer'].strip()}")
    print()

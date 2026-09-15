import json
r = json.load(open("Results/evaluation/llm_benchmark_ollama_llama3.2-latest/benchmark.json"))
for i, row in enumerate(r["rows"], 1):
    q = row["question"]
    cat = row["route_selected"]
    print(f"--- Q{i} [{cat}] {q}")
    print("   grounded:", row["groundedness"], "| confidence:", row["confidence"])
    excerpt = row["answer_excerpt"][:280].replace(chr(10), " ")
    print("   answer:", excerpt)
    if row["warnings"]:
        print("   warnings:", row["warnings"])
    print()

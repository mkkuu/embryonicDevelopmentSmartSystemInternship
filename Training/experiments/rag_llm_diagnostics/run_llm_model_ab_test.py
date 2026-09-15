"""
Model A/B test runner for docs/RAG_LLM_MODEL_AB_TEST.md -- tests whether
a different local Ollama model improves real context usage vs. the
production baseline (`llama3.2:latest`), on the EXACT same 7-question
battery as `docs/RAG_CONTEXT_FAILURE_ANALYSIS.md`/
`docs/RAG_PROMPT_REORDERING_AB_TEST.md` (`Patient_319`, `split=val`,
window 156, the real `t4 -> t6` skip transition) -- same questions, same
Router/Tools/Context Builder/Grounding/prompt.py (prompt reordering
stays active, unmodified), only the Ollama `model`/`request_timeout_seconds`
differ from production's `_CHAT_PROVIDER` default.

Per-question wall-clock latency is recorded explicitly (this session's
central additional concern, given mistral-small3.1:24b's previously-
documented GPU-offload/latency problem, docs/LLM_EVALUATION.md sec 0) --
never silently absorbed into a background metric.

Usage (from Training/, GPU server):
    python -m experiments.rag_llm_diagnostics.run_llm_model_ab_test --model mistral-small3.1:24b \\
        --timeout 300 --run-label run1
"""

from __future__ import annotations

import argparse
import json
import time

from orchestrator import grounding_check, prompt as prompt_module
from orchestrator.llm_provider import OllamaLLMProvider, OllamaUnavailableError
from orchestrator.orchestrator import answer_question

VIDEO_ID = "Patient_319"
SPLIT = "val"
WINDOW = 156  # real, verified skip transition: t4 -> t6 (window_start=156, window_end=157)

QUESTIONS = [
    ("A_dynamic_simple", "Quelle est la phase prédite à cette fenêtre ?", VIDEO_ID, WINDOW),
    ("B_numeric", "Quelle est la probabilité de la phase actuelle ?", VIDEO_ID, WINDOW),
    ("C_transition", "Quelle transition de phase est observée ici ?", VIDEO_ID, WINDOW),
    ("D_skip", "Y a-t-il un saut de phase ici ?", VIDEO_ID, WINDOW),
    ("E_documentary", "Qu'est-ce qu'un HMM ?", None, None),
    ("F_hybrid", "Compare la transition observée avec la prédiction du modèle.", VIDEO_ID, WINDOW),
    ("G_out_of_data", "Y a-t-il eu un direct cleavage ?", VIDEO_ID, WINDOW),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=300.0,
                         help="Generous test-only timeout -- NOT production's default (30s); "
                              "production timeout implications are analyzed separately in the report.")
    parser.add_argument("--run-label", default="run1")
    args = parser.parse_args()

    provider = OllamaLLMProvider(model=args.model, request_timeout_seconds=args.timeout)

    rows = []
    for qid, question, video_id, window in QUESTIONS:
        t0 = time.time()
        try:
            out = answer_question(question, video_id=video_id, window=window, split=SPLIT,
                                   llm=provider, llm_timeout_seconds=args.timeout)
        except OllamaUnavailableError as e:
            elapsed = time.time() - t0
            print(f"[{qid}] FAILED after {elapsed:.1f}s: {e}")
            rows.append({"qid": qid, "question": question, "window": window,
                         "error": str(e), "elapsed_seconds": elapsed})
            continue
        elapsed = time.time() - t0
        context = out["context"]
        response = out["response"]

        system_prompt = prompt_module.build_system_prompt()
        user_prompt = prompt_module.build_user_prompt(context)
        grounding_detail = grounding_check.check_grounding(response["text"], context)

        rows.append({
            "qid": qid, "question": question, "video_id": video_id, "window": window,
            "elapsed_seconds": elapsed,
            "route": out["route"],
            "tool_plan": out["tool_plan"],
            "dynamic_context": context["dynamic_context"],
            "document_context_count": len(context["document_context"]),
            "llm_input_user_prompt_length": len(user_prompt),
            "llm_output_raw": response["text"],
            "llm_self_reported_grounded": response["grounded"],
            "grounding_detail": grounding_detail.to_dict(),
            "final_confidence": response["confidence"],
            "final_warnings": response["warnings"],
        })
        print(f"[{qid}] elapsed={elapsed:.1f}s route={out['route']['category']} "
              f"tools={out['tool_plan']['called']} grounded={grounding_detail.grounded}")

    model_slug = args.model.replace(":", "_").replace(".", "_")
    output_path = (f"../Results/evaluation/event_anomaly_rag_inventory/"
                    f"model_ab_test_{model_slug}_{args.run_label}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "timeout": args.timeout, "video_id": VIDEO_ID,
                   "split": SPLIT, "window": WINDOW, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

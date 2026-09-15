"""
Diagnostic-only script for docs/RAG_CONTEXT_FAILURE_ANALYSIS.md -- NOT a
fix, NOT a benchmark with pass/fail thresholds. Captures, for each of 7
real questions against the exact same real, already-validated case
(`Patient_319`, `split=val`, the real t4->t6 skip transition at window
156->157, `docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md` sec 10):

  QUESTION -> ROUTE -> TOOLS -> CONTEXT (raw dynamic_context) ->
  LLM INPUT (the exact system+user prompt strings actually sent to
  Ollama, built from the SAME context object, via prompt.py -- never a
  guess or a re-derivation) -> LLM OUTPUT (raw text) -> GROUNDING RESULT
  (full detail, not just the boolean) -> FINAL RESPONSE.

Uses the exact same OllamaLLMProvider configuration as production
`/chat` (`Training/webapp_api/app.py::_CHAT_PROVIDER` -- model
"llama3.2:latest", no seed, no temperature override, default 30s
timeout) -- this diagnostic is measuring the REAL production behavior,
not a more lenient test-only configuration.
"""

from __future__ import annotations

import json

from orchestrator import grounding_check, prompt as prompt_module
from orchestrator.llm_provider import OllamaLLMProvider
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
    # Exact same provider config as production /chat (Training/webapp_api/app.py
    # _CHAT_PROVIDER) -- no seed, no temperature, default 30s timeout.
    provider = OllamaLLMProvider(model="llama3.2:latest")

    rows = []
    for qid, question, video_id, window in QUESTIONS:
        out = answer_question(question, video_id=video_id, window=window, split=SPLIT, llm=provider)
        context = out["context"]
        response = out["response"]

        # The exact system+user prompt strings actually built from THIS
        # turn's real context -- built the SAME way OllamaLLMProvider.generate()
        # itself builds them (Training/orchestrator/llm_provider.py), so this
        # is provably the real LLM INPUT, not a guess.
        system_prompt = prompt_module.build_system_prompt()
        user_prompt = prompt_module.build_user_prompt(context)

        grounding_detail = grounding_check.check_grounding(response["text"], context)

        rows.append({
            "qid": qid, "question": question, "video_id": video_id, "window": window,
            "route": out["route"],
            "tool_plan": out["tool_plan"],
            "dynamic_context": context["dynamic_context"],
            "document_context_count": len(context["document_context"]),
            "context_warnings": context["warnings"],
            "llm_input_system_prompt": system_prompt,
            "llm_input_user_prompt": user_prompt,
            "llm_output_raw": response["text"],
            "llm_self_reported_grounded": response["grounded"],
            "grounding_detail": grounding_detail.to_dict(),
            "final_confidence": response["confidence"],
            "final_warnings": response["warnings"],
        })
        print(f"[{qid}] route={out['route']['category']} tools={out['tool_plan']['called']} "
              f"grounded={grounding_detail.grounded}")

    output_path = "../Results/evaluation/event_anomaly_rag_inventory/rag_context_failure_diagnostic.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": VIDEO_ID, "split": SPLIT, "window": WINDOW, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

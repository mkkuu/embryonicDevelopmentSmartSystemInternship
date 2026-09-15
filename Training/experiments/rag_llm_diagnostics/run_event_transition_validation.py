"""
Real end-to-end validation for the Event/Transition RAG axis, P1
(docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md sec 10, task's own sec 17).
Not a benchmark/statistics run -- a small, fixed set of real calls through
the unmodified `orchestrator.answer_question()` pipeline (real router,
real Reporting API incl. the new get_transition_events tool, real Ollama
`llama3.2:latest`, real grounding), against `Patient_319`/`split=val`,
around window 156 -- a REAL, VERIFIED skip transition (t4 -> t6, skips
t5) found in this video's own ground truth (not an arbitrarily-picked
window; see Training/find_skip_case.py's real output).

Also runs the 4 explicitly-unsupported anomaly questions (direct
cleavage / fragmentation / multinucleation / reversal) to confirm the
system still answers honestly, never hallucinating, now that a NEW tool
(get_transition_events) exists that could in principle be misused to
fabricate such data -- it structurally cannot, since TransitionEvent has
no such field.

Usage (from Training/, GPU server, real Ollama required):
    python -m experiments.rag_llm_diagnostics.run_event_transition_validation
"""

from __future__ import annotations

import json

from orchestrator import grounding_check
from orchestrator.llm_provider import OllamaLLMProvider
from orchestrator.orchestrator import answer_question

VIDEO_ID = "Patient_319"
SPLIT = "val"
SKIP_WINDOW = 156  # real, verified skip transition: t4 -> t6 (window_start=156, window_end=157)

QUESTIONS = [
    ("Q1_observed_phase", "Quelle est la phase observée à cette fenêtre ?", SKIP_WINDOW),
    ("Q2_observed_transition", "Quelle transition de phase est observée ici ?", SKIP_WINDOW),
    ("Q3_is_skip", "Y a-t-il un saut de phase ?", SKIP_WINDOW),
    ("Q4_model_prediction", "Que prédit le modèle autour de cette transition ?", SKIP_WINDOW),
    ("Q5_compare", "Compare la transition observée avec la prédiction du modèle.", SKIP_WINDOW),
    ("Q6_why_skip", "Pourquoi cette transition est-elle considérée comme un skip ?", SKIP_WINDOW),
    ("U1_direct_cleavage", "Y a-t-il eu un direct cleavage ?", SKIP_WINDOW),
    ("U2_fragmentation", "Y a-t-il eu une fragmentation ?", SKIP_WINDOW),
    ("U3_multinucleation", "Y a-t-il eu une multinucleation ?", SKIP_WINDOW),
    ("U4_reversal", f"Quand a eu lieu le reversal chez {VIDEO_ID} ?", None),
]


def main() -> int:
    provider = OllamaLLMProvider(model="llama3.2:latest", request_timeout_seconds=60.0)
    rows = []
    for qid, question, window in QUESTIONS:
        out = answer_question(question, video_id=VIDEO_ID, window=window, split=SPLIT,
                               llm=provider, llm_timeout_seconds=60.0)
        response = out["response"]
        # Independent, full-detail re-check (same text/context the pipeline
        # itself already grounded) -- purely for observability in this
        # validation's own output; not a second grounding pass affecting
        # the real answer (docs/RELATIONAL_GROUNDING.md sec 8/13).
        detail = grounding_check.check_grounding(response["text"], out["context"])
        rows.append({
            "qid": qid, "question": question, "window": window,
            "route": out["route"]["category"],
            "tools_called": out["tool_plan"]["called"],
            "tools_not_called": [d["tool"] for d in out["tool_plan"]["not_called"]],
            "dynamic_context_keys": list(out["context"]["dynamic_context"].keys()),
            "n_sources": len(out["context"]["document_context"]),
            "grounded": response["grounded"],
            "warnings": response["warnings"],
            "answer": response["text"],
            "checked_transition_claims": detail.checked_transition_claims,
            "ungrounded_transition_claims": detail.ungrounded_transition_claims,
            "skip_claim_mismatches": detail.skip_claim_mismatches,
        })
        print(f"[{qid}] route={out['route']['category']} tools={out['tool_plan']['called']} "
              f"grounded={response['grounded']}")

    output_path = "../Results/evaluation/event_anomaly_rag_inventory/transition_validation_AFTER.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": VIDEO_ID, "split": SPLIT, "skip_window": SKIP_WINDOW, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

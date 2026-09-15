"""Preflight, NO LLM generation: identity of every answer-affecting condition
against the v1 artefact, plus the 15 contexts rebuilt and hashed."""
import json, hashlib, sys
from pathlib import Path
from orchestrator import prompt as prompt_module, scientific_validation
from orchestrator.compact_v2_render import CONTEXT_FORMAT_COMPACT_V2
from orchestrator.context_builder import build_context
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import execute_plan, plan_tools
from orchestrator.router import classify
from orchestrator.run_validator_experiment import _sha256, _sha256_file, _canonical, _HASHED_MODULES, _TRAINING
from orchestrator.run_fixed_question_benchmark import ollama_server_metadata
from orchestrator import llm_config
from validator.corpus import DEFAULT_CORPUS_PATH

v1 = json.load(open("../Results/evaluation/validator_experiment/paired_validator_off_vs_on_mistral-nemo_12b_validator_v1.json"))
ok = True
def check(name, a, b):
    global ok
    same = a == b
    ok &= same
    print(("OK   " if same else "DIFF ") + name, "" if same else f"\n      now={str(a)[:160]!r}\n      v1 ={str(b)[:160]!r}")

print("engine:", scientific_validation.VALIDATOR_ENGINE)
check("engine is v1.3", scientific_validation.VALIDATOR_ENGINE, "scientific_validator v1.3")
check("questions_sha256", _sha256(_canonical([q.question for q in FIXED_QUESTIONS])), v1["questions_sha256"])
check("system_prompt_sha256", _sha256(prompt_module.build_system_prompt()), v1["system_prompt_sha256"])
check("istanbul_corpus_sha256", _sha256_file(DEFAULT_CORPUS_PATH), v1["istanbul_corpus_sha256"])
check("n_questions", len(FIXED_QUESTIONS), 15)
changed_expected = {"orchestrator/scientific_validation.py", "validator/validator.py", "validator/rules.py",
                    "validator/extraction.py", "validator/provenance.py"}
for m in _HASHED_MODULES:
    now = _sha256_file(_TRAINING / m)
    if m in changed_expected:
        print(("OK   " if now != v1["module_sha256"][m] else "DIFF ") + f"{m} changed (v1.3)")
        ok &= now != v1["module_sha256"][m]
    else:
        check(m, now, v1["module_sha256"][m])
provider = llm_config.build_ollama_provider("mistral-nemo:12b", 240.0, context_format=CONTEXT_FORMAT_COMPACT_V2)
meta = llm_config.describe_provider(provider); meta.update(ollama_server_metadata(provider.base_url, provider.model))
for k in ("model", "num_ctx", "seed", "temperature", "request_timeout_seconds", "context_format",
          "ollama_version", "model_digest"):
    check(f"provider.{k}", meta.get(k), v1["provider"].get(k))
check("provider.quantization", meta.get("model_details", {}).get("quantization_level"),
      v1["provider"]["model_details"]["quantization_level"])
v1rows = {r["question_id"]: r for r in v1["rows"]}
for q in FIXED_QUESTIONS:
    route = classify(q.question)
    plan = plan_tools(q.question, route, video_id=q.video_id, window=q.window)
    results = execute_plan(plan, q.question, q.video_id, q.window, q.split, context_policy=CONTEXT_FORMAT_COMPACT_V2)
    ctx = build_context(q.question, route, results, tools_called=plan.called)
    prompt = prompt_module.build_user_prompt(ctx, CONTEXT_FORMAT_COMPACT_V2)
    r = v1rows[q.question_id]
    check(f"{q.question_id} anchor", (q.video_id, q.window, q.split), ("Patient_319", 156, "val"))
    check(f"{q.question_id} dynamic_context_sha256", _sha256(_canonical(ctx["dynamic_context"])), r["dynamic_context_sha256"])
    check(f"{q.question_id} document_context_sha256", _sha256(_canonical(ctx["document_context"])), r["document_context_sha256"])
    check(f"{q.question_id} scientific_context_sha256", _sha256(_canonical(ctx.get("scientific_context"))), r["scientific_context_sha256"])
    check(f"{q.question_id} user_prompt", prompt, r["llm_input_user_prompt"])
    for bad in ("PASS", "success_criterion", "expected_information", "grades_compact", "validator_v1"):
        ok &= bad not in prompt
    if any(b in prompt for b in ("success_criterion", "expected_information")):
        print("LEAK in prompt", q.question_id)
print("\nPREFLIGHT", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)

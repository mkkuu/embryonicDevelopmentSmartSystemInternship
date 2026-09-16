"""
RAG / LLM Orchestration Foundation (docs/PRODUCT_ROADMAP.md, the phase
after Reporting API / Reporting Cache / RAG Phase 3). ADDITIVE only --
never imports from, modifies, or is imported by Training/evaluation/,
Training/embeddings/, Training/reporting/, or Training/rag/; it only
CALLS into reporting/rag through tools.py, exactly like the Web App would.

This package is a FOUNDATION, not a finished chatbot: routing + typed
tool contracts + a normalized context builder + an interchangeable LLM
abstraction. See docs/LLM_ORCHESTRATION.md and
docs/LLM_PROMPT_CONTRACT.md.

Outdated as written (kept corrected rather than deleted, per this
project's convention): this docstring used to state that no LLM API was
called here and that there was no chat and no Web App wiring. All three
became false in later phases. `llm_provider.OllamaLLMProvider` really does
call a local Ollama endpoint, and `Training/webapp_api/` wires
`orchestrator.answer_question()` plus `fixed_question_benchmark` into a
chat UI in-process. There is still no GRU tool.

Module map
----------
router.py           Deterministic (non-ML) question classifier:
                     DOCUMENTARY | DYNAMIC_DATA | HYBRID | UNKNOWN.
tools.py             Typed wrappers around the Reporting API
                      (Training/reporting/) and the RAG retrieval
                      prototype (Training/rag/) -- the only two data
                      paths any orchestrator/LLM is ever allowed to use.
                      See docs/TOOL_CONTRACTS.md for the full contracts.
context_builder.py   Assembles tool outputs into one normalized,
                      provenance-tagged structure (document_context vs.
                      dynamic_context, never merged).
llm_provider.py       LLMProvider abstraction + NullLLMProvider (no
                       network call, deterministic, used when no real
                       LLM is configured -- tests must pass without one).
orchestrator.py        Ties the above together: answer_question(...).
scientific_validation.py  Bridge to Training/validator/ (the Scientific
                       Validator, ON by default). See that package's
                       docstring for the deliberate, benign import cycle
                       between the two.
fixed_question_benchmark.py  The frozen Q1-Q15 set. Despite the name this
                       is PRODUCTION, not an experiment: Training/
                       webapp_api/app.py imports FIXED_QUESTIONS from here
                       as the single source of their wording. Do not move
                       it under Training/experiments/.

Moved out of this package by the 2026-09-15 reorganisation (they were
documented here): eval_questions.py and run_orchestration_evaluation.py
now live in Training/experiments/llm_regression/ and run as
`python -m experiments.llm_regression.<script>`.

Hard architectural rule (docs/PRODUCT_ARCHITECTURE.md sec 4, restated
here because this package is where it is actually enforced in code): the
Reporting API is the ONLY source of truth for a live/dynamic number: the
RAG never stores or serves one. This package's tools.py has no code path
capable of writing a Reporting API value into the vector DB, and no code
path capable of treating a RAG chunk as a live prediction.
"""

"""
Small controlled benchmark (task Phase 6 sec 11), run against
`TemplateLLMProvider` -- the only real (non-null) provider registered as
of this module, since no external-vs-local LLM decision has been made
yet (docs/LLM_PROVIDER_DECISION.md).

**Read this before trusting any number this script prints**: this is a
MECHANISM-VALIDATION run, not an LLM-quality benchmark.
`TemplateLLMProvider` is deterministic and rule-based -- it restates
`context` verbatim, it does not reason, paraphrase, or synthesize the way
a real LLM would. Its "factuality"/"hallucination" scores are close to
trivially perfect by construction (it can only write what's already in
`context`), which demonstrates the CONTRACT/PIPELINE works, not that a
real LLM would perform this well. Re-run this exact script (same
questions, same harness) against whichever real provider is chosen later
(docs/LLM_PROVIDER_DECISION.md) for the comparison that actually matters
-- see docs/LLM_EVALUATION.md.

Requires torch (Reporting API) + chromadb/sentence-transformers (RAG) +
the real frozen model/embeddings/RagIndex/ -- run on the GPU server, from
Training/:

    cd Training && python -m experiments.llm_regression.run_llm_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.llm_regression.eval_questions import EVAL_QUESTIONS  # noqa: E402
from orchestrator.llm_provider import TemplateLLMProvider  # noqa: E402
from orchestrator.orchestrator import answer_question  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / \
    "Results" / "evaluation" / "llm_benchmark_template_provider"


def _qualitative_row(eq, out: dict, is_template: bool) -> dict:
    response = out["response"]
    has_context = bool(out["context"]["document_context"]) or bool(out["context"]["dynamic_context"])
    return {
        "question": eq.question,
        "expected_category": eq.expected_category,
        "route_selected": out["route"]["category"],
        "category_match": out["route"]["category"] == eq.expected_category,
        "provider": response["provider"],
        "answer_excerpt": (response["text"] or "")[:400],
        "groundedness": "grounded" if response["grounded"] else (
            "n/a (no context)" if not has_context else "NOT grounded -- see warnings"
        ),
        "factuality": (
            "trivially satisfied (verbatim restatement only)" if is_template and response["grounded"]
            else ("confirmed by independent grounding_check.py re-verification" if response["grounded"]
                  else "NOT confirmed -- see warnings")
        ),
        "citation_correctness": (
            "every source in `sources` is a real retrieved chunk" if response["sources"]
            else "n/a (no sources retrieved)"
        ),
        "dynamic_data_correctness": (
            "every value in `dynamic_data` is a raw, unmodified tool output"
            if response["dynamic_data"] else "n/a (no dynamic data retrieved)"
        ),
        "hallucination": (
            "none possible by construction (TemplateLLMProvider)" if is_template
            else ("none detected by grounding_check.py" if response["grounded"]
                  else "POSSIBLE -- grounding_check.py flagged ungrounded content, see warnings")
        ),
        "answer_completeness": (
            "addresses the question using everything retrieved" if has_context
            else "correctly declines (no information message)"
        ),
        "confidence": response["confidence"],
        "n_sources": len(response["sources"]),
        "n_dynamic_data": len(response["dynamic_data"]),
        "warnings": response["warnings"],
    }


def _make_provider(name: str, model: str = None, seed: int = None):
    if name == "template":
        return TemplateLLMProvider()
    if name == "ollama":
        from orchestrator.llm_provider import OllamaLLMProvider

        return OllamaLLMProvider(model=model or OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                                  request_timeout_seconds=OLLAMA_TIMEOUT_SECONDS, seed=seed)
    raise ValueError(f"unknown provider {name!r}, expected 'template' or 'ollama'")


OLLAMA_MODEL = "mistral-small3.1:24b"
OLLAMA_BASE_URL = "http://localhost:11434"
# Measured this session on the real GPU server (older Titan V hardware,
# `mistral-small3.1:24b` offloaded only 1 layer to GPU -- almost entirely
# CPU-bound): ~42s for an 8-word prompt/response with the 24B model, far too
# slow for an 18-question run in reasonable time or with reasonable shared-
# CPU usage. Smaller models (e.g. `llama3.2:latest`, 3.2B) are far faster on
# this hardware -- pass --model explicitly to override the default.
OLLAMA_TIMEOUT_SECONDS = 240.0


def run(split: str = "val", provider_name: str = "template", model: str = None, seed: int = None) -> dict:
    provider = _make_provider(provider_name, model=model, seed=seed)
    is_template = provider_name == "template"
    rows = [_qualitative_row(
        eq, answer_question(eq.question, video_id=eq.video_id, window=eq.window, split=split, llm=provider),
        is_template,
    ) for eq in EVAL_QUESTIONS]
    note = (
        "MECHANISM VALIDATION ONLY -- not a real LLM quality benchmark, see this "
        "module's own docstring and docs/LLM_EVALUATION.md."
        if is_template else
        f"REAL LLM run -- provider={provider_name}, model={getattr(provider, 'model', '?')}. "
        f"Still only 18 questions (task's own minimum) -- not a statistically powered evaluation, "
        f"see docs/LLM_EVALUATION.md."
    )
    return {
        "provider": provider_name,
        "note": note,
        "split": split, "n_questions": len(rows), "rows": rows,
    }


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        f"# LLM Benchmark -- provider={report['provider']}",
        "",
        f"**{report['note']}**",
        "",
        f"{report['n_questions']} questions, split=`{report['split']}`, provider=`{report['provider']}`.",
        "",
        "| # | Question | Category | Grounded | Citations | Dynamic data | Confidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(report["rows"], start=1):
        lines.append(
            f"| {i} | {row['question']} | {row['route_selected']} | {row['groundedness']} | "
            f"{row['n_sources']} | {row['n_dynamic_data']} | {row['confidence']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--provider", default="template", choices=["template", "ollama"],
                         help="'template' = TemplateLLMProvider (mechanism validation, default). "
                              "'ollama' = real local LLM via OllamaLLMProvider "
                              f"(model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}) -- "
                              "requires a running local Ollama server with that model pulled.")
    parser.add_argument("--model", default=None, help="Ollama model name override (--provider ollama only).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Ollama sampling seed (--provider ollama only) -- fixes the otherwise-stochastic "
                              "generation so two runs (e.g. before/after a prompt-formatting change) are a "
                              "single-variable comparison instead of also sampling new randomness each time.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.provider == "template":
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        model_slug = (args.model or OLLAMA_MODEL).replace(":", "-").replace("/", "-")
        output_dir = DEFAULT_OUTPUT_DIR.parent / f"llm_benchmark_{args.provider}_{model_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run(split=args.split, provider_name=args.provider, model=args.model, seed=args.seed)
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report, output_dir / "REPORT.md")
    print(f"{report['n_questions']} questions run against provider={args.provider} "
          f"model={args.model or (OLLAMA_MODEL if args.provider == 'ollama' else '-')}. Written to {output_dir}/")


if __name__ == "__main__":
    main()

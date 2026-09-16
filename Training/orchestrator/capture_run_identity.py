"""
READ-ONLY audit sidecar: records everything needed to prove that two
fixed-question benchmark runs were produced under the same experimental
conditions.

Written for the 2026-09-07c replication of the `prompt_v2_reading_method`
run. It imports nothing from the pipeline that it could perturb, executes
no model, changes no file the benchmark reads, and is never called by the
benchmark runner -- it is invoked separately, before and/or after a run,
and writes ONE JSON file next to the run artefacts.

What it captures:
  * content hashes of every module that can change an answer
    (prompt / temporal_context / orchestrator / router / tools /
    context_builder / grounding_check / fixed_question_benchmark /
    rag.retrieval), plus the SHA-256 of the rendered system prompt text
    itself and of the 15 frozen question strings;
  * the RAG index identity (embedding model version, per-document content
    hashes, collection) from RagIndex/manifest.json;
  * the live Ollama version, the model digest and its quantization;
  * GPU state at capture time -- every process on every GPU, so a run made
    while another user's job occupied GPU0 can never be silently compared
    with one made on a free card.

`python -m orchestrator.capture_run_identity --label <run-label>` from
Training/. Nothing here writes into Results/ except that one JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

MODULES = [
    "orchestrator/prompt.py",
    "orchestrator/temporal_context.py",
    "orchestrator/orchestrator.py",
    "orchestrator/router.py",
    "orchestrator/tools.py",
    "orchestrator/context_builder.py",
    "orchestrator/grounding_check.py",
    "orchestrator/fixed_question_benchmark.py",
    "orchestrator/llm_provider.py",
    "orchestrator/llm_config.py",
    "orchestrator/run_fixed_question_benchmark.py",
    "rag/retrieval.py",
    "rag/vectorstore.py",
    "rag/inventory.py",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def module_hashes(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in MODULES:
        path = root / rel
        out[rel] = _sha256_file(path) if path.is_file() else "MISSING"
    return out


def prompt_and_question_identity() -> Dict[str, Any]:
    """Hashes the ACTUAL rendered strings, not just the files: a refactor
    that moves text between modules without changing what the LLM receives
    must show as identical here."""
    from orchestrator import prompt as prompt_module
    from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS

    system_prompt = prompt_module.build_system_prompt()
    questions = [q.question for q in FIXED_QUESTIONS]
    anchors = sorted({(q.video_id, q.split, q.window) for q in FIXED_QUESTIONS})
    return {
        "system_prompt_sha256": _sha256_text(system_prompt),
        "system_prompt_chars": len(system_prompt),
        "questions_sha256": _sha256_text("\n".join(questions)),
        "n_questions": len(questions),
        "question_ids": [q.question_id for q in FIXED_QUESTIONS],
        "anchors": [{"video_id": v, "split": s, "window": w} for v, s, w in anchors],
    }


def rag_identity(root: Path) -> Dict[str, Any]:
    manifest_path = root.parent / "RagIndex" / "manifest.json"
    if not manifest_path.is_file():
        return {"manifest_found": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", {})
    return {
        "manifest_found": True,
        "embedding_version": manifest.get("embedding_version"),
        "last_run_at": manifest.get("last_run_at"),
        "vector_db": manifest.get("vector_db"),
        "n_documents": len(documents),
        "documents_sha256": _sha256_text(json.dumps(documents, sort_keys=True)),
    }


def retrieval_defaults() -> Dict[str, Any]:
    """The top_k / filters a benchmark turn actually uses -- read from the
    tool's own signature, never re-declared here."""
    import inspect

    from orchestrator import tools

    signature = inspect.signature(tools.retrieve_documents)
    return {
        name: (param.default if param.default is not inspect.Parameter.empty else "REQUIRED")
        for name, param in signature.parameters.items()
        if name != "persist_dir"
    }


def ollama_identity(model: str, base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    info: Dict[str, Any] = {"base_url": base_url, "reachable": False}
    try:
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=5) as r:
            info["version"] = json.loads(r.read().decode("utf-8")).get("version")
            info["reachable"] = True
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        info["error"] = str(e)
        return info
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as r:
            for entry in json.loads(r.read().decode("utf-8")).get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    info["model"] = entry.get("name")
                    info["digest"] = entry.get("digest")
                    info["details"] = entry.get("details")
                    info["size_bytes"] = entry.get("size")
                    break
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        pass
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5) as r:
            info["loaded"] = json.loads(r.read().decode("utf-8")).get("models", [])
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        pass
    return info


def gpu_state() -> Dict[str, Any]:
    """Read-only `nvidia-smi`. Records EVERY process on EVERY GPU, ours or
    not -- an external job on GPU0 changes Ollama's layer placement, so a
    run must carry that fact with it."""
    def _run(args: List[str]) -> List[str]:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
            return [line for line in out.stdout.strip().splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            return []

    gpus = _run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader"])
    apps = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader"])
    owners = []
    for line in apps:
        pid = line.split(",")[1].strip()
        ps = _run(["ps", "-o", "pid,user,etime,cmd", "--no-headers", "-p", pid])
        owners.extend(ps)
    return {"gpus": gpus, "compute_apps": apps, "process_owners": owners}


def capture(label: str, model: str, output_dir: Path) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent  # Training/
    identity = {
        "run_label": label,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "embryo_ollama_num_ctx_env": os.environ.get("EMBRYO_OLLAMA_NUM_CTX", "(unset)"),
        "module_sha256": module_hashes(root),
        "prompt_and_questions": prompt_and_question_identity(),
        "rag": rag_identity(root),
        "retrieval_defaults": retrieval_defaults(),
        "ollama": ollama_identity(model),
        "gpu": gpu_state(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_identity_{label}.json"
    path.write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="mistral-nemo:12b")
    parser.add_argument("--output-dir", default="../Results/evaluation/event_anomaly_rag_inventory")
    args = parser.parse_args()
    identity = capture(args.label, args.model, Path(args.output_dir))
    pq = identity["prompt_and_questions"]
    print(f"system_prompt_sha256={pq['system_prompt_sha256'][:16]} "
          f"questions_sha256={pq['questions_sha256'][:16]} "
          f"rag_docs={identity['rag'].get('documents_sha256', '?')[:16]} "
          f"ollama={identity['ollama'].get('version')} "
          f"digest={(identity['ollama'].get('digest') or '?')[:16]}")
    for line in identity["gpu"]["gpus"]:
        print("GPU:", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

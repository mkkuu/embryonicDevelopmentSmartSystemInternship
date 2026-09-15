#!/usr/bin/env bash
# Sequential, single-GPU (GPU0 via Ollama's own CUDA_VISIBLE_DEVICES=0) benchmark chain.
cd ~/projects/embryonicDevelopmentSciMLExtension/Training || exit 1
PY=~/miniconda3/envs/embryo_env/bin/python
echo "=== $(date) GPU before"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -1
echo "=== llama3.2:latest under the current pipeline"
$PY -m orchestrator.run_fixed_question_benchmark --model llama3.2:latest --timeout 240 --run-label v1_pipeline_20260907
echo "=== pull llama3.1:8b (one model, sequential)"
ollama pull llama3.1:8b 2>&1 | tail -2
ollama list
echo "=== GPU before candidate"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
echo "=== llama3.1:8b candidate"
$PY -m orchestrator.run_fixed_question_benchmark --model llama3.1:8b --timeout 240 --run-label candidate_20260907
echo "=== GPU after"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1; ollama ps
echo "=== $(date) DONE"

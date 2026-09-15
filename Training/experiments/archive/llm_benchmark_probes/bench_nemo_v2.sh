#!/usr/bin/env bash
cd ~/projects/embryonicDevelopmentSciMLExtension/Training || exit 1
echo "=== $(date) GPU0 before"; nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv,noheader
md5sum orchestrator/prompt.py orchestrator/temporal_context.py orchestrator/orchestrator.py orchestrator/fixed_question_benchmark.py
CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/embryo_env/bin/python -m orchestrator.run_fixed_question_benchmark --model mistral-nemo:12b --timeout 240 --run-label prompt_v2_reading_method
echo "=== GPU0 after"; nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv,noheader; ollama ps
echo "=== $(date) DONE"

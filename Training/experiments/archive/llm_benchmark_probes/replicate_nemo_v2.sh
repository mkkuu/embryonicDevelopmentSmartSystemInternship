#!/usr/bin/env bash
# Controlled replication of the `prompt_v2_reading_method` run.
# NOTHING in the pipeline is modified. Conditions reproduced verbatim:
#   mistral-nemo:12b, num_ctx=16384 (default), timeout=240, no seed,
#   no temperature, split=val, Patient_319/window 156, prompt v2,
#   CUDA_VISIBLE_DEVICES="" for THIS process only (RAG embedder on CPU --
#   the same condition under which the counted v2 run was produced;
#   Ollama itself is a separate systemd service pinned to GPU0).
cd ~/projects/embryonicDevelopmentSciMLExtension/Training || exit 1
PY=~/miniconda3/envs/embryo_env/bin/python

for REP in rep1 rep2 rep3; do
  LABEL="prompt_v2_reading_method_${REP}"
  echo "################ ${LABEL} — $(date -Is)"
  echo "--- GPU0 before"
  nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv,noheader
  echo "--- identity capture"
  CUDA_VISIBLE_DEVICES="" $PY -m orchestrator.capture_run_identity --label "${LABEL}" 2>&1 | grep -v "HF_TOKEN"
  echo "--- run"
  CUDA_VISIBLE_DEVICES="" $PY -m orchestrator.run_fixed_question_benchmark \
      --model mistral-nemo:12b --timeout 240 --run-label "${LABEL}" 2>&1 | grep -v "HF_TOKEN\|Loading weights"
  echo "--- GPU0 after"
  nvidia-smi -i 0 --query-gpu=memory.used,memory.free --format=csv,noheader
  ollama ps
  echo "################ ${LABEL} DONE — $(date -Is)"
done
echo "ALL_REPLICATIONS_DONE"

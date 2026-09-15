#!/usr/bin/env bash
# Gated launcher for the Validator v1.3 paired benchmark -- NOT started automatically.
# Refuses to launch unless GPU0 is free of external jobs AND Ollama places the whole
# model in VRAM (gpu_fraction == 1.0, the v1.2 reference). Records VRAM before/after load.
set -euo pipefail
cd ~/projects/embryonicDevelopmentSciMLExtension
OUT=Results/evaluation/validator_experiment
LABEL=${1:-validator_v1_3b}
PY=~/miniconda3/envs/embryo_env/bin/python
declare -A EXPECT=(
  [Training/validator/extraction.py]=8f003661ca2a44c700d99bb398a14db6885d37f8a22496b50bc27c572ef6b25f
  [Training/validator/provenance.py]=a3d1871f2ac39099ab8f0d4ff47cc86e4deeef704314f495d9f5a94a3d590dfc
  [Training/validator/rules.py]=77210c28b31e8c950865a298895e9e9cd49235f8ab9ffa853435d4e953156de1
  [Training/validator/validator.py]=fb5325c48c2c551fa81c6ee8788bd48da753b26dd2dfa3b17496e0fa1902187d
  [Training/orchestrator/scientific_validation.py]=ae2e96429d00881828f7026531c52b7d01dddb89a7e8908945b47a69e42b40ca
  [Tests/validator/test_validator_v13_precision.py]=f1fa77909c88316db147a4579e9945162c0a78e2b466a8a05264ba2196552ff2
)
echo "[gate] $(date -u) verifying v1.3 hashes"
for f in "${!EXPECT[@]}"; do
  h=$(sha256sum "$f" | cut -d' ' -f1); [ "$h" = "${EXPECT[$f]}" ] || { echo "HASH MISMATCH $f"; exit 2; }
done
echo "[gate] hashes OK"
[ ! -e "$OUT/paired_validator_off_vs_on_mistral-nemo_12b_${LABEL}.json" ] || { echo "artefact $LABEL exists"; exit 2; }
# GPU0 must carry no external compute process
ext=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v ollama || true)
[ -z "$ext" ] || { echo "GPU0 busy: $ext"; exit 3; }
before=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[gate] GPU0 memory.used before load = ${before} MiB"
curl -s localhost:11434/api/generate -d '{"model":"mistral-nemo:12b","prompt":"","options":{"num_ctx":16384}}' >/dev/null
frac=$(curl -s localhost:11434/api/ps | python3 -c 'import json,sys;m=[x for x in json.load(sys.stdin)["models"] if x["name"]=="mistral-nemo:12b"][0];print(round(m["size_vram"]/m["size"],4))')
after=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[gate] GPU0 memory.used after load = ${after} MiB ; gpu_fraction = ${frac}"
[ "$frac" = "1.0" ] || { echo "PLACEMENT NOT COMPARABLE (gpu_fraction=$frac, v1.2=1.0) -- not launching"; exit 4; }
echo "[gate] preflight (no LLM)"; ( cd Training && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. $PY /tmp/preflight_v1_3.py 2>&1 | grep -E "^(DIFF|PREFLIGHT|LEAK)" | grep -v -E "dynamic_context_sha256|user_prompt" ) || true
printf 'gpu0_mem_before_load_mib=%s\ngpu0_mem_after_load_mib=%s\ngpu_fraction=%s\ngate_time_utc=%s\n' "$before" "$after" "$frac" "$(date -u +%FT%TZ)" > "$OUT/run_${LABEL}.gate.txt"
cd Training
CUDA_VISIBLE_DEVICES="" nohup setsid $PY -m orchestrator.run_validator_experiment --model mistral-nemo:12b --timeout 240 --replications 3 --run-label "$LABEL" > "../$OUT/run_${LABEL}.log" 2>&1 < /dev/null &
echo $! > "../$OUT/run_${LABEL}.pid"; echo "[gate] launched PID $(cat ../$OUT/run_${LABEL}.pid) at $(date -u)"

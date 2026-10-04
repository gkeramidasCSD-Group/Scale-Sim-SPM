#!/usr/bin/env bash
# smm/run_all_models.sh
#
# Runs the full SMM benchmark (baseline x3 + Hom + Het, across all 5 of the
# paper's own GLB sizes) for every model currently available, one command,
# meant to be left running unattended.
#
# Resilience, learned the hard way (see smm/docs/smm_verification.md /
# smm_implementation.md): each (model, GLB size) combination is its own
# `python3` process, not one long-lived process looping internally -- so a
# crash (OOM, or anything else) on one combination does NOT take down the
# rest of the sweep, and whatever already finished is safely on disk before
# the next combination even starts. Output is unbuffered (`python3 -u`) and
# appended immediately, so a hard crash (process kill, machine reboot) loses
# at most the one combination in progress, never anything already done.
#
# Usage:
#   bash smm/run_all_models.sh                 # foreground
#   nohup bash smm/run_all_models.sh &          # survives terminal close
#   tmux new -d 'bash smm/run_all_models.sh'    # recommended over SSH: survives disconnects too
#
# Check progress any time with:
#   tail -f smm/results/all_models_<timestamp>/_summary.log
#   tail -f smm/results/all_models_<timestamp>/<Model>.log

set -uo pipefail   # no -e: one failing combination must not stop the rest

cd "$(dirname "$0")/.."   # repo root, regardless of cwd when invoked

MODELS=(
  "ResNet18:smm/topologies/resnet18_same_padded.csv"
  "MobileNet:cosma/_exported/MobileNet/model.json"
  "MobileNetV2:cosma/_exported/MobileNetV2/model.json"
  "GoogLeNet:cosma/_exported/GoogLeNet/model.json"
  "EfficientNetB0ish:cosma/_exported/efficient50/model.json"
)
GLB_SIZES=(64 128 256 512 1024)
CONFIG="configs/scale_smm.cfg"

OUT_ROOT="smm/results/all_models_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/_summary.log"

echo "=== SMM full sweep starting $(date) ===" | tee -a "$SUMMARY"
echo "Output: $OUT_ROOT" | tee -a "$SUMMARY"
echo "Models: ${#MODELS[@]}  GLB sizes: ${GLB_SIZES[*]}" | tee -a "$SUMMARY"

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  model_path="${entry#*:}"

  if [ ! -f "$model_path" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name -- $model_path not found" | tee -a "$SUMMARY"
    continue
  fi

  model_log="$OUT_ROOT/${name}.log"
  model_start=$(date +%s)

  for glb in "${GLB_SIZES[@]}"; do
    echo "### $name GLB=${glb}kB starting $(date) ###" >> "$model_log"
    combo_start=$(date +%s)

    python3 -u smm/run_smm.py \
      --model "$model_path" \
      --config "$CONFIG" \
      --glb_kb "$glb" \
      --objective accesses \
      --out "$OUT_ROOT/${name}_traces" >> "$model_log" 2>&1
    exit_code=$?

    combo_secs=$(( $(date +%s) - combo_start ))
    echo "### $name GLB=${glb}kB finished exit=$exit_code (${combo_secs}s) $(date) ###" >> "$model_log"
    echo "[$(date +%H:%M:%S)] $name  GLB=${glb}kB  exit=$exit_code  (${combo_secs}s)" | tee -a "$SUMMARY"
  done

  model_secs=$(( $(date +%s) - model_start ))
  echo "[$(date +%H:%M:%S)] --- $name done (${model_secs}s total) ---" | tee -a "$SUMMARY"
done

echo "=== SMM full sweep finished $(date) ===" | tee -a "$SUMMARY"
echo "Per-model logs: $OUT_ROOT/<Model>.log" | tee -a "$SUMMARY"

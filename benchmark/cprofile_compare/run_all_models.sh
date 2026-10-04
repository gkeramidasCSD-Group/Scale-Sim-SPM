#!/bin/bash
# Profile every model x bandwidth mode (vanilla vs. optimized) with run_network.sh, on one
# hardware configuration, for the all-models breakdown figure (paper/make_paper_assets.py).
#
# Settings (environment variables, defaults in brackets):
#   VANILLA_REPO, OPTIMIZED_REPO, VANILLA_PYTHON, OPTIMIZED_PYTHON   passed to run_network.sh
#   ARRAY [32]  SRAM_KB [64]    the same hardware for every run (SCALE-Sim's default config)
#   MODES ["CALC USER"]
#   MODELS    topology CSVs relative to OPTIMIZED_REPO, longest first
#             [transformer_fwd, Googlenet, mobilenet, alexnet, vit_b]
#   JOBS [2]  configurations running at once; each runs vanilla + optimized in parallel,
#             so JOBS=2 means 4 simulator processes
#
# Resumable: a configuration whose profiles/<name>_report.md exists is skipped, so rerunning
# after an interruption only does what's missing. Per-run logs: profiles/logs/<name>.log.
#
# Example (server):
#   cd /data/grizos/Scale-Sim-SPM
#   VANILLA_REPO=/data/grizos/SCALE-Sim VANILLA_PYTHON=/data/grizos/SCALE-Sim/venv/bin/python \
#   OPTIMIZED_PYTHON=/data/grizos/Scale-Sim-SPM/venv/bin/python JOBS=2 \
#   nohup benchmark/cprofile_compare/run_all_models.sh > cprofile_all_models.log 2>&1 &
D=$(dirname "$(readlink -f "$0")")
export OPTIMIZED_REPO=${OPTIMIZED_REPO:-$(readlink -f "$D/../..")}
export ARRAY=${ARRAY:-32} SRAM_KB=${SRAM_KB:-64} PARALLEL=1
MODES=${MODES:-"CALC USER"}
MODELS=${MODELS:-"topologies/transformer/transformer_fwd.csv topologies/conv_nets/Googlenet.csv topologies/conv_nets/mobilenet.csv topologies/conv_nets/alexnet.csv topologies/ispass25_models/vit_b.csv"}
JOBS=${JOBS:-2}
mkdir -p "$D/profiles/logs"

tasks=()
for topo in $MODELS; do
  path=$topo; [ -f "$path" ] || path=$OPTIMIZED_REPO/$topo
  [ -f "$path" ] || { echo "topology not found: $topo"; exit 1; }
  for bw in $MODES; do
    name=$(basename "$path" .csv | tr A-Z a-z)_a${ARRAY}_s${SRAM_KB}_$(echo "$bw" | tr A-Z a-z)
    if [ -f "$D/profiles/${name}_report.md" ]; then
      echo "skip $name (already done)"
    else
      tasks+=("$path $bw $name")
    fi
  done
done
echo "${#tasks[@]} configuration(s) to run, $JOBS at a time: array ${ARRAY}x${ARRAY}, ${SRAM_KB} KB"

run_one() {
  local path=$1 bw=$2 name=$3
  echo "$(date '+%F %T') start $name"
  TOPO=$path BW=$bw "$D/run_network.sh" > "$D/profiles/logs/$name.log" 2>&1
  if grep -q '^DONE' "$D/profiles/logs/$name.log"; then
    echo "$(date '+%F %T') done  $name  ($(grep -c 'identical' "$D/profiles/${name}_report.md")/4 outputs identical)"
  else
    echo "$(date '+%F %T') FAILED $name -- see profiles/logs/$name.log"
  fi
}
export -f run_one
export D

printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -L 1 bash -c 'run_one $0 $1 $2'
echo "all done; now run: python3 $D/paper/make_paper_assets.py"

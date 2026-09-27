#!/bin/bash
# Full-network cProfile of vanilla vs. optimized SCALE-Sim, then correctness check + % breakdown.
#
# Everything is set through environment variables (defaults in brackets):
#   VANILLA_REPO      vanilla SCALE-Sim checkout             [/home/george/Music/SCALE-Sim]
#   OPTIMIZED_REPO    optimized checkout                     [the repo this script lives in]
#   VANILLA_PYTHON    python for the vanilla run (e.g. its venv)    [python3]
#   OPTIMIZED_PYTHON  python for the optimized run                  [python3]
#   TOPO              topology CSV            [$OPTIMIZED_REPO/topologies/conv_nets/Googlenet.csv]
#   ARRAY             array height = width                   [16]
#   SRAM_KB           ifmap = filter = ofmap SRAM size, KB   [16]
#   BW                InterfaceBandwidth: CALC or USER       [CALC]
#   DATAFLOW          ws | os | is                           [ws]
#   CFG               use this .cfg instead of generating one from the four values above
#   INPUT_TYPE        conv | gemm   [auto: gemm if the topology header is "Layer,M,N,K"]
#   OUT               scratch dir for run outputs            [/tmp/scalesim_cprofile/<name>]
#   PARALLEL          1 = run both versions at the same time (server); 0 = one after the other [0]
#   MIN_FREE_MB       watchdog: kill a run if MemAvailable drops below this   [700]
#
# Example (server, USER mode, small array/SRAM, both at once, survives logout):
#   VANILLA_REPO=/data/grizos/SCALE-Sim VANILLA_PYTHON=/data/grizos/SCALE-Sim/venv/bin/python \
#   OPTIMIZED_PYTHON=/data/grizos/Scale-Sim-SPM/venv/bin/python \
#   TOPO=/data/grizos/Scale-Sim-SPM/topologies/conv_nets/mobilenet.csv \
#   ARRAY=16 SRAM_KB=16 BW=USER PARALLEL=1 \
#   nohup benchmark/cprofile_compare/run_network.sh > cprofile_mobilenet_user.log 2>&1 &
#
# Results land in benchmark/cprofile_compare/profiles/<name>_*:
#   <name>_{vanilla,optimized}.pstats    raw profiles
#   <name>_report.md                     correctness check + % breakdown + changed-function table
#   <name>_{vanilla,optimized}_{tottime,cumtime}.txt   top-40 cProfile text reports
# where <name> = <topology>_a<ARRAY>_s<SRAM_KB>_<bw>[_<dataflow> if not ws], e.g. googlenet_a16_s16_calc.
D=$(dirname "$(readlink -f "$0")")
VANILLA_REPO=${VANILLA_REPO:-/home/george/Music/SCALE-Sim}
OPTIMIZED_REPO=${OPTIMIZED_REPO:-$(readlink -f "$D/../..")}
VANILLA_PYTHON=${VANILLA_PYTHON:-python3}
OPTIMIZED_PYTHON=${OPTIMIZED_PYTHON:-python3}
TOPO=$(readlink -f "${TOPO:-$OPTIMIZED_REPO/topologies/conv_nets/Googlenet.csv}")
ARRAY=${ARRAY:-16}
SRAM_KB=${SRAM_KB:-16}
BW=$(echo "${BW:-CALC}" | tr a-z A-Z)
DATAFLOW=${DATAFLOW:-ws}
PARALLEL=${PARALLEL:-0}
MIN_FREE_MB=${MIN_FREE_MB:-700}

for r in "$VANILLA_REPO" "$OPTIMIZED_REPO"; do
  [ -d "$r/scalesim" ] || { echo "not a SCALE-Sim checkout: $r"; exit 1; }
done
[ -f "$TOPO" ] || { echo "topology not found: $TOPO"; exit 1; }
case "$BW" in CALC|USER) ;; *) echo "BW must be CALC or USER"; exit 1;; esac
if [ -z "$INPUT_TYPE" ]; then
  second_col=$(head -1 "$TOPO" | tr -d '\r\357\273\277' | cut -d, -f2 | tr -d ' ')
  [ "$second_col" = M ] && INPUT_TYPE=gemm || INPUT_TYPE=conv
fi

if [ -n "$CFG" ]; then
  CFG=$(readlink -f "$CFG")
  NAME=$(basename "$CFG" .cfg)
else
  NAME=$(basename "$TOPO" .csv | tr A-Z a-z)_a${ARRAY}_s${SRAM_KB}_$(echo "$BW" | tr A-Z a-z)
  [ "$DATAFLOW" != ws ] && NAME=${NAME}_$DATAFLOW
  mkdir -p "$D/configs"
  CFG=$D/configs/$NAME.cfg
  # Same template as benchmark/config_gen.py; only the parameters above vary.
  cat > "$CFG" <<CFGEOF
[general]
run_name = $NAME

[architecture_presets]
ArrayHeight:    $ARRAY
ArrayWidth:     $ARRAY
IfmapSramSzkB:   $SRAM_KB
FilterSramSzkB:  $SRAM_KB
OfmapSramSzkB:   $SRAM_KB
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 10
Dataflow : $DATAFLOW
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: $BW
UseRamulatorTrace: False
CFGEOF
fi
OUT=${OUT:-/tmp/scalesim_cprofile/$NAME}
RUN_NAME=$(awk -F'=' '/^run_name/{gsub(/ /,"",$2); print $2}' "$CFG")

echo "run:       $NAME"
echo "config:    $CFG"
echo "topology:  $TOPO  (input type: $INPUT_TYPE)"
echo "vanilla:   $VANILLA_REPO  ($VANILLA_PYTHON)"
echo "optimized: $OPTIMIZED_REPO  ($OPTIMIZED_PYTHON)"
echo "outputs:   $OUT   parallel=$PARALLEL"
echo

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" "$D/profiles"
TIME_V=""; [ -x /usr/bin/time ] && TIME_V="/usr/bin/time -v"

launch() {  # launch <tag> <repo> <python>; sets LAUNCHED_PID
  local tag=$1 repo=$2 py=$3
  rm -rf "$OUT/$tag"
  (cd "$repo" && exec $TIME_V "$py" "$D/profile_network.py" "$repo" "$tag" "$OUT/$tag" "$CFG" "$TOPO" "$INPUT_TYPE") \
    > "$OUT/$tag.log" 2>&1 &
  LAUNCHED_PID=$!
  echo "[$tag] started (pid $LAUNCHED_PID), per-layer progress: $OUT/$tag/layers.log"
}

watch_pids() {  # watch_pids <pid>...; kills everything if RAM runs low
  while :; do
    local alive=0
    for p in "$@"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt "$MIN_FREE_MB" ]; then
      echo "KILLED: MemAvailable=${avail}MB < ${MIN_FREE_MB}MB"
      for p in "$@"; do pkill -P "$p"; kill "$p" 2>/dev/null; done
      break
    fi
    sleep 1
  done
  for p in "$@"; do wait "$p"; done
}

finish() {  # finish <tag>
  local tag=$1
  grep -h "TOTAL\|Traceback\|Error" "$OUT/$tag.log"
  grep -h "Elapsed\|Maximum resident" "$OUT/$tag.log" | sed "s/^/[$tag] /"
  cp "$OUT/$tag/$tag.pstats" "$D/profiles/${NAME}_$tag.pstats" 2>/dev/null
}

if [ "$PARALLEL" = 1 ]; then
  launch optimized "$OPTIMIZED_REPO" "$OPTIMIZED_PYTHON"; P1=$LAUNCHED_PID
  launch vanilla "$VANILLA_REPO" "$VANILLA_PYTHON"; P2=$LAUNCHED_PID
  watch_pids $P1 $P2
  finish optimized; finish vanilla
else
  # Optimized first: it's shorter and shows the per-layer RAM peak early.
  launch optimized "$OPTIMIZED_REPO" "$OPTIMIZED_PYTHON"; watch_pids $LAUNCHED_PID; finish optimized
  launch vanilla "$VANILLA_REPO" "$VANILLA_PYTHON"; watch_pids $LAUNCHED_PID; finish vanilla
fi

if [ ! -f "$D/profiles/${NAME}_vanilla.pstats" ] || [ ! -f "$D/profiles/${NAME}_optimized.pstats" ]; then
  echo "a run did not finish -- see $OUT/*.log"; exit 1
fi

# Correctness: reports byte-compared, trace CSVs compared by md5 (they're deleted per layer).
REPORT=$D/profiles/${NAME}_report.md
{
  echo "# cProfile: $NAME"
  echo
  echo "- topology: \`$TOPO\` ($INPUT_TYPE)"
  echo "- config: array ${ARRAY}x${ARRAY}, SRAM ${SRAM_KB} KB, InterfaceBandwidth $BW, dataflow $DATAFLOW (\`$CFG\`)"
  echo "- vanilla: \`$VANILLA_REPO\`; optimized: \`$OPTIMIZED_REPO\`; parallel=$PARALLEL; host $(hostname); $(date -Iseconds)"
  for tag in vanilla optimized; do
    echo "- $tag: $(grep -h TOTAL "$OUT/$tag.log" | sed 's/.*TOTAL //')," \
         "peak RSS $(awk '/Maximum resident/{printf "%.0f MB", $NF/1024}' "$OUT/$tag.log")"
  done
  echo
  echo "## Same simulated result?"
  echo
  for f in COMPUTE_REPORT BANDWIDTH_REPORT DETAILED_ACCESS_REPORT; do
    cmp -s "$OUT/vanilla/$RUN_NAME/$f.csv" "$OUT/optimized/$RUN_NAME/$f.csv" \
      && echo "- $f.csv: identical" || echo "- $f.csv: **DIFFERS**"
  done
  diff -q "$OUT/vanilla/trace_md5.txt" "$OUT/optimized/trace_md5.txt" >/dev/null \
    && echo "- trace CSVs: identical ($(wc -l < "$OUT/vanilla/trace_md5.txt") files, md5)" \
    || echo "- trace CSVs: **DIFFER**"
  echo
  echo "## Where the time goes"
  echo
  python3 "$D/breakdown.py" "$D/profiles/${NAME}_vanilla.pstats" "$D/profiles/${NAME}_optimized.pstats"
  echo
  echo "## Changed functions"
  echo
  echo '```'
  python3 "$D/compare.py" "${NAME}_"
  echo '```'
} > "$REPORT"
cat "$REPORT"
echo
echo "report: $REPORT"
echo DONE

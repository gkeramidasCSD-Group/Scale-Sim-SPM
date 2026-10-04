#!/bin/bash
# Profile vanilla, then optimized, one at a time (never in parallel: this laptop has 7GB RAM).
# A watchdog kills the running profile if MemAvailable drops below 700MB.
# Override the repo paths with VANILLA_REPO / OPTIMIZED_REPO, and the output dir with OUT.
D=$(dirname "$(readlink -f "$0")")
VANILLA_REPO=${VANILLA_REPO:-/home/george/Music/SCALE-Sim}
OPTIMIZED_REPO=${OPTIMIZED_REPO:-$(readlink -f "$D/../..")}
OUT=${OUT:-/tmp/scalesim_cprofile_compare}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -rf "$OUT"; mkdir -p "$OUT"
for pair in "vanilla:$VANILLA_REPO" "optimized:$OPTIMIZED_REPO"; do
  tag=${pair%%:*}; repo=${pair#*:}
  (cd "$repo" && exec /usr/bin/time -v python3 "$D/profile_driver.py" "$repo" "$tag" "$OUT/$tag") > "$OUT/$tag.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt 700 ]; then pkill -P $pid; kill $pid; echo "[$tag] KILLED: MemAvailable=${avail}MB"; break; fi
    sleep 1
  done
  wait $pid
  grep -h "^\[$tag\]\|Traceback\|Error" "$OUT/$tag.log"
  grep -h "Elapsed\|Maximum resident" "$OUT/$tag.log" | sed "s/^/[$tag] /"
  cp "$OUT/$tag/$tag.pstats" "$D/profiles/" 2>/dev/null
done

# Same simulated result? Byte-compare every report/trace CSV.
n=0; bad=0
for bw in CALC USER; do
  for f in $(cd "$OUT/vanilla/$bw" && find . -name '*.csv' | sort); do
    n=$((n+1))
    cmp -s "$OUT/vanilla/$bw/$f" "$OUT/optimized/$bw/$f" || { bad=$((bad+1)); echo "DIFF $bw/$f"; }
  done
done
echo "output CSVs compared=$n differing=$bad"
python3 "$D/compare.py"

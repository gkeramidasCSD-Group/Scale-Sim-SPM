# Testbench Parameters: COSMA vs. OnSRAM vs. SMM (Zouzoula)

Parameters to sweep so each paper's real strengths/weaknesses show up in
SCALE-Sim, instead of only whichever regime that paper's own evaluation
happened to pick. Grounded in each paper's own methodology section.

**The testbench needs a traffic model that captures re-fetches inside a layer under a finite buffer and also residency across layers. The most defensible option is probably an analytic per-layer tiling model, with SCALE-Sim supplying compute cycles only. We need to decide this before running any sweep.**

## Parameters

| Parameter | Levels to test | Why it matters |
|---|---|---|
| SPM / GLB capacity | Below `M_R` (COSMA's structural minimum) -> `M_P` (peak footprint) -> 2-4x `M_P` | Each paper's own sweet spot differs (Zouzoula: 64KB-1MB, OnSRAM: 1-4MB). The `[M_R, M_P]` band is where eviction pressure actually bites (see below). |
| Off-chip bandwidth | 8-64 GB/s | Zouzoula fixes this and never sweeps it. At high BW everything converges to baseline (OnSRAM Fig. 13) — this is the axis that shows *when* SPM management stops mattering at all. |
| Array size | 8x8 / 16x16 / 32x32 (Zouzoula: 16x16, OnSRAM's paper config: 39x39) | A bigger array finishes compute faster, pushing layers memory-bound sooner — same effect as bandwidth, opposite direction. |
| Dataflow | WS / OS / IS | Zouzoula's baseline is OS specifically. COSMA doesn't simulate array timing at all (pure byte-count ILP), so it's dataflow-agnostic by construction — sweeping this checks whether tiling-policy papers are dataflow-sensitive in a way graph-level papers aren't. |
| Data width | 8 / 16 / 32-bit | Zouzoula shows a large effect (Fig. 7) since it just rescales tensor bytes against a fixed capacity — the same tradeoff applies to COSMA/OnSRAM even though their papers don't test it. |
| Memory partitioning | (1) fixed separate buffers (2) unified GLB, activations only (3) unified GLB, activations+weights (4) unified activation SPM + dedicated weight SPM | (1)/(2) = Zouzoula's baseline/Hom-Het, (3) = COSMA's "+parameter" setting, (4) = OnSRAM's Fig. 10 variant. Each paper studies a different one of these four. |
| Workload topology | sequential chain -> residual/branchy -> NAS-irregular | OnSRAM's entire motivating axis. Also the axis most likely to expose COSMA's real edge: its ILP handles arbitrary graphs optimally, Zouzoula's Algorithm 1 has zero cross-branch reasoning. |
| Layer-type mix | plain conv / depthwise / pointwise / grouped / FC | Zouzoula's own policies 1-3 don't support depthwise (only 4/5 do) — an all-plain-conv workload set hides this limitation. |
| Model scale | small (CIFAR-sized) vs. real (ImageNet-sized) | Solve-time and "how tight is the budget" claims aren't comparable across this gap — control for it. |



## Metrics to evaluate after each run

Report the same set for all three schemes — don't grade each paper only on
the metric its own paper happens to emphasize.

**Quality**

| Metric | What it tells you |
|---|---|
| DRAM traffic (bytes), split ifmap/filter/ofmap, read/write | Raw off-chip cost; what COSMA directly optimizes |
| Latency (cycles) | What Zouzoula and OnSRAM ultimately optimize |
| Speedup vs. shared "no SPM management" baseline | Normalizes across SPM sizes/configs — raw cycles alone aren't comparable |
| Compute-bound vs. memory-stall cycle fraction | Explains *why* a curve plateaus (ties to the bandwidth/array-size sweep) |
| Energy (CACTI, applied uniformly to all three's traffic) | DRAM access is ~50-100x SRAM access energy |

**Cost**

| Metric | What it tells you |
|---|---|
| Offline solve/compile time | COSMA's ILP (seconds, can time out) vs. Zouzoula's closed-form estimate (near-instant) vs. OnSRAM-Static's graph pass (ms-low seconds) |
| Feasibility under extreme tightness | Hard fail (COSMA below `M_R`) vs. documented fallback (Zouzoula, no policy fits) vs. silent near-no-op (OnSRAM pins almost nothing) |

## The sharpest differentiator: force eviction under pressure

Pick SPM sizes strictly between `M_R` and `M_P`. In that band, a
still-needed tensor *must* be evicted to make room for something else right
now:

- Zouzoula's Algorithm 1 structurally never evicts — it only ever selects a
  policy that already fits.
- OnSRAM-Static never evicts a live tensor either — it just declines to pin
  one in the first place (FoM greedy skip).
- COSMA alone has real spill/retrieve with optimal ILP replacement.

This is where the three papers mechanically diverge, not just numerically —
already reproduced once in this repo (small custom DenseNet fixture @
550KB: 0.9985x, net slower, under COSMA's own real spill/retrieve cost).


## A way to make the sweep affordable
Once all three papers use analytic traffic, SCALE-Sim's compute cycles depend only on the model, the array size and the dataflow. Simulate once per combination of those three and cache the result. Everything else (capacity, bandwidth, data width, partitioning) then becomes a cheap re-computation instead of a new SCALE-Sim run. That turns a full grid over all the parameters, which is days of simulation on this machine, into minutes.

### We need to :
  - Settle on the shared cost model.
  - Cache compute per model, array size and dataflow.
  - Sweep capacity, bandwidth, topology and data width one factor at a time around the paper config.
  - Leave out partitioning options (3)/(4) and eager execution until something implements them.


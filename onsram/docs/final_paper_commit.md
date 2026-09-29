- **Non-conv layers are now costed** (both COSMA and OnSRAM): DENSE runs
  through SCALE-Sim as a 1×1 conv; ADD, CONCAT, pooling, PAD, SUB/MUL,
  REDUCE_MEAN and SOFTMAX are costed as pure data movement (each input
  read once, each output written once), with the same SPM rules as conv
  layers. Before, they cost 0 in both runs.
- **Unfused models**: `spm_common/unfuse_model.py` rebuilds the separate
  BatchNorm/BiasAdd/ReLU nodes that TFLite folded into the convs (ported
  from the reference implementation's `onsram_bw/onsram_unfused.py`, with
  ReLU only where the model had one and BatchNorm only for networks that
  had it). Output: `cosma/_exported/<model>_unfused/model.json`.
- **We dont use the memory system of scale-sim:**
  For taking the measurements of onsram we dont use scale-sim memory system, since It's the paper's own model. §6 is bandwidth-centric: "each data element is fetched once", no bank conflicts, no DRAM latency effects. So we model this , a one time fetch. But scale-sim still runs the compute cycle count.
  
### What "compute time" is on each side

- **Our side:** SCALE-Sim simulates the 39×39 systolic array cycle by
  cycle, including how the layer is cut into tiles and how long each tile
  takes to fill and drain.
- **An ideal array:** `multiplications ÷ 1,521 units`, every unit busy
  every cycle. This is what "3 TFLOP" means on paper.
- **The paper:** compute time from a performance model calibrated to its
  real chip (§6, refs [10]/[93]). The paper doesn't print the formula, but
  its results only work if compute is a small part of the time (∞ SPM of
  5.17× on MobileNet means compute ≤ 1/5 of the baseline).


### The bigger reason: the paper's graphs have BatchNorm/ReLU as separate nodes

**What the paper states, and what is our inference.** The paper doesn't
label its results "fused" or "unfused". What it states:
- OnSRAM-Static takes "an optimized graph of the DNN from the DL framework
  (e.g., TensorFlow's ProtoBuf graph)" (§4). In such graphs BatchNorm,
  ReLU and BiasAdd are separate ops.
- §7.1.3 / Fig. 11 report ReLU and BatchNorm as their own layer types, with
  their inputs pinned: "the activation-bound layers such as ReLU, Pooling,
  and 'others' get pinned… the performance benefits stem from the fact
  that the data transfer times for these critical layers are drastically
  reduced."
- §7.1.6 "Comparison with Layer Fusion", one paragraph: "We implemented
  layer fusion for OnSRAM-Static combining Conv/MatMul layers with
  BatchNorm-ReLU-BiasAdd. With that, OnSRAM-Static achieves a speedup of
  1.01–2.17× (average 1.31×) relative to No SPM Mgmt with layer fusion."
  A range and an average across models; no per-model numbers.
  Paper's OnSRAM-Static values read this way: AlexNet 1.02, VGG16 1.02,




## All four original causes are fixed (status as of 2026-09-28):

| # | Cause | Direction it pushed our OnSRAM speedup | Status |
|---|---|---|---|
| 1 | Depthwise compute ~40–60× too slow | **down, a lot** (capped MobileNet at 1.28×) | Fixed 2026-09-24: channels-across-columns mapping, in both COSMA's and OnSRAM's topology builders |
| 2 | Overwrite hand-off reads pinned inputs from DRAM | down | Fixed 2026-09-24: `'H'` hand-off reads served from SPM (OnSRAM only) |
| 3a | float32 SPM budget / 1-byte DRAM elements vs FP16 | down | Fixed 2026-09-25: FP16 everywhere (OnSRAM only) |
| 3b | Re-reads, partial-sum write-backs | mixed | Fixed 2026-09-25: fetch-once traffic (OnSRAM only) |
| 4 | Every output write free | **up** | Fixed 2026-09-24: only pinned outputs stay on-chip (OnSRAM only) |

What's left is described above under "Why compute takes so much of the
execution time": SCALE-Sim's array needs 1.6–2× the ideal compute time,
which caps the memory-bound mobile networks (MobileNetV1, ResNeXt). The
open choice is the same one Cause 1 raised: keep SCALE-Sim's systolic-array
compute (honest for this array, not the paper's hardware), switch to ideal
FLOPs ÷ peak (closer to the paper's ideal-dataflow assumption; its actual
compute model is calibrated to its chip and not published), or report both. That affects all
three papers the same way, so it should be one shared decision.

## What we keep, and what changes for the three-paper testbench

Two different goals, two different rules:
- **Reproducing the OnSRAM paper**: evaluate OnSRAM with the paper's own
  model. The first table is that setup, as it stands on 2026-09-28.
- **Comparing COSMA, OnSRAM and SMM**: each algorithm still makes its own
  paper-faithful decisions (what to pin, spill, retrieve; the schedule),
  but every plan is costed by one shared evaluator with the same rules.
  Otherwise a difference in numbers mixes the algorithms with the cost
  models. The second table is what that needs.

### Keep: OnSRAM setup that reproduces the paper

| Area | What we keep | Paper basis | Where |
|---|---|---|---|
| Figure of Merit | Eq. 1 with α=0.4, β=0.5, γ=0.1 | §4.2, §6 | `onsram_helpers/fom.py` |
| Schedule | BFS-DFS hybrid: activation-bound children to the head of the queue | §4.1 | `onsram_helpers/scheduling.py` |
| Pinning | Greedy by FoM, whole lifetime only (inclusive), activations only; weights streamed, never pinned | §3.2, §4.2 | `onsram_helpers/pinning.py` |
| Overwrite Optimization | Reclaimed tensor vacates one step early; its last read is an `'H'` hand-off served from the SPM | §4.2 | `pinning.build_handoff_action()` |
| SPM | One shared 2 MB scratchpad | §3.1, §6 | `--spm-mb 2`, `SpmAllocator` |
| Hardware | 39×39 array at 1 GHz (≈3.04 TFLOP), 32 B/cycle DRAM, batch 1 | §6 | `configs/scale_onsram.cfg` |
| Precision | FP16, 2 bytes per element, for the SPM budget and all traffic | §6 (FP16 SIMD unit; tensor precision is our inference) | `scale_sim_runner.BYTES_PER_ELEMENT` |
| Timing model | Per layer `max(compute, transfer)`; no stalls, no bank conflicts, no DRAM latency. §3.3 calls the paper's tool a "cycle-accurate simulator", but §6 describes this bandwidth-centric model | §6 | `run_onsram_scale_sim()` |
| Traffic | Each real tensor once: input unless in the SPM, output unless pinned, weights always | §6: "ideal tiling … each data element is fetched once" | `scale_sim_runner._simulate_layer()` |
| Output rule | Only pinned outputs stay on-chip; the rest are written back | §3.1, §3.2 | `scale_sim_runner._ofmap_pinned()` |
| Non-conv ops | Pure data movement (compute ≈ 0); DENSE by SCALE-Sim's fold formula | Our model; §7.1.3 treats these as activation-bound | `scale_sim_runner._nonconv_layer_stats()` |
| Graphs | Unfused exports (BatchNorm/BiasAdd/ReLU as separate nodes) for comparison against Fig. 7 / Table 1 | §4, §7.1.3, §7.1.6 (our reading) | `spm_common/unfuse_model.py` |
| Compute source | SCALE-Sim's stall-free cycles for conv layers; depthwise mapped channels-across-columns | Not the paper's (calibrated, unpublished model); open choice | `onsram_helpers/topology.py`, `_simulate_layer()` |
| Physical check | Real SPM addresses + `SpmAllocator` replay; goes beyond the paper, which has no addresses | — | `run_onsram.check_physical_validity()` |
| Paper targets | Fig. 7 OnSRAM-Static bars (mostly read off the figure) and Table 1 ∞ SPM | Fig. 7, Table 1 | `run_paper_reproduction.py` |

Still open for matching the paper: the compute model (SCALE-Sim vs ideal;
mainly MobileNetV1), how the paper's ResNeXt graph was built, VGG16
(memory), Inception-v4/SSD300/PTB-LSTM/Multi-Head Attention (not
sourced or not modelled), OnSRAM-Eager, and the energy results.

### Change: one testbench for COSMA, OnSRAM and SMM

| Area | Today | Needed for a common testbench |
|---|---|---|
| Evaluator | Each paper's runner costs its own plan (`cosma/helpers/baseline.py`, `onsram/onsram_helpers/scale_sim_runner.py`) | One shared evaluator (e.g. in `spm_common/`) that takes any plan (resident, hand-off, spill and retrieve actions plus the schedule) and applies the same cost rules. The per-paper runners stay as they are for the paper reproductions |
| Baseline | Each runner computes its own "no SPM management" pass | One shared baseline per model and config |
| Bytes per element | COSMA: float32 SPM budget, 1 byte per element of traffic. OnSRAM: FP16 for both | One data-width parameter for all three |
| Traffic model | COSMA: SCALE-Sim's own DRAM counts (re-reads, partial sums). OnSRAM: fetch-once | One model for all three. Fetch-once hides SMM's benefit (re-fetches inside a layer); SCALE-Sim's counts thrash when buffers are small. With SMM in, this needs an analytic model of tiling inside a layer with finite buffers |
| Compute model | OnSRAM: SCALE-Sim cycles minus stalls. COSMA: SCALE-Sim cycles including stalls (the 09-25 change was OnSRAM-only) | One source for all three (SCALE-Sim stall-free, or ideal), cached per model, array size and dataflow so the rest of a sweep needs no re-simulation |
| Hardware config | Chosen per runner (`run_onsram.py` defaults to `configs/scale.cfg`, `run_paper_reproduction.py` to `configs/scale_onsram.cfg`) | One config per experiment point, shared by all three |
| Non-conv ops | Same rules, duplicated in both runners | Move into the shared evaluator |
| Weights in the SPM budget | No implementation charges weights against the budget; the free-room split is only logged | One rule for all three. OnSRAM's paper streams weights through the same SPM; COSMA's "+parameter" mode isn't built |
| SMM | Port lives on the `sim-opt` branch | Merge into the same structure (`spm_common/` plus its own folder) |
| Common models | No model runs on all three yet | Pick a set all three can run (e.g. MobileNetV1, ResNet-50) |
| Depthwise/grouped mapping | Same in both topology builders | Keep (already shared) |
| Physical check | `SpmAllocator` shared | Keep (already shared) |
| Output-on-chip rule | COSMA creates every output in the SPM (its Eq. 7); OnSRAM only pinned ones | Keep per paper: it's part of each plan, and the shared evaluator just follows it |
| Scheduling freedom | COSMA's ILP and OnSRAM's BFS-DFS reorder layers; SMM uses a fixed order | Keep per paper, but report it next to every result, since reordering is a bigger problem than placement alone |
| Metrics | Each runner prints its own | Same set for all: DRAM bytes (ifmap/filter/ofmap), cycles, speedup vs the shared baseline, share of compute-bound layers, solve time |

## Getting closer to the paper's numbers

The paper's compute model (DeepTools [93], calibrated to its chip) and its exact graphs are not published, so an exact match isn't possible. These steps close part of the gap, most defensible first:

| Step | Why it's justified | Expected effect |
|---|---|---|
| **Ideal compute option** (multiply-adds ÷ 1,521 units), reported next to the SCALE-Sim column | §6: the baseline "assumes ideal tiling, prefetching and dataflow" | MobileNetV1's ceiling moves from 2.86× into a bound of roughly 3.4–6.8× (estimated from totals, not yet measured), which contains the paper's 5.17×. Must check that AlexNet, ResNet-50 and Inception-v3, which match today, don't overshoot |
| **Check Fig. 11 for how the paper's graphs are built** (mean/variance/split nodes) | §7.1.3 lists "mean, variance, reshape, concat" among the pinned layer types | If BatchNorm appears as separate statistics nodes, add that to `unfuse_model.py`. Affects MobileNetV1 and ResNeXt most |
| **ResNeXt built as split + 32 convs + concat** | TensorFlow had no grouped conv at the time; the paper runs inside TensorFlow's graph compiler (§6) and cites only the ResNeXt paper [101], with no depth or construction | Could explain ResNeXt's gap. Report as a sensitivity result, not a tuned model |
| **SqueezeNet v1.0 instead of v1.1** | The paper cites the original SqueezeNet paper [45] (Iandola et al. 2016), which defines v1.0; v1.1 only appeared later in the authors' repository. The paper doesn't state the version | v1.0 has more compute, so it's less memory-bound; should pull our +31% down |

## Explaining the remaining differences

Split the gap into two independent parts:

- **Algorithm efficiency = our speedup ÷ our own ∞-SPM ceiling.** This checks the port. It is **91%** (geomean, unfused). The paper states OnSRAM-Static's improvement "is already 90% of that achievable by the ideal Infinite SPM" (§7.1.1).
- **Platform and graph = our ceiling ÷ the paper's ceiling (Table 1).** This does not depend on any SPM policy; it measures how memory-bound the same network is on our simulator versus theirs.

The speedup gap follows the ceiling gap, model by model (unfused, 2 MB, 2026-09-28 run):

| Model | Our ceiling ÷ paper ceiling (Table 1) | Our speedup ÷ paper speedup (Fig. 7) |
|---|---:|---:|
| AlexNet | 1.00 | 1.02 |
| ResNet-50 | 1.01 | 1.08 |
| Inception-v3 | 0.95 | 1.10 |
| GoogLeNet | 0.91 | 0.88 |
| SqueezeNet | 1.28 | 1.31 |
| MobileNetV1 | 0.55 | 0.58 |
| ResNeXt | 0.50 | 0.45 |

Argument: our OnSRAM port recovers the same share of the available benefit as the paper's; the remaining per-model differences are explained by how memory-bound each network is on our platform, which is measured independently of OnSRAM.

What makes this hold up:

- **Every modelling choice has a paper quote** (fetch-once, per-node `max(compute, transfer)`, only pinned outputs stay on-chip). Inferences are labelled as inferences (e.g. FP16 tensors).
- **The ceiling is a valid bound:** per-layer compute cycles are identical in the baseline and OnSRAM passes (SqueezeNet1.1 unfused: 0 of 92 layers differ).
- **The plans are physically valid:** 0 `SpmAllocator` violations in all 16 runs, and every Overwrite Optimization hand-off read served from the SPM.
- **What can't be known is disclosed:** the paper's compute model is calibrated to its chip and unpublished, its graph construction isn't stated, and five of the seven Fig. 7 values were read off the chart (±0.02).
- **Sensitivity is reported, not tuned:** fused and unfused, SCALE-Sim and ideal compute, and every ResNeXt variant tried. The conclusion ("~90% of the ceiling; the gap follows the ceiling") should hold in each.

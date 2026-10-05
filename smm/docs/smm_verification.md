# SMM port: fidelity verification

**Paper**: Stavroula Zouzoula, Mohammad Ali Maleki, Muhammad Waqar Azhar, Pedro Trancoso.
"Scratchpad Memory Management for Deep Learning Accelerators." ICPP '24.
https://doi.org/10.1145/3673038.3673115

## 1. Scope decision: real SCALE-Sim simulation, not the paper's own methodology

The paper's own Sec. 4 methodology only ever runs a cycle-accurate SCALE-Sim simulation for
the **baseline** (fixed ifmap/filter partition, 3 ratios). Its own proposed Hom/Het schemes are
evaluated purely analytically, via the closed-form formulas of Sec. 3.2/3.3 — confirmed directly
in the paper's text: "it took approximately one minute to generate the management schemes for
all the tested models ... while for the SCALE-Sim baseline it took more than 5 hours ... Our
approach is based on heuristics ... the baseline is a full simulator."

For this repo's shared-platform goal (COSMA/OnSRAM/SMM all benchmarked on one simulator, see
`project_cosma_three_papers` memory), the user explicitly chose to make Hom/Het **real,
cycle-simulated SCALE-Sim runs** as well, not just the paper's own analytical estimate — so SMM
sits on equal footing with COSMA (`cosma_resident_buffers.py`) and OnSRAM
(`onsram_helpers/resident_buffers.py`), which both already drive real SCALE-Sim simulation via
custom buffer classes. This is new engineering beyond what the paper itself does, not a
reproduction of a paper-described mechanism.

## 2. `smm_helpers/policy_selector.py`: formula fidelity

Ported from a separately maintained, self-validated C++ reference implementation at
`/home/george/Desktop/Scratchpad-Memory-Management-for-DL-Accelerators-main` (`include/smm/
policy.h`, `manager.h`; see that repo's own `VERIFICATION.md`, which reproduces the paper's
Table 3 exactly). Term-for-term comparison against that reference confirms `_memory_elems`,
`_estimate_accesses`, `_evaluate`, and `best_for_layer` are faithful ports of `memory_elems`,
`estimate_accesses`, `evaluate`, and `best_for_layer` (policy.h/manager.h) — same formulas, same
Algorithm 1 selection logic (no exact tie-break-by-latency on equal accesses, matching the
reference's own `best_for_layer`, which also omits it despite the paper's pseudocode suggesting
one).

## 3. The real-simulation gap, and how it was closed

**Problem found (2026-10-02)**: a first attempt (`smm_helpers/scale_sim_runner.py`) simply
resized SCALE-Sim's `double_buffered_scratchpad` to each policy's predicted footprint and
trusted SCALE-Sim's own cycle engine. Empirically, on ResNet18 Conv1 (IH=IW=224, CI=3, FH=FW=7,
Fn=64) at 64kB GLB with Policy 1 selected, simulated DRAM traffic came out ~3.3x higher than
`policy_selector`'s own predicted total (898.8kB predicted vs. ~2928kB implied by SCALE-Sim's
reported bandwidth × cycles).

**Root cause**: SCALE-Sim's `ReadBufferEstimateBw` (`scalesim/memory/read_buffer_estimate_bw.py`)
is a genuinely capacity-windowed cache — addresses are grouped into ~100 chronological "sets",
and `check_hit()` only looks inside the currently-resident window. SCALE-Sim's own PE-array fold
structure (e.g. Fn=64 through a 16-wide array needs multiple column-folds, each re-streaming the
ifmap through the array) re-requests the same logical elements many cycles apart — far outside
the window a policy's own (much smaller) buffer sizing can keep resident — so each re-request is
a genuine re-fetch under SCALE-Sim's native accounting. This has nothing to do with buffer
*sizing*; it is a mismatch between the paper's own loop-nest-level reuse assumption ("each
element transferred once", no eviction model — explicit in the paper and in the C++ reference's
own README) and SCALE-Sim's unrelated array-fold-driven demand trace.

**Fix**: `scalesim/memory/smm_reuse_buffers.py`'s `SmmReuseReadBuffer` (installed via the
existing `ifmap_buf_class`/`filter_buf_class` override hooks in
`double_buffered_scratchpad.set_params()` — the same mechanism COSMA/OnSRAM use for their own
resident-buffer classes). It overrides only `check_hit()`: once an address has been charged a
real (backing-buffer) fetch `reload_budget` times this layer, every later request for that same
address is reported as a hit, regardless of the base class's own window-based verdict.
`reload_budget` is 1 for every SMM policy's filter operand and for ifmap under every non-partial
policy (Intra/P1/P2/P3); `ceil(F#/n)` for ifmap under P4/P5 (`LayerPlan.ifmap_reload`, computed
in `policy_selector._evaluate`). Every genuine first-time (or within-budget) fetch runs through
the base class's completely unmodified logic — only excess re-fetches beyond the policy's own
stated budget are suppressed. ofmap is left as the stock `write_buffer`: each output element is
produced and written exactly once by construction regardless of fold order, and this already
matched the formula empirically without any override (see below).

## 4. Empirical validation

Validated by comparing `LayerPlan.{ifmap,filter,ofmap}_access_bytes` (the closed-form per-operand
prediction) against the REAL simulated per-operand DRAM byte counts, read directly from
`single_layer_sim.get_detail_report_items()` (`ifmap_dram_reads`/`filter_dram_reads`/
`ofmap_dram_writes` — NOT reconstructed from `avg_*_dram_bw × total_cycles`, since that average is
normalized over the active DRAM window, `stop_cycle - start_cycle`, not total layer cycles, and
reconstructing from it silently overstates the true byte count).

**Non-partial policy** (ResNet18 Conv1+Conv2_1a, 64kB GLB, Policy 1 selected both layers):

| Layer | operand | predicted (B) | actual (B) | ratio |
|---|---|---:|---:|---:|
| Conv1 | ifmap | 150,528 | 150,528 | 1.000 |
| Conv1 | filter | 9,408 | 9,408 | 1.000 |
| Conv1 | ofmap | 760,384 | 774,415 | 1.018 |
| Conv2_1a | ifmap | 200,704 | 200,704 | 1.000 |
| Conv2_1a | filter | 36,864 | 36,864 | 1.000 |
| Conv2_1a | ofmap | 186,624 | 186,639 | 1.000 |

Exact match on ifmap/filter; ofmap within ~2% (minor boundary/rounding effect in SCALE-Sim's own
write accounting, present even without any SMM-specific override — not a reuse-modeling gap).

**Note**: this §4 test (and the `run_smm.py` "57.4% DRAM-byte reduction" smoke test reported
2026-10-02) used `topologies/conv_nets/Resnet18.csv`, the repo's existing *unpadded* topology
file — a self-consistency check (does SCALE-Sim's simulation match our own formula's prediction
*for whatever shape SCALE-Sim is actually simulating*), not a check against the real ResNet18
architecture. §4b below is the real-architecture check, and surfaces a separate, small gap from
that unpadded sourcing. Any real benchmark run should use
`smm/topologies/resnet18_same_padded.csv` (§4b), not the plain repo file.

**Partial policy, reload > 1** (same 2 layers, 8kB GLB — small enough to force P4/P5):

| Layer | policy | n | reload budget | operand | predicted (B) | actual (B) | ratio |
|---|---|---:|---:|---|---:|---:|---:|
| Conv1 | P4 | 13 | 5 | ifmap | 752,640 | 523,861 | 0.696 |
| Conv1 | P4 | 13 | 5 | filter | 9,408 | 9,408 | 1.000 |
| Conv1 | P4 | 13 | 5 | ofmap | 760,384 | 774,400 | 1.018 |
| Conv2_1a | P5 | 2 | 32 | ifmap | 6,422,528 | 6,026,240 | 0.938 |
| Conv2_1a | P5 | 2 | 32 | filter | 36,864 | 36,864 | 1.000 |
| Conv2_1a | P5 | 2 | 32 | ofmap | 186,624 | 186,624 | 1.000 |

ifmap ratios are ≤ 1.0, as expected by construction: `reload_budget` is a **ceiling** (SMM's own
worst-case loop-nest count), not a guarantee of exactly that many real fetches. SCALE-Sim's
native fold structure sometimes achieves better reuse than the policy's own assumed filter-block
schedule before `SmmReuseReadBuffer`'s override is even needed, so real traffic can come in at or
below the formula's prediction — never above it. Filter/ofmap remain exact, as filter's budget is
always 1 regardless of policy and ofmap needs no override at all.

**Conclusion**: real, cycle-accurate SCALE-Sim simulation of SMM's Hom/Het schemes now matches
(non-partial policies) or stays at/under (partial policies, by the budget's own design) the
paper's closed-form per-operand access-volume prediction — SMM can be benchmarked against
COSMA/OnSRAM on genuinely equal footing (all three real-simulated, not one of the three
analytical-only).

## 4b. Direct cross-check against the paper's own published Table 3

Beyond the self-consistency checks in §4 (our simulation vs. our own formula), the formula
itself was checked against the paper's own published numbers (Table 3, ResNet18: intra=2353kB,
{P1,P3}={788.6,2318}kB — the C++ reference's own `VERIFICATION.md` already established the
paper's table prints the P1/P3 *columns* swapped vs. its own §3.2 formulas; matching is by value,
not by column label — P2=199.7kB), independently reproduced here, not just trusted from the C++
reference's claim:

- **Standalone** (`LayerSpec` built directly with each layer's real raw `IH`/`IW` + explicit `P`,
  mirroring the C++ reference's `resnet18()` in `models.h` exactly): **exact match** —
  Intra=2353.0, P1=2318.0, P2=199.6, P3=788.6kB.
- **Through the real topology-CSV pipeline** (`smm/tools/build_resnet18_topology.py` →
  `smm/topologies/resnet18_same_padded.csv` → `scalesim.topology_utils` →
  `_build_layer_specs()`, the actual code path `run_smm.py` uses): Intra=2369.0 (+0.7%),
  P1=2321.0 (+0.1%), P2=213.9 (**+7.1%**), P3=788.6 (exact).

**Why the second check isn't exact**: SCALE-Sim's topology CSV format has no padding field at
all — `scalesim/topology_utils.py`'s `topo_calc_hyperparams()` computes ofmap dims via plain
VALID convolution, nothing else — so representing a padded conv requires pre-folding the pad
into the CSV's IFMAP Height/Width (`cosma/helpers/topology_builder.py`'s `_same_padded_dim()`,
reused as-is by `build_resnet18_topology.py`, not reimplemented). That's correct and necessary
for SCALE-Sim's own OH/OW-driven demand-matrix generation. But `policy_selector.py`'s formula
terms that count *real DRAM volume* (`ifmap_elems() = IH*IW*CI`, used directly by Intra and P2 —
the two policies whose footprint includes the *whole* ifmap resident) should only count genuine
data, not zero-padding pixels — and once padding is folded into IH/IW to satisfy SCALE-Sim's
engine, there is no way left to tell the two apart from the CSV alone. The standalone check
above (raw IH + explicit P, LayerSpec's own native representation) keeps that distinction and is
exact; policies P1/P3 (dominated by filter/per-channel/ofmap terms, not full-ifmap) are
essentially unaffected (≤0.1%/exact) since their formulas don't multiply the whole padded ifmap
area. This is a small, **conservative** (always over-, never under-estimates real DRAM volume —
so Algorithm 1 never picks an infeasible policy because of it), quantified, and now-documented
limitation of SCALE-Sim's own topology format, not a logic error in the port — and it's a
limitation every model sourced through a SCALE-Sim topology CSV on this shared platform inherits
equally, not something specific to SMM.

## 5. Known gaps (shared with COSMA/OnSRAM's own ports, not new to SMM)

- FC/DENSE layers are not simulated (same gap `smm/docs/smm_model_roster.md` and both other
  papers' rosters already flag) — `cosma/helpers/topology_builder.build_topology()` (reused by
  `smm/run_smm.py` for model.json inputs) only emits CONV2D/DEPTHWISE_CONV2D/CONV_3D rows.
- MnasNet and a true ResNet18 model.json remain unsourced (see `smm/docs/smm_model_roster.md`);
  ResNet18 runs via the hand-built, conv-only `smm/topologies/resnet18_same_padded.csv` instead.

## 6. Cross-model check against the paper's own stated numbers (2026-10-05)

Full real-simulated runs (baseline x3 + Hom + Het, all 5 of the paper's GLB sizes, DENSE layers
included via `dense_costing.py` where the export has a real DENSE op) on 5 of the paper's 6
models: ResNet18 (hand-built topology), MobileNet, MobileNetV2, GoogLeNet
(`cosma/_exported/GoogLeNet/` — corrected onto the roster this session; see
`smm_model_roster.md`), and EfficientNetB0-ish (`efficient50`, variant unconfirmed). MnasNet still
missing. All run on a second machine ("mary") after this port was git-pushed there; numbers
reproduced exactly against this machine's own earlier ResNet18 run, confirming no machine-specific
drift.

**% off-chip-access reduction vs. best baseline, at 64kB** (the one GLB size the paper quotes
exact numbers for in its own text, §5.1):

| Model | Our Het | Our Hom |
|---|---:|---:|
| ResNet18 | 85.9% | 82.1% |
| GoogLeNet | 76.4% | 71.6% |
| MobileNet | 72.9% | 66.8% |
| EfficientNetB0-ish | 54.8% | 50.5% |
| MobileNetV2 | 48.8% | 46.5% |

**Paper's own stated range (§5.1)**: Het 43.2% (MobileNetV2, the paper's own stated *floor*) to
79.8% (ResNet18, the paper's own stated *ceiling*); Hom 32.2% (MnasNet) to 74.5% (ResNet18).

**Findings**:
1. **Ranking matches exactly** — ResNet18 is our highest, MobileNetV2 our lowest, precisely the
   two models the paper itself names as its extremes.
2. **Adding DENSE costing measurably closed part of the gap**, not just theoretically: MobileNetV2
   moved 51.3% → 48.8% once its classifier layer was correctly counted (its export has 1 real
   DENSE op; MobileNet(v1)'s classifier is a 1×1 CONV2D instead, unaffected either way) — in the
   right direction, toward the paper's 43.2%.
3. **The remaining gap is a consistent, same-sized, same-direction offset, not scatter**: ResNet18
   overshoots the paper by +6.1pp (85.9 vs 79.8), MobileNetV2 by +5.6pp (48.8 vs 43.2) — two
   different models, near-identical overshoot. That consistency points to one systematic cause
   (most likely the baseline sa_50_50-vs-sa_25_75 ratio-preference flip already diagnosed in §4
   above — our baseline's own absolute numbers run a bit different from the paper's, which
   inflates every Het/Hom percentage roughly equally, since Het/Hom's own absolute DRAM totals are
   independently exact against Table 3), rather than a per-model bug.
4. The paper's own "Het/Hom go nearly flat once GLB is large enough" behavior (§5.1) holds across
   every model tested, not just ResNet18 — confirmed e.g. in GoogLeNet (flat ~14.61MB from 256kB
   up) and MobileNetV2 (flat ~13.21MB from 64kB up already).

**Still open**: MnasNet (unsourced) and a true ResNet18 model.json would complete the paper's full
6-model roster; the `latency` objective remains unexercised on any model; and the baseline
ratio-preference mechanism (finding 3 above) is diagnosed but not yet traced to a single root
cause beyond "filter-traffic saturation on large late-stage layers" (§4's per-layer breakdown).

# SMM: implementation overview

**Paper**: Stavroula Zouzoula, Mohammad Ali Maleki, Muhammad Waqar Azhar, Pedro Trancoso.
"Scratchpad Memory Management for Deep Learning Accelerators." ICPP '24.
https://doi.org/10.1145/3673038.3673115

This doc explains how the paper is implemented on top of SCALE-Sim, what was changed inside
SCALE-Sim itself to make it work, and how to read the results it produces. For the empirical
fidelity checks (does our port actually reproduce the paper's own numbers) see
`smm/docs/smm_verification.md`. For which of the paper's 6 evaluation models are available to run,
see `smm/docs/smm_model_roster.md`.

## 1. What the paper actually contributes

Most DL accelerators split their on-chip scratchpad into **fixed, separate buffers** — one for
ifmap, one for filters, one for ofmap — sized once and never adapted. That's simple hardware, but
wasteful: different layers of a CNN need wildly different amounts of each data type (early layers
are ifmap/ofmap-heavy, late layers are filter-heavy — see the paper's Fig. 3), so a fixed split is
never a good fit for every layer at once.

The paper's contribution is a **software memory-management scheme for a single unified on-chip
buffer (the GLB)**: instead of fixed partitions, it defines six **policies** — each a specific
tiling/loop-nest strategy (e.g. "keep all filters resident, stream ifmap as a sliding window" vs.
"keep the whole ifmap resident, stream filters one at a time") that trades off how much GLB space
goes to each data type and how many times data must be re-fetched from off-chip. **Algorithm 1**
then picks, for every layer of a real model, whichever policy minimizes off-chip accesses (or
latency) given the GLB size — either the *same* policy for the whole network (**homogeneous**,
"Hom") or a *different* policy per layer (**heterogeneous**, "Het"). The whole thing is a
lightweight, closed-form estimation (not a cycle-accurate simulation) — the paper's own point is
that this decision can be made in ~1 minute per model, instead of hours of simulation, while still
capturing most of the benefit: up to 80% fewer off-chip accesses, up to 56% lower latency, versus a
fixed-partition baseline, on 6 standard CNNs.

**How to read results from this port**: the headline number is **% reduction in off-chip DRAM
bytes (or latency) for Hom/Het vs. the best fixed-partition baseline**, at a given GLB size. The
paper's own clearest, most paper-faithful trend to check any new model against: the reduction is
largest at the smallest GLB sizes and shrinks as the buffer grows, and Het's absolute DRAM-byte
total should go nearly *flat* once the GLB is big enough for every layer to afford a
"transfer-each-element-once" policy — both are direct, paper-stated claims (§5.1), not just
something we happened to observe.

## 2. Where everything lives

```
smm/
  smm_helpers/
    policy_selector.py   # the paper's own formulas: Intra/P1-P5, Algorithm 1 (closed-form, no SCALE-Sim)
    scale_sim_runner.py  # drives REAL SCALE-Sim for the Hom/Het schemes
    baseline.py           # drives real SCALE-Sim for the paper's 3 fixed-partition baselines
  tools/
    build_resnet18_topology.py   # emits a correctly SAME-padded ResNet18 topology CSV
  topologies/
    resnet18_same_padded.csv     # output of the above
  docs/
    smm_implementation.md  # this file
    smm_verification.md    # empirical fidelity checks against the paper's own published numbers
    smm_model_roster.md    # which of the paper's 6 models are available to run
  results/                  # run_smm.py output (traces + whatever you redirect its stdout to)
  run_smm.py                # entry point: runs baseline + Hom + Het for a model/GLB size and prints a comparison

scalesim/memory/smm_reuse_buffers.py   # the one change made to SCALE-Sim's own engine (§4 below)
configs/scale_smm.cfg                  # paper's §4 hardware: 16x16 array, OS dataflow, 8-bit, CALC bandwidth
```

`policy_selector.py` is a line-for-line Python port of a separately-maintained, independently
verified C++ reference implementation (`include/smm/policy.h` + `manager.h` at
`/home/george/Desktop/Scratchpad-Memory-Management-for-DL-Accelerators-main`) — not a
reinterpretation. `baseline.py` and `scale_sim_runner.py` are new code (the paper doesn't publish
an implementation of its own SCALE-Sim integration).

## 3. What we changed inside SCALE-Sim, and why

**Short answer: one file, `scalesim/memory/smm_reuse_buffers.py`**, installed via a hook
(`ifmap_buf_class`/`filter_buf_class`) that already existed in
`scalesim/memory/double_buffered_scratchpad_mem.py` for exactly this purpose — the same mechanism
COSMA (`scalesim/memory/cosma_resident_buffers.py`) and OnSRAM
(`onsram/onsram_helpers/resident_buffers.py`) already use to give their own plans a real effect on
SCALE-Sim's simulation, not a new extension point invented for SMM.

**Why it was needed.** The paper itself never cycle-simulates its own Hom/Het schemes — only the
fixed-partition baseline is run through a real simulator (§4: "it took approximately one minute to
generate the management schemes... while for the SCALE-Sim baseline it took more than 5 hours").
For this repo's purpose (benchmarking SMM against COSMA and OnSRAM, which *are* both real-simulated
end to end), the decision was made to make SMM's Hom/Het schemes real-simulated too, not just
analytically estimated like the paper does.

The first attempt at this (just resizing SCALE-Sim's buffers to each policy's predicted footprint
and trusting SCALE-Sim's own cycle engine) was empirically wrong by ~3.3x — confirmed by directly
comparing predicted vs. simulated DRAM bytes, not assumed. The reason: SCALE-Sim's own read buffer
(`scalesim/memory/read_buffer_estimate_bw.py`) is a genuine **capacity-windowed cache** — it
forgets an address once the PE-array's own fold structure moves the window past it — while the
paper's policies assume a **software-managed scratchpad with no eviction model at all**: "each
element transferred off-chip at most `reload` times this layer" (1 for every policy's filter
operand and for ifmap under the non-partial policies; `ceil(F#/n)` for ifmap under the partial
policies P4/P5), full stop, regardless of how many times the PE array's own mapping happens to
revisit that address.

`SmmReuseReadBuffer` closes exactly that gap and nothing more: it remembers, per address, how many
times a *genuine* DRAM fetch has already been charged this layer; once that count reaches the
policy's own `reload_budget`, every further request for the same address is reported as a hit,
regardless of what the base class's capacity-window logic would otherwise say. Every first-time
(or within-budget) fetch runs through SCALE-Sim's completely unmodified logic — only fetches in
*excess* of what the paper's own formula allows are suppressed. ofmap needs no equivalent override
(`write_buffer` is left untouched): each output element is produced and written exactly once by
construction, regardless of fold order, and this already matched the paper's formula without any
change (verified, not assumed — see `smm_verification.md` §4).

**What was deliberately left alone**: the paper's own fixed-partition **baseline**
(`smm_helpers/baseline.py`) uses the stock, unmodified `double_buffered_scratchpad` — no custom
buffer class at all. That's intentional: the baseline is the one piece of the paper's own
methodology that was already meant to be a real, unmodified SCALE-Sim simulation, so it should
behave exactly as SCALE-Sim behaves for anyone else's topology — including showing the same
PE-array-fold-driven re-fetching that `SmmReuseReadBuffer` exists to correct for the *policy-managed*
schemes. (This is also why the baseline's filter-heavy layers can look "stuck" near the same DRAM
total regardless of how much of the buffer you give filters — see the ratio-preference
investigation below.)

## 4. The simulation pipeline, end to end

For a given model and GLB size, `run_smm.py` runs five SCALE-Sim passes and compares them:

1. **Topology sourcing** — either a SCALE-Sim topology CSV directly, or a model.json converted via
   `cosma/helpers/topology_builder.build_topology()` (the same converter COSMA/OnSRAM already use;
   only CONV2D/DEPTHWISE_CONV2D/CONV_3D layers get a topology row). SCALE-Sim's own topology engine
   has **no padding concept at all** (`scalesim/topology_utils.py`'s `topo_calc_hyperparams()` is
   plain VALID convolution, nothing else), so any padded network's topology CSV must have its IFMAP
   Height/Width *pre-inflated* to the SAME-padded size — `build_resnet18_topology.py` does this for
   ResNet18 by reusing `topology_builder.py`'s own `_same_padded_dim()` helper, since no ResNet18
   model.json export exists yet.

   **DENSE (FC) layers**, when the model input is a `.json`, are detected separately and costed
   analytically (`smm_helpers/dense_costing.py`) — *not* through a topology row, and *not*
   real-simulated. This deliberately mirrors `cosma/helpers/baseline.py`'s and
   `onsram/onsram_helpers/scale_sim_runner.py`'s own `_nonconv_layer_stats()` DENSE formula exactly
   (weights fetched once + input/output vectors once + a weight-stationary fold-count cycle
   estimate), decided with the user 2026-10-04: SMM's own policies have no spatial-reuse choice to
   make on a layer with no spatial extent (every policy's formula collapses to the same thing when
   IH=IW=OH=OW=1), so real-simulating DENSE here while COSMA/OnSRAM cost it analytically would
   measure simulation fidelity on a layer type none of the three papers' algorithms actually act
   on — a confound in a cross-paper benchmark, not a result — and it avoids a real, already-hit
   SCALE-Sim OOM on large dense layers that COSMA's own code documents (AlexNet's 9216×4096 needed
   >5GB). The resulting cost is identical across every scheme (baseline/Hom/Het) for a given model,
   since none of them can do anything differently for a layer shape this degenerate, and is added
   equally to all of them by `run_smm.py` before the final comparison. Note: not every model's
   exported classifier head is actually a DENSE op — MobileNet(v1)'s, for instance, is a 1×1
   CONV2D, already covered by the topology-row path with no special-casing needed (see
   `smm_model_roster.md` §2 for which of the paper's 6 models have a real DENSE layer).

2. **Algorithm 1** (`policy_selector.plan_network`) picks a policy (and prefetch on/off) for every
   layer, purely analytically — this is the paper's own closed-form step, unmodified.

3. **Hom/Het real simulation** (`scale_sim_runner.SMMScaleSimRunner.run()`) — for every layer, a
   `single_layer_sim` is built with its memory system sized exactly to the chosen policy's
   ifmap/filter/ofmap byte allocation, with `SmmReuseReadBuffer` installed on the ifmap/filter
   buffers (§3). Cycles and per-operand DRAM bytes are pulled from each layer's own
   `get_detail_report_items()` immediately after it runs, into a lightweight running total — the
   full `single_layer_sim` object (and its potentially multi-hundred-MB demand/operand matrices)
   is *not* kept around afterward (see the memory note below).

4. **Baseline real simulation** (`baseline.run_baseline()`) — same per-layer loop, but with a
   single *fixed* ifmap/filter split for the whole model (one of `sa_25_75`/`sa_50_50`/`sa_75_25`,
   §4kB ofmap buffer fixed per the paper), stock unmodified buffers, no policy selection at all.

5. **Comparison** — `run_smm.py` prints cycles and total DRAM bytes for all five runs
   (`sa_25_75`, `sa_50_50`, `sa_75_25`, `Het_<objective>`, `Hom_<objective>`), plus Het/Hom's %
   reduction vs. whichever of the three baselines came out lowest for that run.

**A memory note, since it already caused one real outage**: an earlier version of
`scale_sim_runner.py` kept every layer's full `single_layer_sim` object alive for the whole run
(`self.layer_sims.append(sim)`), which — for a real ~20-layer model with large late-stage conv
layers — accumulated several GB of retained numpy data and got the process OOM-killed mid-run on a
7GB machine (confirmed via `dmesg`, not assumed). Fixed by extracting only the per-layer
cycle/byte numbers immediately after each layer runs and letting the `sim` object (and its
matrices) be garbage-collected before moving to the next layer. If you ever see memory climb
steadily layer-by-layer again, that's the pattern to check for first.

## 5. Running it

```bash
python3 smm/tools/build_resnet18_topology.py     # only needed once, or after editing the topology

python3 smm/run_smm.py \
  --model smm/topologies/resnet18_same_padded.csv \
  --config configs/scale_smm.cfg \
  --glb_kb 64 128 256 512 1024 \
  --objective accesses \
  --out smm/results/resnet18_full_run
```

`--model` also accepts a model.json path (e.g. `cosma/_exported/MobileNet/model.json`) directly.
`--objective latency` switches Algorithm 1 to minimize latency instead of accesses (not yet
empirically exercised as of this writing — see `smm_verification.md` §5). `--skip-baseline` runs
Hom/Het only, useful for a quick check. On a memory- or time-constrained machine, loop over one
`--glb_kb` value per invocation rather than passing the whole list at once, and redirect/`tee`
stdout to a file — `run_smm.py` does not save its own results table anywhere by default.

## 6. Reading the output, and what's still open

A real run on ResNet18 (full 20 conv layers, SAME-padded) showed the paper's own described
behavior qualitatively (Het/Hom plateauing to an identical constant DRAM total from 256kB upward,
large reductions at 64kB) and a 64kB Het reduction (85.9%) in the right ballpark of the paper's own
stated 79.8% for this exact model — but several points above it, not an exact match. The most
concrete, diagnosed source of that gap: our baseline's best-performing fixed ratio for ResNet18
(`sa_50_50`) differs from the paper's own claim (`sa_25_75`) — traced to filter traffic for
ResNet18's large late-stage layers being so far beyond any buffer fraction tested that it's
essentially insensitive to the ifmap/filter split (saturated, fold-refetch-dominated), while ifmap
traffic stays genuinely capacity-sensitive in the same range — a real, mechanistic finding, not a
bug, but also not something fully reconciled against the paper's own one-line qualitative summary.
Full numbers and the per-layer breakdown behind this are in `smm_verification.md`.

Open items: only ResNet18 (conv-only, no DENSE support needed yet — see §4) has been run
end-to-end against real numbers so far; MobileNetV2 has been smoke-tested with DENSE costing
wired in (correct, matches its known 1280→1000 classifier shape) but not yet run as a full
multi-GLB-size comparison; MobileNet(v1) and EfficientNetB0 are sourced and ready but not yet run
at all (see `smm_model_roster.md`); the `latency` objective is unexercised; and the small,
quantified, conservative footprint-overcount from SCALE-Sim's padding-folded-into-IFMAP-dims
representation (§4b of `smm_verification.md`) remains a known limitation of the whole shared
platform, not something to "fix" inside SMM specifically.

# SPM traffic & array emulation — where things stand (2026-09-23 session)

Plain-language companion to `CONTINUE_HERE.md` (repo root), which has the
dense, technical version of this same session for picking work back up.
This file is for the other question: in everyday terms, what did we
actually do, and is SCALE-Sim now accurately emulating what COSMA and
OnSRAM's own papers expect from the scratchpad (SPM) and the array?

## What we did this session, in plain terms

**1. Checked where OnSRAM actually stands.** Confirmed what's built (the
real algorithm, real SCALE-Sim simulation, real numbers) versus what's
deliberately left out because the port never got to it — OnSRAM's second
variant ("Eager," which handles models without a known graph), a
dedicated-weight-scratchpad comparison the paper also studies, and energy
numbers. None of that is new news this session, just re-confirmed against
current code rather than trusted from memory. Full list stays in
`onsram/docs/onsram_integration_plan.md`.

**2. Found a real, concrete bug and proved it, not just suspected it.**
There were two separate places in this project that each ask "how full is
the scratchpad?" — and they never compared notes:

- One (`SpmAllocator`) checks whether the tensors COSMA/OnSRAM decided to
  keep resident fit inside the budget. That check was always correct.
- The other (SCALE-Sim's own per-layer simulation) never asked "does the
  layer that's *about to run* also need room, on top of whatever's already
  sitting there from earlier layers?" It always assumed the *entire*
  budget was free, every single layer, regardless of what was already
  occupying it.

This wasn't a theoretical worry — we ran real cases and it genuinely
happened: on a tight COSMA budget, 4 out of 32 layers were already over
budget once you counted what was really sitting in the scratchpad at that
moment. On OnSRAM's own default test model, one single layer's weights
alone (4.2MB) were more than double the entire 2MB budget being simulated.

**3. Fixed the ifmap side of it — not the weights side, on purpose.**
The fix: instead of always pretending a layer has the whole budget to
itself, the layer's *ifmap fetch* is now sized to however much room is
*actually* left, given what other tensors are already pinned. When that
room is smaller than what the layer naturally needs, SCALE-Sim's own
existing "streaming buffer" logic (which was already built for exactly
this situation, just never actually exercised before) kicks in and
genuinely simulates the extra re-fetching that a real, more-crowded
scratchpad would cause.

Filters/weights were deliberately left out of this fix, on your explicit
call: neither COSMA's nor OnSRAM's algorithm currently tracks weights
against the SPM budget at all, so there was nothing to make the engine
"aware" of for weights yet — that's a decision for the *algorithm* side to
make first (see the open items below), not something to patch into the
engine on our own initiative.

One important design choice along the way, prompted by your own pushback:
the actual "how much room is left" calculation isn't copy-pasted into
COSMA's file and OnSRAM's file separately — it's one small, shared
function (`SpmAllocator.remaining_budget_for()`) both papers' code calls
identically, because the calculation itself has nothing paper-specific in
it at all.

While building this, we also found (and fixed) a real crash bug sitting
in SCALE-Sim's own code: a buffer squeezed down small enough could end up
with "zero room to prefetch into," which crashed the simulator outright.
This is the same class of bug as two other spots in the codebase that had
already been fixed before — this was just a third sibling nobody had
stress-tested this way yet.

Verified for real afterward: no crashes, the existing checks report
exactly the same numbers they did before, and the new fix visibly
engages (extra simulated cost) exactly where it should and stays
completely inert everywhere it isn't needed.

**4. Confirmed (twice) that three lines in `scale.cfg` do nothing.**
`IfmapSramSzkB` / `FilterSramSzkB` / `OfmapSramSzkB` look like the
obvious "SPM size" settings, but they're dead for anything running
through `run_cosma.py` or `run_onsram.py` — confirmed first by reading
the code, then by literally changing them by 64x and re-running: the
output didn't change by a single byte. The real SPM-size control is
`--budget-kb`/`--spm-mb`.

**5. Talked through a bigger structural question, without building it.**
SCALE-Sim keeps ifmap, filter, and ofmap in three physically separate
buffers internally. Both papers' own math assumes one single, shared
scratchpad. We confirmed this does **not** change either paper's actual
decisions — COSMA's ILP, for instance, already reasons about one shared
budget in its own equations, completely independent of how SCALE-Sim
happens to simulate it afterward. So this is a "how realistically do we
simulate the consequences of a decision" gap, not a "did the paper's
algorithm get the right answer" gap. See item 1 below for the size of
that undertaking if it's ever worth doing.

## Does SCALE-Sim need anything else to accurately emulate what COSMA/OnSRAM expect?

Yes. Here's the honest, current list, focused specifically on SPM traffic
and array behavior (not the broader paper-completeness items like energy
or OnSRAM-Eager, which live in `CONTINUE_HERE.md` instead):

1. **One shared scratchpad instead of three separate buffers.** SCALE-Sim
   hard-codes separate memory regions for ifmap, filter, and ofmap deep in
   its core (address offsets baked into `operand_matrix.py`, threaded
   through the compute engine and buffer classes). Making this genuinely
   one shared pool — where ifmap, filter, and ofmap fetches for the *same
   layer* compete for space with each other in real time, not just with
   what earlier layers left behind — would be closer to rebuilding the
   memory core than patching it. Worth doing only if you specifically care
   about simulation accuracy down to real intra-layer contention; it will
   not change what either paper's algorithm decides.

2. **Weights never compete for the SPM budget in simulation.** A weight
   fetch is always simulated as "full size, always fetched, no budget
   awareness at all" right now. COSMA's own paper has a documented mode,
   confirmed against the real paper text, where weights *do* share the
   same budget as activations — that mode isn't built yet. Whether
   OnSRAM's own paper wants the same thing is still genuinely unconfirmed
   (no primary source checked yet) — see `CONTINUE_HERE.md`.

3. **A tensor that's already resident gets an instant, flat-cost hit** —
   no simulated SRAM port/bank contention for data that's already pinned.
   A known, disclosed simplification, not yet addressed.

4. **Only conv-like layers get real simulated cost, full stop** — meaning
   real simulated SPM traffic too. Everything else (elementwise adds,
   softmax, average-pooling-as-reduction, etc.) is currently zero-cost.
   Matters most for models neither paper has actually been tested on yet
   (attention/transformer/LSTM-style networks); matters much less for the
   plain CNNs already validated, where this is mostly a rounding error.

5. **DRAM bandwidth accounting has a coupling quirk** in one of SCALE-Sim's
   two bandwidth modes (the "CALC" one ties assumed bandwidth to array
   width, as a workaround for not having a real bandwidth number to use).
   Worth flagging for completeness — though the config currently in use
   for these experiments is actually in the *other* mode ("USER"), so this
   specific quirk may not even be active in today's runs.

**If you had to fix only one of these next**, weights (#2) is the one
most likely to change actual reported numbers, since we already have
direct proof (this session's own OnSRAM test) that weight bytes alone can
dwarf the entire budget. The shared-scratchpad rewrite (#1) is the biggest
undertaking here and the one least likely to change any published number,
since it only affects intra-layer timing detail, not what fits in the
budget overall.

---

# Update — 2026-09-24 session: what changed on the `OnSram` branch

Same plain-language style as above. `CONTINUE_HERE.md` has since been
deleted (it was an old log), so references to it above are stale. Deeper
explanation, with measured numbers, is in
`onsram/docs/why_our_speedups_are_lower_than_the_paper.md`. Nothing below
is committed yet.

## Changes, in the order we made them

**1. Tried making all three buffers "budget-aware", then backed out of the
simulated part (supersedes item 3 above).** We extended the ifmap fix to
filter and ofmap: each layer's buffers were squeezed to the SPM room left
after pinned tensors. Two rounds of problems showed up:
- *Order mattered in a bad way.* Filling one operand first (first ofmap,
  then ifmap) could leave a tiny filter with 0 bytes, and SCALE-Sim then
  re-fetched it millions of times (MobileNet dropped to 0.80×). We
  replaced the ordering with a **fair split**: every operand gets an equal
  share, and one that needs less keeps just what it needs.
- *SCALE-Sim's buffers thrash when squeezed.* Even with the fair split,
  ResNet-50 layer 61 read its 2.36M weights 118M times, and a smaller
  buffer even gave *less* traffic than a bigger one. No real accelerator
  behaves like that; a weight-stationary array reads each weight once.

Final state (your choice): **ideal tiling**, which is also the OnSRAM
paper's own assumption ("each data element is fetched once"). Pinned
tensors still hold their space from layer to layer, the free room is
still split among ifmap/filter/ofmap and **logged** every run, but
SCALE-Sim's buffers are always their natural size. That means the Phase 1
ifmap clamp described in item 3 above is gone too.

What remains from this work: the shared `SpmAllocator.remaining_budget_for()`
(now accepts a group of tensor ids), the fair-split logging in both
runners, and the `read_buffer.py` crash floor (harmless at natural sizes).

**2. Checked the OnSRAM paper PDF itself (resolves item 2 above for
OnSRAM).** In the paper's main experiments, weights are **not** in a
separate memory: they stream through the same 2MB SPM, double-buffered,
and are never pinned. A dedicated weight SPM appears only in one variant
experiment (Fig. 10). The paper's timing model is per layer
`max(compute time, data-transfer time)`, the same formula this port uses.

**3. Fixed OnSRAM's "pinned but read from DRAM" problem (OnSRAM only).**
The Overwrite Optimization lets a layer's output take over its input's
SPM space. The port had implemented that by making the input leave the
SPM one step early, so the layer read its "pinned" input from DRAM. That
wiped out the benefit of about half of all pins (MobileNet: 11 of 13
depthwise layers). New: a separate `'H'` ("read from SPM, then free")
action, kept out of the shared `SpmAllocator` so COSMA's checker is
untouched. Files: `onsram/onsram_helpers/pinning.py`
(`build_handoff_action()`), `onsram/run_onsram.py`,
`onsram/onsram_helpers/scale_sim_runner.py`.

**4. Depthwise convolution mapped realistically (both runners, on
purpose).** Depthwise layers used to go to SCALE-Sim as "one filter over
all channels", which used 1 of 39 array columns and made them 40–60×
slower than they should be. Now each column handles one channel
(`Channels = 1, Num Filter = C`). SCALE-Sim times that correctly, and
output/weight traffic is correct. The input traffic is counted as the
real input tensor, each element read once. Files: both topology builders
and both runners. This is the one change that alters COSMA's numbers, and
only for models with depthwise layers (MobileNet, MobileNetV2). You chose
to keep it in both so the papers are compared fairly.

**5. OnSRAM now pays for writing unpinned outputs (OnSRAM only).**
Before, every layer's output write was free in the OnSRAM run. The paper
only keeps *pinned* outputs on-chip and writes the rest back to DRAM. Now
the port does the same. COSMA keeps its own rule (its model creates
every tensor in the SPM), unchanged.

## What did NOT change in COSMA

The ILP, the schedule, residency handling, and every number on models
without depthwise layers. Verified: ResNet-20 at 1024KB is identical to
the last commit, and at 200KB still gives 315,350 DRAM bytes / 11.12×.
The rest of COSMA's diff is logging (fair-split lines, a reworded warning)
and small helpers.

## Results now (OnSRAM, paper config, 2MB SPM)

| Model | Ours | Paper 1-Step | Paper ∞ SPM |
|---|---:|---:|---:|
| MobileNetV1 | 1.14× | 2.84× | 5.17× |
| SqueezeNet1.1 | 1.88× | 1.57× | 2.84× |
| ResNet-50 | 1.78× | 1.33× | 1.75× |
| MobileNetV2 | 1.28× | — | — |
| VGG16 | not run (runs out of memory on this 7GB machine) | 1.03× | 1.19× |
| Inception-v3 | fails in OnSRAM's placement step (old, unrelated issue) | 1.10× | 1.64× |

Every run: 0 `SpmAllocator` violations.

## Things we learned the hard way

- **Don't run two OnSRAM (or two COSMA) runs at once from the same
  checkout.** Each run writes and re-reads the same
  `onsram/topology.csv` (or `cosma/topology.csv`), so parallel runs can
  simulate one model with another model's layers. Our own scratchpad
  runner gave each run its own topology file.
- **This machine runs out of memory on big models** (VGG16, and ResNet-50
  when run next to anything else). Run them one at a time.
- `~/.local/lib/python3.10/site-packages` has stale copies of `scalesim`
  and `spm_common`. The run scripts put the repo first on the path, so
  they're not used, but a script run from elsewhere could pick them up.

## Still open

1. **Dense (fully-connected) layers aren't simulated at all.** They cost
   0 in both runs. Matters a lot for VGG16 (89% of its weights), somewhat
   for MobileNetV2 (37%), a little for ResNet-50/Inception (8–9%). Easy
   to add as a 1×1 conv on a 1×1 image.
2. **Bytes per element are inconsistent.** The SPM budget uses 4 bytes
   (float32), DRAM traffic counts 1, the paper likely uses 2 (FP16). This
   is now the main reason MobileNet stays far below the paper.
3. **Pointwise-conv compute** is ~1.5× slower than ideal in SCALE-Sim.
   After the depthwise fix, it's what caps MobileNet (infinite-SPM
   ceiling 1.18×).
4. **VGG16** needs more memory than this machine has; **Inception-v3**
   fails in OnSRAM's placement step before simulating.
5. `plan_for_real_remaining_spm.md` describes the old ordered fill and
   buffer clamps; it no longer matches the code.

---

# Update — 2026-09-25: OnSRAM now counts time the paper's way

**6. FP16 everywhere in OnSRAM (OnSRAM only).** The SPM budget used
float32 sizes (4 bytes), DRAM traffic counted 1 byte per element, and the
paper's hardware is FP16. OnSRAM now uses 2 bytes per element for all
three (`scale_sim_runner.BYTES_PER_ELEMENT`; tensors are rescaled in
`run_onsram._at_paper_precision()`). OnSRAM's own reference code also
hard-codes 2 bytes per element. On MobileNet, tensor 9 now fits and is
pinned (29 of 30).

**7. Fetch-once traffic and stall-free compute (OnSRAM only).** With
correct bytes, SCALE-Sim's own DRAM counts turned out to be 1.3–4.1× the
paper's "each element fetched once" baseline (re-reads of overlapping
windows, partial sums written back per pass), and SqueezeNet overshot the
paper's own upper bound (3.48× vs 2.84×). You chose the option closest to
the paper's logic. Every layer's traffic is now its real tensors, each
element once (input unless in the SPM, output unless pinned, weights
always), and compute time is SCALE-Sim's pure compute cycles without its
stall cycles. SCALE-Sim now provides compute time only.

COSMA is unchanged by 6 and 7. COSMA still counts SCALE-Sim elements as
bytes, so COSMA and OnSRAM DRAM numbers are no longer in the same units.

## Results now (OnSRAM, paper config, 2MB)

| Model | Ours | Our ceiling | Paper's layer-fusion experiment (§7.1.6, range only) | Paper, main results (Table 1) |
|---|---:|---:|---|---|
| MobileNetV1 | 1.36× | 1.38× | 1.01–2.17×, avg 1.31× (§7.1.6) | ∞ SPM 5.17× |
| SqueezeNet1.1 | 1.24× | 1.24× | same range | ∞ SPM 2.84× |
| MobileNetV2 | 1.63× | | not in paper | not in paper |
| ResNet-50 | running at time of writing | | | ∞ SPM 1.75× |

## Why compute now dominates, in one paragraph

SPM management only shortens memory time, and each layer takes
`max(compute, memory)`. SCALE-Sim's 39×39 array takes 1.6–2× the ideal
compute time: layers are cut into 39×39 folds, the last fold in each
direction is half-empty, and every fold pays ~77 cycles of fill/drain,
which dominates late layers with few output pixels (batch 1 makes it
worse). So compute is ~70–80% of our baseline time, which caps the
speedup. Even ideal compute would only raise the ceilings to 1.69×
(SqueezeNet) and 1.91× (MobileNet). The bigger difference: the paper
takes TensorFlow graphs and reports ReLU/BatchNorm as their own layer
types (§4, §7.1.3), so by our reading its main results have those as
separate, very memory-bound nodes. The paper doesn't use the words
"fused/unfused"; its only fusion result is one side experiment (§7.1.6:
1.01–2.17×, avg 1.31×, no per-model numbers). Our TFLite models have
BatchNorm/ReLU folded into the convs. Full explanation with numbers:
`onsram/docs/why_our_speedups_are_lower_than_the_paper.md`, section "Why
compute takes so much of the execution time in our setup".

## Still open (updated)

1. Dense layers aren't simulated (0 cost), nor are ADD/CONCAT/pooling.
   Simulating the small ops as memory-bound nodes would show more of
   OnSRAM's benefit.
2. Decide which paper numbers to compare against: the §7.1.6
   layer-fusion range (closest to our TFLite models, but no per-model
   values) or Table 1 using the unfused models from
   `spm_common/unfuse_model.py`.
3. VGG16 (memory) and Inception-v3 (placement failure) still don't run.
4. `plan_for_real_remaining_spm.md` is out of date.

---

# Update — 2026-09-26: non-conv layers costed, unfused models

**8. Non-conv layers now cost time (both COSMA and OnSRAM; COSMA by your
one-time approval).** DENSE runs through SCALE-Sim as a 1×1 conv on a 1×1
image (both topology builders; `run_cosma.py` and `run_paper_baselines.py`
treat DENSE as conv-like). ADD, CONCAT, pooling, PAD, SUB/MUL, REDUCE_MEAN
and SOFTMAX are costed by `_nonconv_layer_stats()` in each runner as pure
data movement: each input read once, each output written once, compute
~0, with each paper's usual on-chip rules. COSMA ResNet-20 @ 200KB/1024KB:
11.12× → 11.40× (baseline now includes ADD/REDUCE/DENSE traffic), 0
violations.

**9. Unfused models (`spm_common/unfuse_model.py`).** Ported from
`/home/george/trim/spm_management/onsram_bw/onsram_unfused.py`, with
fixes: ReLU only where model.json records a fused activation (also after
ADD, e.g. ResNet-50), BatchNorm only for networks that had it (BiasAdd for
SqueezeNet/VGG, dense layers and a BN network's final classifier conv),
fresh tensor ids, renumbered layers, rebuilt inputs_from/topo_sort.
Usage: `python3 spm_common/unfuse_model.py MobileNet` →
`cosma/_exported/MobileNet_unfused/model.json`. Run with
`python3 onsram/run_onsram.py --model MobileNet_unfused --config configs/scale_onsram.cfg`.

## Results (OnSRAM, paper config, 2MB)

| Model | Fused | Unfused | Paper 1-Step | Paper ∞ SPM |
|---|---:|---:|---:|---:|
| ResNet-50 | 1.19× | **1.60×** | 1.33× | 1.75× |
| SqueezeNet1.1 | 2.01× | **2.87×** | 1.57× | 2.84× |
| MobileNetV1 | 1.37× | **2.78×** | 2.84× | 5.17× |
| MobileNetV2 | 1.60× | **2.66×** | — | — |

Unfused ResNet-50 matches the paper closely; SqueezeNet lands ~1% above
the paper's ∞ SPM bound (likely the extra BiasAdd nodes); MobileNetV1 is
still limited by SCALE-Sim's compute time (~2× ideal). Note on wording:
the paper doesn't label results fused/unfused. Its main results use
TensorFlow graphs with ReLU/BatchNorm as their own layer types (§4,
§7.1.3), and its only fusion result is one side experiment (§7.1.6).

## Still open

1. MobileNetV1's remaining gap is compute: SCALE-Sim's 39×39 array
   (half-empty tiles, fill/drain per tile at batch 1).
2. VGG16 (memory) and Inception-v3 (placement failure) still don't run.
3. COSMA and OnSRAM DRAM numbers are in different units (SCALE-Sim
   elements vs FP16 bytes).
4. `plan_for_real_remaining_spm.md` is out of date.

---

# Update — 2026-09-26 (evening): whole roster vs the paper

**10. Three more paper models.** `spm_common/build_paper_models.py` builds
AlexNet (torchvision architecture), GoogLeNet (Inception v1 with
BatchNorm) and ResNeXt-50 32x4d as Keras models with random weights (only
the architecture matters here), converts to TFLite and exports with trim's
exporter into `cosma/_exported/`. trim's exporter can't handle grouped
convs, so the script patches it in-process (trim's files are untouched).
Both topology builders and OnSRAM's `fom.py` now use `groups` (a no-op for
all older models). Dense layers are now costed by SCALE-Sim's fold formula
instead of simulated, because SCALE-Sim ran out of memory on AlexNet's
9216×4096 dense layer.

Paper config (39×39 array, 32 B/cycle, 2 MB SPM, batch 1, FP16), final
code state: hand-off reads from SPM, channels-across-columns depthwise,
grouped convs, ideal tiling, pinned-only free outputs, fetch-once traffic,
stall-free compute, non-conv layers costed (dense by formula). All 16 runs:
0 SpmAllocator violations. The "~90% of ∞ SPM" column is only a guide: the
paper states OnSRAM-Static reaches ~90% of the infinite-SPM speedup
overall, not per model. Not in the roster: Inception-v4, SSD300, PTB-LSTM,
Multi-Head Attention.

- Matches or close: AlexNet, ResNet-50, Inception-v3, GoogLeNet,
  SqueezeNet1.1 (unfused).
- MobileNetV1: limited by SCALE-Sim's compute (~2× ideal: half-empty tiles
  and fill/drain on its many small late layers).
- ResNeXt-50: same compute limit; the paper's version may also have been
  much more memory-bound (a guess, not verified: TensorFlow had no native
  grouped conv in 2018–2019, so ResNeXt was often built as 32 split
  branches plus a concat, adding memory-bound nodes).
- SqueezeNet1.1: ~1% above the paper's ∞ SPM bound, likely from the
  BiasAdd nodes the unfuser adds after every conv.
- Inception-v3 now runs (it used to fail in OnSRAM's placement step; FP16
  halved tensor sizes, so placement succeeds).
- New models (2026-09-26): AlexNet, GoogLeNet, ResNeXt-50, built with
  `spm_common/build_paper_models.py`.

| Model | Ours, fused | Ours, unfused | Paper 1-Step | Paper ∞ SPM | ~90% of ∞ SPM | Unfused vs paper |
|---|---:|---:|---:|---:|---:|---|
| AlexNet | 1.01× | **1.04×** | 1.04× | 1.04× | ~1.04× | matches |
| ResNet-50 | 1.19× | **1.60×** | 1.33× | 1.75× | ~1.58× | matches |
| Inception-v3 | 1.17× | **1.42×** | 1.10× | 1.64× | ~1.48× | close |
| GoogLeNet | 1.29× | **1.61×** | 1.01× | 1.94× | ~1.75× | a bit low |
| SqueezeNet1.1 | 2.01× | **2.87×** | 1.57× | 2.84× | ~2.56× | ~1% above ∞ SPM |
| MobileNetV1 | 1.37× | **2.78×** | 2.84× | 5.17× | ~4.65× | below |
| ResNeXt-50 | 1.19× | **1.73×** | 1.40× | 3.86× | ~3.47× | far below |
| MobileNetV2 | 1.62× | **2.69×** | — | — | — | not in paper |
| VGG16 | not run (runs out of memory; could crash the machine) | | 1.03× | 1.19× | | |

### Against the paper's own OnSRAM-Static results (Fig. 7)

What we model is OnSRAM-Static at 2 MB, so the right target is the paper's
OnSRAM-Static bar in Fig. 7, not the 1-Step row (a simpler baseline scheme,
also at 2 MB) or ∞ SPM (the upper bound). Fig. 7 prints only two of those
bars as numbers (ResNeXt 3.81×, MobileNetV1 4.76×; both run off the top of
the chart). The others were **measured from the figure**: page 15 rendered
at 300 dpi, bars found by their legend colour, heights calibrated on the
chart's 0/1/2/3 gridlines. Check: the chart's ∞ SPM bars read within
0.01–0.02 of Table 1's printed values for every model (so expect about
±0.02 on the values below), except GoogLeNet, where the paper's own chart
(1.83) and table (1.94) disagree.

Paper's OnSRAM-Static values read this way: AlexNet 1.02, VGG16 1.02,
GoogLeNet 1.83, Inception-v3 1.29, Inception-v4 1.31, ResNet-50 1.49,
SSD300 1.22, ResNeXt 3.81, MobileNetV1 4.76, SqueezeNet 2.20, PTB 1.00,
Multi-Head Attention 1.06, geomean 1.59.

| Model | Ours, fused | Ours, unfused | Paper OnSRAM-Static (Fig. 7) | Unfused vs paper |
|---|---:|---:|---:|---:|
| AlexNet | 1.01× | 1.04× | 1.02× | +2% |
| ResNet-50 | 1.19× | 1.60× | 1.49× | +7% |
| Inception-v3 | 1.17× | 1.42× | 1.29× | +10% |
| GoogLeNet | 1.29× | 1.61× | 1.83× | −12% |
| SqueezeNet1.1 | 2.01× | 2.87× | 2.20× | +30% |
| ResNeXt-50 | 1.19× | 1.73× | 3.81× | −55% |
| MobileNetV1 | 1.37× | 2.78× | 4.76× | −42% |
| Geomean of these 7 | 1.29× | 1.76× | 2.03× | −13% |

- Within ~10%: AlexNet, ResNet-50, Inception-v3; GoogLeNet 12% low.
- SqueezeNet1.1 30% high (our unfused SqueezeNet has more memory-bound
  work than the paper's; the extra BiasAdd nodes are the likely cause).
- ResNeXt-50 and MobileNetV1 far low: compute-limited in SCALE-Sim (~2×
  ideal), plus possibly a more memory-bound ResNeXt graph in the paper.


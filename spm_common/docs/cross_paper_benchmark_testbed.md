# Cross-paper benchmark testbed: COSMA vs. OnSRAM vs. SMM

This doc lives under `spm_common/docs/` because that's already the
established home for cross-cutting SPM analysis in this repo (see
`spm_traffic_fidelity.md` next to it). **SMM is in scope for this doc despite
not using `spm_common`'s shared code at all** — SMM has its own model
pipeline (`smm/smm_helpers/`, `smm/topologies/`), completely separate from
the `spm_common`/`cosma/_exported` pipeline COSMA and OnSRAM share. This is a
methodology spec only: no new runner code, no generated `.cfg` files, no
results actually run. It specifies what a reader would need to execute the
comparison themselves, and — more importantly — what would make that
comparison honest rather than misleading.

## 1. Purpose

COSMA, OnSRAM, and SMM are each individually validated against their own
paper's reported numbers, on their own model rosters, at their own default
hardware configs. That answers "is each port faithful to its paper." It does
not answer the practically useful question: **for a given workload and a
given on-chip memory budget, which of the three memory-management strategies
actually wins, and under what conditions does the ranking flip?**

Answering that needs the same models, the same budget points, and the same
array/bandwidth settings run through all three papers' own already-existing
run scripts, with results normalized into one schema. That's this doc's
subject — specified as a **full factorial** (every axis combined, not a
cheaper staged subset), while being explicit about where full-factorial
numbers are and aren't comparable, and what they cost to actually produce.

## 2. Five confounds to read before touching the numbers

These aren't hypothetical. Confound 0 and 3 are newly confirmed in this
session by reading the repo's own docs and `.cfg` files directly; 1, 2, and 4
were already on record in `spm_traffic_fidelity.md` from earlier sessions.
Every comparison table later in this doc restates the relevant ones inline —
they are not a disclaimer to read once and forget.

### 2.0 Only SMM's paper actually used SCALE-Sim

Confirmed directly from this repo's own docs, not assumed:

- **SMM's paper genuinely evaluated through SCALE-Sim.**
  `smm/docs/smm_implementation.md:81-82` quotes the paper's own §4 directly:
  "it took approximately one minute to generate the management schemes...
  while for **the SCALE-Sim baseline** it took more than 5 hours."
  `configs/scale_smm.cfg`'s 16×16 / output-stationary setup is a literal
  replay of the paper's own simulator.
- **COSMA's paper has no simulator at all.** `cosma/docs/PIPELINE.md:13`:
  "paper's own evaluation is purely analytical (its objective value, no
  simulator). We went further: we plug COSMA's actual decisions into
  [SCALE-Sim]." COSMA's array size, bandwidth, and cycle numbers in this
  testbed are this repo's own invention — there is no "39×39" or "32×32" the
  COSMA paper ever specified, because it never ran on any array simulator.
  The **only** COSMA metric with a real paper number behind it is
  non-compulsory DRAM-byte reduction % (`cosma/docs/results_plan.md`'s own
  mapping of the paper's §V-A metrics onto this port's evaluation). COSMA's
  speedup/cycle numbers are simulation-only, unvalidated against anything
  the paper reported.
- **OnSRAM's paper used its own separate, non-SCALE-Sim tool.**
  `onsram/docs/why_our_speedups_are_lower_than_the_paper.md:680`: the paper's
  §3.3 calls that tool a "cycle-accurate simulator," though this repo's own
  prior analysis notes its §6 methodology description reads more like a
  closed-form bandwidth model. Unlike COSMA, OnSRAM's paper *does* publish
  speedup numbers (Fig. 7 / Table 1) — but from that other tool.
  `configs/scale_onsram.cfg`'s 39×39 / 32 B-cycle setup is this repo's
  **best-effort reconstruction**, tuned so SCALE-Sim's replay lands close to
  those published numbers — not a replay of OnSRAM's own simulator, because
  OnSRAM's own simulator isn't SCALE-Sim.

**Consequence:** the array-size/bandwidth axis in §4 is not "each paper's own
hardware regime." It's this repo's SCALE-Sim-replay sensitivity probe —
literal and paper-grounded only for SMM, an approximation-target for OnSRAM,
and ungrounded-but-still-useful for COSMA. Wherever COSMA appears in a
comparison, DRAM-byte-reduction % — not speedup — is the metric with an
actual paper number behind it.

### 2.1 Equal KB isn't equal hardware across papers

COSMA and OnSRAM budget *only activations* — weights stream through
un-tracked against the budget (`spm_traffic_fidelity.md`: "Weights never
compete for the SPM budget in simulation"). SMM's GLB budget is a *unified*
ifmap+filter+ofmap buffer — weights count against it. A "256KB" SMM run and a
"256KB" COSMA/OnSRAM run are not the same amount of usable on-chip memory.
State this next to every comparison table, not once in an intro.

### 2.2 DRAM-byte units didn't match across papers — now a real, swept `--precision` axis (§4.5)

**Correction to an earlier draft of this doc:** COSMA's own cost-accounting
(`spm_common/graph_builder.py:62`, `compute_size_bytes(shape, dtype)`) was
never "raw element counts" — it already reads each tensor's real `dtype`
field out of model.json and multiplies correctly, confirmed by COSMA's own
code comment at `onsram/onsram_helpers/scale_sim_runner.py:69`: "COSMA is
unaffected (its own runner keeps model.json's dtype sizes)." The actual gap
was narrower: OnSRAM hardcoded `BYTES_PER_ELEMENT = 2` (FP16) in
`onsram/onsram_helpers/scale_sim_runner.py:70`, and SMM hardcoded
`bytes_per_elem = 1` (INT8) directly in `smm/smm_helpers/scale_sim_runner.py:140`
(plus matching constants in `smm/smm_helpers/dense_costing.py:52` for DENSE
layers and `smm/smm_helpers/baseline.py`'s `word_size` default for its own
fixed-partition baseline) — both deliberately, matching each paper's own
hardware precision regardless of the input model's export format.

**This has now been wired up as a real, swept parameter (§4.5)** rather than
left as a documented limitation. The units point still matters for
interpreting results: compare `dram_reduction_pct`/`speedup_pct` (relative,
safe even across differing precisions) rather than raw absolute bytes across
papers *unless* both sides of a comparison were run at the same
`--precision` value.

### 2.3 SMM's paper-matched config has bandwidth coupled to array width, not fixed

`configs/scale_smm.cfg` sets `InterfaceBandwidth: CALC`, while
`configs/scale.cfg` (COSMA's default) and `configs/scale_onsram.cfg` both set
`InterfaceBandwidth: USER` with an explicit numeric `Bandwidth`. Under `CALC`
mode, SCALE-Sim derives effective bandwidth from array geometry instead of
using the literal `Bandwidth: 16` value (`scalesim/scale_config.py:81-138`,
`scalesim/memory/read_buffer_estimate_bw.py`). This means:

- SMM's existing paper-validated numbers were produced with bandwidth *not*
  independently fixed from array size.
- Before running the array-size × bandwidth axis in §4 for SMM, any `.cfg`
  variant derived from `scale_smm.cfg` must be flipped to
  `InterfaceBandwidth: USER` with an explicit `Bandwidth` value — otherwise
  array size and bandwidth are not orthogonal axes for SMM, and the point of
  a 2D sweep breaks for that paper specifically.

Separately: COSMA's resident-buffer classes
(`scalesim/memory/cosma_resident_buffers.py`) subclass `ReadBufferEstimateBw`
— the same buffer class `CALC` mode uses. COSMA's own `Bandwidth` field
behavior should be cross-checked against `cosma/ITERATION_HISTORY.md` (which
already investigated this) before trusting COSMA's bandwidth axis. This doc
flags it as a pre-flight item (§7) rather than asserting an answer that
wasn't independently re-derived here.

### 2.4 Dataflow is paper-defining, not a knob

COSMA/OnSRAM use weight-stationary; SMM uses output-stationary, by paper
design. Hold this fixed per paper rather than "equalizing" it — it's not a
missing axis, it's a deliberate one.

## 3. Model roster: the honest overlap, not an invented one

Built from each paper's *actually-validated* roster — not every model each
paper could in principle run, only ones with confirmed end-to-end numbers
today.

| Model | COSMA | OnSRAM | SMM | Why it's in the matrix |
|---|---|---|---|---|
| GoogLeNet | ✅ | ✅ | ✅ | Only true 3-way match — the head-to-head anchor model for every stage, including the array×bandwidth axis |
| MobileNetV2 | ✅ (CIFAR-10 variant — flag input-resolution mismatch vs. the other two) | ✅ | ✅ | Depthwise/grouped-conv-heavy, memory-bound — OnSRAM's and SMM's claimed sweet spot |
| ResNeXt-50 | ✅ (COSMA's strongest claimed win: 85.6% DRAM reduction, 2.86× speedup at tight budget) | ✅ (paper's 2nd-strongest claim, 3.81×) | ❌ | High tensor reuse / skip connections — the regime where COSMA's ILP generality should beat OnSRAM's greedy heuristic, if it does |
| SqueezeNet (small / 1.1) | ✅ | ✅ | ❌ | Compute-bound control case — low expected benefit for either |
| ResNet family | ResNet-20-CIFAR10 (COSMA's documented *non*-win: linear chain, nothing to reuse) | ResNet-50 | ResNet-18 | Report as **three separate rows, explicitly labeled by variant** — different depths and datasets, never implied equivalent |

GoogLeNet is the only model where one row can honestly compare all three.
Every other row is 2-of-3, or 3-of-3 with a depth/dataset mismatch flagged in
the row itself — every table using this roster repeats that caveat in a
footnote rather than hiding it in prose above the table.

## 4. Parameter axes and values

### 4.1 Budget (primary axis — every script sweeps this natively in one invocation)

64, 128, 256, 512, 1024 KB — matches SMM's own default sweep exactly.

| Paper | Flag | Values to pass |
|---|---|---|
| COSMA | `--budgets-kb` | `64 128 256 512 1024` |
| OnSRAM | `--spm-mb` | `0.0625 0.125 0.25 0.5 1.0` |
| SMM | `--glb_kb` | `64 128 256 512 1024` |

### 4.2 Array size

16×16, 32×32, 64×64. Per §2.0: literal paper replay only for SMM (16×16 is
the paper's own config); for COSMA, a pure simulation-sensitivity probe with
no paper counterpart (32×32 is this repo's existing default, not a COSMA
paper value); for OnSRAM, brackets the value (39×39) this repo already uses
to approximate the paper's published speedups — keep 39×39 as a **separate
reference row** outside the clean `{16,32,64}` grid rather than forcing the
grid onto it.

### 4.3 Bandwidth

8, 16, 32 (explicit `USER`-mode value, in each paper's own config unit —
bytes/cycle for COSMA/OnSRAM, elements/cycle for SMM at its 1-byte/elem
precision, numerically equal here but a distinct unit worth noting).
Brackets SMM's own paper value (16) and the value (32) this repo's OnSRAM
reconstruction uses to approximate its paper; has no COSMA-paper counterpart,
for the same reason as the array-size axis.

### 4.4 Held fixed (not swept)

- Dataflow: paper-native per §2.4 (COSMA/OnSRAM = weight-stationary, SMM =
  output-stationary).
- Batch size: 1, matching all three papers' own evaluation convention.
- Solver (COSMA only): CBC unless a Gurobi license is available on the run
  machine. Record which solver, and its time limit, alongside every COSMA
  row — it affects both runtime and whether the ILP reported `status ==
  Optimal` or a time-limited bound.

### 4.5 Precision — fp16, int8 (real axis, not held fixed)

Per your choice, every (model, budget, array, bandwidth) point runs twice,
once at each precision, for all three papers:

| Paper | Mechanism | Default (flag omitted) | New flag/edit |
|---|---|---|---|
| COSMA | No source change needed — COSMA already reads each tensor's real `dtype` from model.json (§2.2). Pre-export the model once per precision via `spm_common/model_resolver.py`'s existing `mode=` support (e.g. `python3 /home/george/Desktop/trim/python_scripts/export_model.py --model <tflite> --out <dir> --mode fp16`, and again with `--mode int8`), then point `--model-json`/`--models` at each resulting `model.json`. | fp32 (unaffected — not part of this axis) | none — orchestration-only, `cosma/` untouched |
| OnSRAM | `onsram/run_onsram.py --precision {fp32,fp16,int8}`, sets `scale_sim_runner.BYTES_PER_ELEMENT` to {4,2,1} before the run. Leaves `onsram_helpers/fom.py`'s own hardcoded `bytes_per_element = 2` in `_infer_node_type()` untouched on purpose — that's a preserved reference-implementation quirk for node-type-classification fidelity (see that file's module docstring), orthogonal to the tensor/DRAM byte accounting this flag controls. | `fp16` (byte-identical to every previously-validated number — regression-checked: MobileNet@2MB, 29/30 pinned, 1.5312 MB peak, matches with and without the flag) | `onsram/run_onsram.py`, `onsram/onsram_helpers/scale_sim_runner.py` (just the module-level `BYTES_PER_ELEMENT` default, read dynamically) |
| SMM | `smm/run_smm.py --precision {fp32,fp16,int8}`, threads bytes-per-element {4,2,1} into all three cost paths: the Hom/Het runner (`SMMScaleSimRunner(bytes_per_elem=...)`), the fixed-partition baseline (`run_baseline(word_size=...)`), and DENSE costing (`cost_dense_layers_in_model(bytes_per_element=...)`). | `int8` (the paper's own Sec. 4 hardware — matches every previously-validated number) | `smm/run_smm.py`, `smm/smm_helpers/scale_sim_runner.py`, `smm/smm_helpers/baseline.py` |

Both OnSRAM's and SMM's defaults were verified to reproduce prior numbers
exactly with the flag omitted, and to change pinning/tiling decisions (not
just rescale a byte count) when a different `--precision` is passed — e.g.
OnSRAM's MobileNet@2MB goes from 29/30 pinned at `fp16` to 30/30 pinned at
`int8` (a real decision change: smaller elements let one more tensor win the
FoM competition), confirming the flag reaches the actual budget-fit logic,
not just final reporting. SMM's secondary, non-benchmark debug CLI at the
bottom of `smm/smm_helpers/scale_sim_runner.py` (`if __name__ == "__main__":`)
was deliberately left without a `--precision` flag — it's not part of the
documented `run_smm.py`/`run_all_models.sh` pipeline this testbed uses, and
it keeps working unchanged at the preserved `int8` default.

This doubles §5's matrix: every (model, budget, array, bandwidth) point now
has an `fp16` row and an `int8` row, for all three papers.

## 5. Full factorial matrix

5 models × 3 array sizes × 3 bandwidths × 2 precisions × 3 papers = **270
process invocations**, each sweeping all 5 budgets natively via the script's
own multi-value flag — so the resulting dataset is **1350 rows**
(model × budget × array × bandwidth × precision × paper) without 1350
separate process launches. Precision (§4.5) requires a separate invocation
per value for all three papers — for OnSRAM/SMM because `--precision` is a
per-process flag like `--config`, not a multi-value sweep flag; for COSMA
because each precision is a physically different pre-exported `model.json`.

**Cost reality, not hidden:** OnSRAM's own 16-run budget-only sweep already
took ~1h55m (`onsram/docs/onsram_model_roster.md`). COSMA's ILP solve time
grows with budget and model size under CBC. A full 270-invocation grid is
realistically a multi-day serial undertaking.

**Mitigation:** each paper writes to its own scratch topology file
(`cosma/topology.csv`, `onsram/topology.csv`, SMM's own) — so **the three
papers can run concurrently with each other** (3-way parallel), just never
two invocations of the *same* paper at once (per each paper's own
documented constraint).

## 6. Config generation: the 9 (array, bandwidth) combinations

For the array×bandwidth axis, each paper needs 9 derived `.cfg` variants
(27 total), built from that paper's existing native config
(`configs/scale.cfg`, `configs/scale_onsram.cfg`, `configs/scale_smm.cfg`) by
overriding `ArrayHeight`/`ArrayWidth` and `Bandwidth` (plus the matching
`IfmapSRAMBankBandwidth`/`FilterSRAMBankBandwidth` fields to the same value,
following each base config's own existing convention of keeping those three
fields equal). **Critically, for every SMM-derived variant, also set
`InterfaceBandwidth: USER`** (overriding the base config's `CALC`) — per
§2.3, without this the bandwidth column below is not actually independent of
array size for SMM.

| Combo | ArrayHeight/Width | Bandwidth | Applies to |
|---|---|---|---|
| 1 | 16 | 8  | all 3 papers |
| 2 | 16 | 16 | all 3 papers |
| 3 | 16 | 32 | all 3 papers |
| 4 | 32 | 8  | all 3 papers |
| 5 | 32 | 16 | all 3 papers |
| 6 | 32 | 32 | all 3 papers |
| 7 | 64 | 8  | all 3 papers |
| 8 | 64 | 16 | all 3 papers |
| 9 | 64 | 32 | all 3 papers |

`IfmapSRAMBankBandwidth` and `FilterSRAMBankBandwidth` are set equal to the
combo's `Bandwidth` value in every variant, matching how each base `.cfg`
already keeps those three fields equal. `InterfaceBandwidth: USER` is set in
all 27 variants (already true for the COSMA/OnSRAM base configs; a required
override for every SMM-derived one). Everything else in each base config
(SRAM sizes, offsets, dataflow, sparsity section) is left untouched.

This table is the spec; actually writing the 27 `.cfg` files is execution
work, out of scope for this doc.

## 7. Pre-flight checklist (read before running anything from this doc)

1. Flip `InterfaceBandwidth` to `USER` with an explicit value in every
   array/bandwidth `.cfg` variant derived from `scale_smm.cfg` (§2.3, §6).
2. Before trusting COSMA's bandwidth axis, re-check
   `cosma/ITERATION_HISTORY.md`'s existing finding on which buffer class
   `scale.cfg`'s bandwidth mode actually exercises for COSMA's resident
   buffers (§2.0, §2.3).
3. Exclude VGG16 from the roster on this machine (documented OOM in prior
   sessions).
4. One sweep at a time *per paper* (shared scratch `topology.csv`/similar);
   different papers may run concurrently with each other (§5).
5. Record COSMA's solver (CBC vs. Gurobi) and time limit per row (§4.4).

## 8. Output normalization schema

One flat schema, filled in by hand (or a future script) from each paper's
own existing output — no new instrumentation needed:

```
paper, model, model_variant_note, array_h, array_w, bandwidth, budget_kb,
precision, solver_or_policy, baseline_dram, optimized_dram,
dram_reduction_pct, baseline_cycles, optimized_cycles, speedup_pct
```

| Paper | Existing output to read from |
|---|---|
| COSMA | `run_experiments.py --out-csv` summary CSV (`cosma/results/*.csv`) |
| OnSRAM | `run_onsram.py --out-csv` summary CSV (`onsram/results/*.csv`) |
| SMM | `run_smm.py --out`-directory CSV, or `run_all_models.sh`'s per-model logs + `_summary.log` |

**Primary comparison metric: `dram_reduction_pct`**, relative to each
paper's own baseline — safe across the unit mismatch in §2.2, and the only
metric with a genuine paper number behind it for all three papers (per §2.0,
COSMA's paper validates *only* this metric).

`speedup_pct` is secondary: a real, paper-sanity-checkable metric for OnSRAM
and SMM, but for COSMA it must be labeled in every table and plot as
**unvalidated against the paper** — a SCALE-Sim-replay artifact this repo
produces, not a number COSMA's paper ever reported.

Absolute byte columns (`baseline_dram`, `optimized_dram`,
`baseline_cycles`, `optimized_cycles`) are within-paper diagnostics only —
never compare them paper-to-paper directly (§2.2).

## 9. Presentation

- **Line plot, budget sweep:** x = `budget_kb`, y = `dram_reduction_pct`, one
  line per paper, small-multiple per model (5 panels, per §3's roster) —
  shows the crossover points ("below X KB, paper Y wins; above, paper Z
  wins") that motivate this whole exercise.
- **Heatmap, hardware sweep:** x = array size, y = bandwidth, color =
  `speedup_pct`, one grid per paper at the GoogLeNet anchor model + mid
  (256KB) budget — shows each paper's compute-bound-vs-memory-bound regime
  boundary. **COSMA's grid is captioned as simulation-sensitivity only**
  (§2.0), not a paper comparison, since no paper speedup number exists to
  validate it against.

## 10. What this doc deliberately does not include

- No new runner/orchestration script — each paper's own existing,
  already-validated entry point (`run_cosma.py`/`run_experiments.py`,
  `run_onsram.py`, `run_smm.py`/`run_all_models.sh`) is reused as-is.
- No generated `.cfg` files — §6 is the spec for them, not the files
  themselves.
- No results actually run — this doc's scope ends at the written
  methodology.

Precision (§4.5) *was* implemented as real code, as an explicit exception to
the above: `onsram/run_onsram.py`, `onsram/onsram_helpers/scale_sim_runner.py`,
`smm/run_smm.py`, `smm/smm_helpers/scale_sim_runner.py`, and
`smm/smm_helpers/baseline.py` now take a `--precision` flag, regression-tested
to reproduce every previously-validated number unchanged at their preserved
defaults (`fp16` for OnSRAM, `int8` for SMM). `cosma/` was not touched, per
standing preference to keep it stable — COSMA's precision axis is handled
entirely by which pre-exported `model.json` you point it at (§4.5).

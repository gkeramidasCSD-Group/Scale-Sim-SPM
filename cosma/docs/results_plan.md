# Results Plan: mapping our evaluation onto the COSMA paper's

Standing reference for running tests that produce genuinely comparable
numbers to the COSMA paper (arXiv:2311.18246), and for knowing exactly which
numbers *aren't* comparable yet, and why. Update this file as gaps close —
it's meant to stay current, not be a one-time snapshot.

## 1. Purpose

Every individual gap between our implementation and the paper's own
evaluation has been noted somewhere in `STATUS.md`/`ITERATION_HISTORY.md`
over time, but never assembled into one place describing *how to actually
run a test* that means the same thing as one of the paper's own numbers.
That's what this file is for.

## 2. Paper's evaluation methodology (§V-A, arXiv:2311.18246)

**Three memory budgets per model** (§V-A2):
- **`M_R`** ("Minimum Memory Required"): *"the maximum memory required by a
  single operator in the DNN. This is the sum of the sizes of all input and
  output tensors that must reside in memory during the operation."*
- **`M_P`** ("Minimum Peak Memory Footprint... required for executing the
  entire DNN"): the true, schedule-optimal peak — computed via COSMA's own
  §III-E1 optimization (Eq.13–15) for human-designed DNNs, or via a separate
  tool called HMCOS for NAS models.
- **`M_H`** ("Hybrid"): `(M_R + M_P) / 2`.

**Primary metric**: *"non-compulsory off-chip data access volume"* — bytes
spilled plus bytes retrieved. Secondary metric: solving time (seconds).

**Two tensor-tracking settings**: "activation tensors only" vs. "activation
tensors and parameter tensors." Parameter tensors are described as *"used
only for single operators, e.g., the weight tensors of a convolution
layer,"* while activation tensors are *"used across different operators."*

**Comparison baselines** (§V-A2), 4 combinations: TensorFlow-Lite's linear
allocator × {default schedule, MPMF schedule} × {Belady's algorithm,
ILP-based greedy replacement}.

**Solver/hardware** (§V-A): Gurobi, Apple M1 Pro, 16GB RAM, 24-hour time
limit per ILP call. Human-designed DNNs solved in **0.296s on average**; NAS
models (via heuristics) in ~2 minutes.

**Models tested** (§V-A1):
- 10 human-designed (Figure 3): image classification — **ResNet-50,
  DenseNet, ResNeXt**; video classification — R2Plus1D, S3D; semantic
  segmentation — FCN, L-RASPP, DeepLabV3; transformer-based — Transformer,
  ViT.
- 4 NAS-generated (Figure 4, Table II): PNASNet-5, AmoebaNet-D, NASNet-A,
  DARTS — noted as having *"complex graph structure and irregular wiring
  between nodes,"* requiring divide-and-conquer/fixed-schedule heuristics
  rather than direct solving.

## 3. Mapping table: paper concept → our implementation status

| Paper concept | Our status | Where |
|---|---|---|
| `M_R` | **Have it, faithful** | `cosma_Ilp.compute_structural_minimum_bytes()` — max over operators of activation-input + output bytes. Now confirmed against the primary source (§V-A, Fig.3 caption): the paper's own `M_R` is *also* activation-only — ours isn't under-counting it. The paper separately reports a parameter-inclusive `M_Rp` (same definition, bigger tensor set) alongside `M_R` for every human-designed DNN; this port doesn't compute that second number yet (see the activation+parameter row below). |
| `M_P` | **Have it, real §III-E1/Eq.13-15 solve** | `cosma_Ilp.compute_true_mpmf_bytes()` — a genuinely separate, free-schedule ILP (`C[a,t]` a real variable, not fixed), reusing none of `build_cosma_model()`'s memory-allocation machinery (no `L`/Eq.9/10/11 — the paper's own text: "memory allocation is not considered" in this mode). `compute_mpmf_bytes()` (the old fixed-schedule proxy) is unchanged and still used by `--bounds-only`'s fast default path — `M_R ≤ true_M_P ≤ MPMF-proxy` always holds, verified on 6 models (see §4). Exposed via `visualize_spm.py --bounds-only --true-mpmf` (opt-in — a real solve, not instant). |
| `M_H` | **Have it** | `(M_R + true_M_P) / 2`, computed inline (`visualize_spm.print_true_mpmf()`) wherever `--true-mpmf` is used — no dedicated function needed, it's one line. |
| Primary metric (spill+retrieve bytes) | **Have it now** | `run_cosma.py`'s `total_non_compulsory_access_bytes` (added alongside this doc) = `total_idealized_spill_bytes + total_idealized_retrieve_bytes + total_real_retrieve_bytes`. |
| Activation-only tracking | **Have it — it's the only mode** | `graph_builder.load_graph()`, unconditional. |
| Activation+parameter tracking | **Not implemented** | Corrected against the primary source (§V-B): the paper's own "both activation tensors and parameter tensors" setting is *not* a different formulation — same Eq.1-12, same C/P/S/R model, just weights included in tensor set `A`. It's one of the paper's two standard settings, reported for all 10 human-designed DNNs in Fig.3 (as `M_Rp`/`M_Hp`/`M_Pp`), not a side study. The ILP side of this port needs no new constraint logic to match it — just extending `graph_builder.load_graph()`'s tensor scope. See §6 below for the real remaining question, which is about the *SCALE-Sim-simulated* numbers, not the ILP. |
| Comparison baselines (TFLite × Belady/greedy) | **Have it, done and verified** | `run_paper_baselines.py` — TFLite's real linear-allocator algorithm (`helpers/tflite_arena_allocator.py`, ported from `arena_planner.cc`/`simple_memory_arena.cc`) × {default, MPMF} schedule × {Belady, ILP-greedy} replacement, all 4 combos run through the same real SCALE-Sim/`run_cosma_aware()` accounting COSMA's own numbers use. Built and verified across 6 phases (incl. reproducing the paper's own "Belady is suboptimal vs. greedy-ILP" finding) — see `baseline_construction.md`. This line was stale (said "not implemented") after that work landed; corrected 2026-09-29. |
| Operator scheduling | **Implemented in the main pipeline too** | `build_cosma_model(..., free_schedule=True)` — `C[a,t]` is now a real decision inside the *main* spill/retrieve pipeline itself (not just the isolated `M_P` model), and the ILP's chosen order drives a real SCALE-Sim re-simulation via `baseline.run_cosma_aware()`'s `schedule` param. Opt-in, default `False` (byte-identical to every previously published number). Verified on both toy fixtures, the small custom DenseNet fixture (real SCALE-Sim run), and ResNet-20-CIFAR10 — see §4/§7. Not yet run at ImageNet scale (Inception-V3/ResNet-50/DenseNet-121) — see §6 item 1. |
| Divide-and-conquer (NAS-scale) | **Permanently out of scope** | Per explicit standing project direction. |
| Solver | **Gurobi available, use it** | A working Gurobi license (WLS, academic) is active in this environment — confirmed 2026-09-29 (`run_cosma.py --solver gurobi` on ResNet-50: solved in 0.40s, right in line with the paper's own 0.296s average). This line previously said "PuLP/CBC, not Gurobi... fixed difference," which is stale/wrong — pass `--solver gurobi` explicitly (CBC stays the default everywhere for no-license environments). Not yet the default because most of this project's existing recorded numbers were produced with CBC; re-verify byte-identical objective values before assuming solver choice is transparent to results. |

## 4. Model roster and bounds

**2026-09-30 additions below the original table**: ResNet-50/DenseNet-121
INT8 (datatype parity with the paper's "all data are 8-bit"), and 3 new
sourced models (ResNeXt, S3D, FCN) plus DeepLabV3 unblocked — see
`paper_model_roster.md` for the full per-model engineering detail (exporter
fixes, blockers found/fixed); this table only has the bounds/results
summary. "MPMF proxy" below is always the fixed-schedule proxy
(`compute_mpmf_bytes()`), not a real free-schedule `true_M_P` solve, unless
a row says otherwise — for every model below where `M_R == MPMF proxy`
exactly, that distinction is moot (the inequality `M_R ≤ true_M_P ≤
MPMF-proxy` collapses to one number).

| Model | `M_R` | MPMF proxy | **True `M_P`** | `M_H` | Full SCALE-Sim run done? |
|---|---|---|---|---|---|
| **ResNet-50 (INT8)** | 2352.00 KB | 2352.00 KB (= `M_R`, degenerate) | same (forced equal) | 2352.00 KB | **Yes — Gurobi, all 3 schemes `Optimal`, 0 bytes, 95.6% DRAM reduction, 14.4394× speedup.** Confirmed genuinely int8 (not a boundary-only quantization fallback) via direct tensor-dtype inspection. |
| **DenseNet-121 (INT8)** | 1596.25 KB | 2058.00 KB — real `M_R != MPMF` gap | 2058.00 KB (same value; only the fixed-schedule proxy tested, not a real free-schedule solve) | 1827.13 KB | **Yes, both ends of the range.** At `M_R`: all 3 schemes tie, 2,609,152 bytes, 97.19%, 17.2726×. At `MPMF`=2058KB: both baselines **fail outright** (`TfliteArenaAllocationError`, fragmentation — can't place tensors even with a theoretically-sufficient budget), COSMA `Optimal`, 0 bytes, 97.53%, 18.3223×. First confirmation (alongside ResNeXt-50 below) of the paper's §V-B.2 claim on a real non-degenerate model — and a *stronger* form of it than the paper's own text describes. |
| **ResNeXt-50** (sourced 2026-09-30 — needed real exporter fixes: grouped-conv axis inference, `PADV2` typemap entry; see `paper_model_roster.md`) | 9408.00 KB | 9636.00 KB — real `M_R != MPMF` gap | 9636.00 KB (fixed-schedule proxy only) | 9522.00 KB | **Yes, both ends.** At `M_R`: both baselines **fail outright** (fragmentation), COSMA `Optimal`, 3,426,564 bytes, 85.58%, 2.8625×. At `MPMF`=9636KB: same failure pattern — both baselines still fail outright even with the theoretically-sufficient budget, COSMA `Optimal`, 0 bytes, 87.32%, 3.0019×. The cleanest, most paper-consistent result in this project so far. |
| **S3D** (sourced 2026-09-30 — needed new `MAXPOOL_3D`/`AVGPOOL_3D` custom-op dispatch, TFLite has no native 3D pooling builtin; reused CONV_3D infra from R2Plus1D-18 unchanged) | 119168.00 KB | 119168.00 KB (= `M_R`, degenerate) | same | 119168.00 KB | **Yes — Gurobi, all 3 schemes `Optimal`, 0 bytes, 98.7% DRAM reduction, 3.99× speedup.** Degenerate case (no interesting spill/retrieve range at this input scale), but confirms the full CONV_3D + new pooling pipeline runs cleanly end-to-end. |
| **FCN** (sourced 2026-09-30 — needed new `RESIZE_BILINEAR` typemap entry) | 18816.00 KB | 18816.00 KB (= `M_R`, degenerate) | same | 18816.00 KB | **No.** Export/bounds only — this is the paper's own one documented exception where all 4 baselines are claimed to tie COSMA at `M_P` (§V-B.2), and we still haven't actually checked that claim on this model. SCALE-Sim couldn't complete a real simulation on this 7GB machine (dilated ResNet-50 backbone keeps high spatial resolution through `layer3`/`layer4`; killed as a precaution after ~1040s, available memory down to ~400MB). Needs a more powerful machine. |
| **DeepLabV3** (export unblocked 2026-09-30 by the same `RESIZE_BILINEAR` fix as FCN) | 39522.38 KB | 39522.38 KB (= `M_R`, degenerate) | same | 39522.38 KB | **No**, same SCALE-Sim memory wall as FCN (killed after ~240s, ~880MB available). Also: this `.tflite` was sourced externally and its real input shape is `[1,513,513,3]`, not the paper's `(1,3,224,224)` — even with a successful run, numbers here would not be directly comparable to the paper's own DeepLabV3 bars. |

| `toy_spill_model.json` | 200 B | 210 B | **200 B (= M_R)** | 200 B | N/A — no real conv params |
| `toy_branching_model.json` | 144.00 KB | 208.00 KB | **144.00 KB (= M_R)** | 144.00 KB | N/A — no real conv params |
| MobileNetV2-CIFAR10 | 60.00 KB | 60.00 KB | 60.00 KB (squeeze-forced equal) | 60.00 KB | Yes — 64KB: 32.5% DRAM reduction, 1.0016× speedup |
| ResNet-20-CIFAR10 | 192.00 KB | 192.00 KB | 192.00 KB (squeeze-forced equal) | 192.00 KB | Yes — 256KB: 91.1%, 1.0299× |
| SqueezeNet-small-CIFAR100 | 80.00 KB | 80.00 KB | 80.00 KB (squeeze-forced equal) | 80.00 KB | Yes — 96KB: 92.0%, 1.3599× |
| Inception-V3 | 8103.38 KB | 8103.38 KB | not yet run — expect a real solve-time jump (§6) | — | No — never run to completion at a real budget |
| ResNet-50 | 9408.00 KB | 9408.00 KB | **9408.00 KB (= M_R, Gurobi, Optimal in 0.40s)** | 9408.00 KB | **Yes — 9408KB (=M_R=M_P), Gurobi, 2026-09-29: 0 non-compulsory bytes, 95.8% DRAM reduction, 9.5417× speedup.** Supersedes the earlier "80MB"/unclear-unit partial note below. |
| DenseNet-121 (full ImageNet-scale) | 6328.25 KB | 8232.00 KB — **first real `M_R != MPMF-proxy` gap** | attempted 2026-09-29 at 7280KB (≈`M_H`) with Gurobi, 600s time limit: hit the limit at 100% gap (`Not Solved`, feasible incumbent accepted, 0 `SpmAllocator` violations) — 1439802 rows/768306 columns/19.7M nonzeros before presolve, 93s presolve alone. Real `M_P` still unknown; CBC never got this far at all (47+ min, killed, item 29) | 7280.13 KB | **Yes (first-ever, at the accepted feasible-not-optimal incumbent) — 7280KB, Gurobi, 2026-09-29: 16,827,776 non-compulsory bytes (real spill/retrieve fired for the first time at production scale), 94.6% DRAM reduction, 8.8343× speedup.** Pushed the machine into ~1.7GB swap running alone — do not run this alongside another large model's solve at the same time (confirmed 2026-09-29 after a parallel ResNet-50+DenseNet-121 run crashed the machine; inconclusive which one, or the combination, was the actual cause, but this machine has only 7GB RAM). **Re-attempted 2026-09-30 at the same 7280KB after an unrelated Eq.9 fix (300s limit): still `Not Solved`, root LP bound stuck at exactly 0.0 the whole time — see `results_plan.md` §6 item 7 / `ITERATION_HISTORY.md` item 34. Confirmed inherent to Eq.10's formulation at this graph's scale, not fixable by this change.** |
| Small custom DenseNet (18 conv layers, 2 blocks × 4 units, growth rate 12, 32×32 input — this session's own fixture, random weights) | 512.00 KB | 608.00 KB | **608.00 KB (no improvement over the fixed schedule — free scheduling confirmed today's arbitrary op order was already optimal for peak footprint here)** | 560.00 KB | Yes — 550KB: 84.2% DRAM reduction, **0.9985× speedup (net slower)** — genuine spill/retrieve tradeoff cost, first real (non-toy) demonstration of a retrieve becoming a *new* bottleneck |
| **R2Plus1D-18** (paper's own video model — 2026-09-29, new CONV_3D engine support, torchvision `r2plus1d_18`, random weights) | ~~28224.00 KB~~ **225792.00 KB (corrected 2026-09-30)** | ~~34496.00 KB~~ **275968.00 KB (corrected)** — real `M_R != MPMF-proxy` gap | not yet run | — | **SUPERSEDED, do not use**: the `25,690,112 non-compulsory bytes / 92.7% / 3.2912×` number this row used to report was measured on a model exported at `[1,8,112,112,3]` — half the frames, half the resolution of the paper's stated `(1,3,16,224,224)` input (Fig.3 caption). Found and fixed 2026-09-30 (input-shape audit, `paper_model_roster.md`). Re-exported at the correct `[1,16,224,224,3]`; bounds above are the real, corrected ones (8× the old byte scale, matching 2×frames×4×spatial exactly). **No real SCALE-Sim result exists at the correct resolution** — both the full 4-baseline comparison and the lighter single-pass `run_cosma.py` were attempted and killed as memory-safety precautions after 40-50 min each (available memory down to ~900MB/~470MB), same SCALE-Sim performance wall as FCN/DeepLabV3 below. A more powerful machine is the real next step here, not more code. |

**A real, informative split emerged**: on every model where the fixed-
schedule `MPMF` proxy already equalled `M_R` (all 5 originally-tested real
models, plus both toy fixtures), `true_M_P` is *mathematically forced* to
equal `M_R` too (by `M_R ≤ true_M_P ≤ MPMF-proxy`) — confirmed by an actual
solve, not just the inequality, on every one of them. But the two toy
fixtures (the only models with a real `M_R != MPMF-proxy` gap that have
actually been solved) split differently: `toy_spill_model.json`/
`toy_branching_model.json` both improved all the way down to `true_M_P ==
M_R` (free scheduling found a real, better order), while the small custom
DenseNet fixture did **not** improve at all (`true_M_P == MPMF-proxy` — the
existing fixed order was already schedule-optimal for peak footprint, even
though it isn't for spill/retrieve minimization, which is a genuinely
different objective). Whether DenseNet-121 itself would improve is unknown
— not yet run (see above).

**Main-pipeline free scheduling** (`free_schedule=True`, distinct from the
`true_M_P` column above, which only ever optimizes peak footprint): on
ResNet-20-CIFAR10, an actual `free_schedule=True` solve (real SCALE-Sim
re-run, 240s CBC limit) reproduced the fixed schedule's numbers exactly (0
non-compulsory bytes, 78.3% reduction, 1.0000x, both) — no improvement, same
pattern as `true_M_P` on this architecture class. On the small custom
DenseNet fixture, free scheduling found *no* reduction in total
non-compulsory bytes either (425984 both) — but did find a real, different
schedule that avoids landing a retrieve on layer 16's `CONCAT` (free in
baseline, a new bottleneck under the fixed COSMA schedule), improving real
simulated cycles 61458 → 58130 at the same budget. This is a genuinely new
kind of result: scheduling freedom can matter for real performance even
when it doesn't move the paper's own primary metric at all. Both real runs
had 0 `SpmAllocator` violations. A deliberately adversarial branching toy
fixture (`toy_branching_model.json`) confirmed the expected complexity
ceiling: the ASAP/ALAP pair-filter prunes 0% of pairs there (vs. ~92% for
the DenseNet fixture), and the solve didn't reach `Optimal` within ~9
minutes of continuous CBC time. See `ITERATION_HISTORY.md` item 32 for the
full detail.

**Overlap with the paper's own model list**: `ResNet-50` and `DenseNet` are
both literally in the paper's own 10 human-designed models (§V-A1) — not
approximations. Our small custom DenseNet is a fast-solving *stand-in* for
the same architectural family, not a literal replication of the paper's own
(unspecified-size) DenseNet run. We have no analog for the paper's 4
NAS-generated models (PNASNet-5, AmoebaNet-D, NASNet-A, DARTS) — no
divide-and-conquer heuristic exists here to make those tractable, per the
standing scope decision.

## 5. What we can run today, faithfully

For any model:
```bash
# 1. Get M_R, the fast MPMF proxy, AND the real M_P/M_H (real ILP solve,
#    not instant -- can be slow on a large model, see §6/§4's DenseNet-121
#    note)
PYTHONPATH=..:. python3 visualize_spm.py --model-json <model> \
    --bounds-only --true-mpmf

# 2. Run at M_R, true M_P, M_H (and the old fixed-schedule proxy too, if it
#    differs from true M_P -- worth comparing directly when it does)
PYTHONPATH=..:. python3 run_experiments.py --models <model> \
    --budgets-kb <M_R> <M_H> <true_M_P> --config <config>
```
Report `total_non_compulsory_access_bytes` (paper-comparable primary metric)
alongside our own `dram_traffic_reduction_pct`/`speedup` (broader, real-
SCALE-Sim-simulated, not the same measurement as the paper's numbers, but
genuinely useful in its own right). All three of `M_R`/`M_H`/`M_P` are now
real, reportable numbers — no more caveats needed on that front. If
`--true-mpmf` is skipped (e.g. for speed on a large model), say explicitly
that only the fixed-schedule proxy was used, not real `M_P`.

## 6. What blocks full comparability, ranked

1. **`M_P`/`M_H` at production scale via the real free-schedule solve
   (`--true-mpmf`)** — implemented and verified correct (§4), but still
   not run on any ImageNet-scale model (Inception-V3/ResNet-50/
   DenseNet-121/ResNeXt-50/etc.), since `C[a,t]` becoming a full `|T|x|A|`
   binary block is exactly the paper's own `O(|T|x|A|^2)` worst case
   (§III-F) — expect the same kind of solve-time jump already seen for the
   main pipeline on DenseNet-121 (item 29: 47+ min of CBC time, killed
   before finishing). **Distinct from, and not fixed by, the 2026-09-30
   `M_P`-budget tests** (ResNeXt-50/DenseNet-121-INT8 in §4 above) — those
   tested *at* the cheap fixed-schedule MPMF-proxy value, not via a real
   `--true-mpmf` solve. The MPMF-proxy is a valid upper bound on true
   `M_P` (so those results are still real and correct), but a genuinely
   *tighter* true `M_P` could exist and hasn't been solved for on any
   ImageNet-scale model.
2. **Activation+parameter tracking** — corrected against the primary source
   (see the mapping table above): the paper's own ILP needs no special
   handling for weights, it just includes them in `A`. That means this
   item isn't really an ILP-design question at all, it's a
   *SCALE-Sim-simulation-accounting* question, specific to this port: an
   activation's `'C'` (create) event maps naturally onto a real, already-
   free SCALE-Sim event (computing a layer's own ofmap has no DRAM cost
   in the engine, full stop) — but a weight tensor's `'C'` has no
   equivalent free event to map onto, since a weight is never "computed
   on-chip," it always needs at least one real DRAM fetch. The paper's own
   ILP sidesteps this cleanly by never charging *any* `'C'` event in its
   objective (Eq.12 only ever sums `S`/`R` terms) — compulsory cost, for
   any tensor, activation or weight, is defined as outside the optimized
   quantity entirely, so the ILP itself never needs to know weight-`'C'`
   is "different." Simulating that same plan for real, though, requires
   this port to decide how `baseline.run_cosma_aware()` charges a weight's
   first, unavoidable fetch — the paper's own analytical accounting never
   has to answer that, since it has no engine underneath it at all. Worked
   out carefully before trusting any resulting numbers.
3. **Comparison baselines** (TFLite linear allocator × Belady/greedy) — done
   and verified (see the mapping table above / `baseline_construction.md`).
   **Now run on real ImageNet-scale paper-listed models, 2026-09-30**:
   ResNeXt-50, DenseNet-121 (both precisions), S3D — see §4. Both
   baselines fail outright (fragmentation) on ResNeXt-50/DenseNet-121 at
   both `M_R` and `M_P`; a real, resolved item, no longer open. Still not
   run on Inception-V3, FCN, DeepLabV3 (blocked by the SCALE-Sim memory
   wall, item 6 below), or R2Plus1D-18 (same wall, at its correct
   resolution).
4. **Full operator rescheduling for the main (spill/retrieve) pipeline** —
   now implemented (`build_cosma_model(..., free_schedule=True)`, opt-in,
   default off), and re-simulated for real via `baseline.run_cosma_aware()`'s
   `schedule` param — not just an isolated bound anymore. What's left:
   only exercised so far on both toy fixtures, the small custom DenseNet
   fixture, and ResNet-20-CIFAR10 (see §4/§7) — Inception-V3/ResNet-50/
   DenseNet-121 haven't been attempted under `free_schedule=True` yet, and
   a deliberately adversarial branching toy fixture showed this can hit
   real, substantial solve-time cost when the ASAP/ALAP pair-filter can't
   prune much (0% pruning on that fixture vs. ~92% on the DenseNet one).
   Divide-and-conquer for NAS-scale graphs remains out of scope, per
   standing project direction.
5. **Solver/hardware** (PuLP/CBC vs. Gurobi, different machine) — acknowledged
   fixed difference, not something to chase.
6. **Datatype (FP32 vs. the paper's 8-bit)** — closed for 2 of 3 model
   families, 2026-09-30. `cosma/tools/quantize_model.py` (new) does real
   full-integer PTQ; ResNet-50 and DenseNet-121 both have verified INT8
   exports (genuinely int8 throughout, not a boundary-only fallback — see
   §4). R2Plus1D-18's `CONV_3D` has no real INT8 kernel in the standard
   `onnx2tf`/`TFLiteConverter` toolchain (confirmed via tensor-dtype
   inspection, not just "it errored") — stays FP32, a real, documented gap
   for that one model family. ResNeXt-50/S3D/FCN/DeepLabV3 not attempted
   in INT8 yet.
7. **Eq.9 formulation bug in `cosma_Ilp.py`, found and fixed 2026-09-30,
   did not resolve DenseNet-121's solve difficulty** — the paper states
   Eq.9 unconditionally (`L[a,t]+Size(a)<=MB`, no residency gate); this
   port had added one, making the constraint vacuous whenever a tensor
   isn't resident. Fixed (folded into `L`'s own variable bound instead of
   a separate row — verified byte-identical on 2 regression fixtures
   before/after). A short diagnostic re-solve of DenseNet-121 @ `M_H`
   afterward showed the root LP relaxation bound still sits at exactly
   `0.0` — this fix was correct and worth keeping (paper-fidelity, fewer
   rows) but did not touch the actual cause, which is Eq.10's own big-M
   pairwise non-overlap encoding (already verified to match the paper's
   Eq.10 exactly) being inherently weak-bounding at this graph's scale —
   a property of the paper's own formulation on a 311-tensor graph with
   liveness spans up to 121 timesteps, not a bug in this port. DenseNet-121
   @ `M_H` remains `Not Solved` (COSMA's own incumbent worse than
   ILP-greedy's result there) and is not expected to improve without a
   genuinely different formulation.
8. **SCALE-Sim memory/performance wall on large-tensor models, confirmed
   2026-09-30, blocking 3 models' full pipeline runs** — FCN, DeepLabV3,
   and R2Plus1D-18 (at its correct, paper-matching resolution) have all
   had a real SCALE-Sim simulation attempt killed as a memory-safety
   precaution on this 7GB machine (available memory dropping to
   400MB-900MB after 40+ minutes each, no result). Confirmed independent
   of SCALE-Sim pass count (R2Plus1D-18 hit the same wall via both the
   5-pass `run_paper_baselines.py` comparison and the 1-pass
   `run_cosma.py` alone) — it's each model's own per-layer tensor size
   (FCN/DeepLabV3's dilated-backbone high spatial resolution; R2Plus1D-18's
   full 16-frame×224×224 video volume), not the number of simulation
   passes. Not yet root-caused (is this a fundamental cost of
   cycle-accurate simulation at this array size, or a real inefficiency
   in `single_layer_sim.py`/`baseline.py`'s per-layer loop that could be
   fixed once and unblock all three?) — genuinely unknown, not
   investigated. `run_paper_roster.py`'s roster now includes all three
   with this noted, intended to be run on a more powerful machine (see
   its own module docstring's "Portability" section).

## 7. Recommended next test to run

**(Superseded 2026-09-30 — DenseNet-121 has since been run to completion
multiple times, at `M_H`, both before and after the Eq.9 fix; it remains
`Not Solved` with COSMA's incumbent worse than ILP-greedy, and this is now
understood to be an inherent Eq.10 formulation-hardness issue, not
something more solve time or more code fixes — see §6 item 7. Re-running
DenseNet-121 again is not the recommended next step any more.)**

Current recommendation, in order:

1. **Get the 3 blocked models a real result on a more powerful machine** —
   FCN (the paper's own documented M_P-tie exception, never actually
   checked), DeepLabV3 (export now works, just needs the compute), and
   R2Plus1D-18 at its correct resolution (bounds known, SCALE-Sim just
   needs to actually finish). `run_paper_roster.py`'s roster already has
   all three with portability notes — see its own docstring.
2. **`free_schedule=True` on ResNeXt-50 or DenseNet-121 at a real,
   non-degenerate budget** (e.g. ResNeXt-50 @ `M_R`=9408KB, where fixed-
   schedule COSMA already succeeds cleanly and fast) — never tried on any
   model with a genuine `M_R != MPMF` gap where it could plausibly help;
   every prior free-schedule test either had no gap at all (ResNet-20,
   ResNet-50) or didn't improve on it (small DenseNet fixture). DenseNet-121
   itself is a poor choice for this specifically because its *fixed*-
   schedule solve already can't reach optimality (item 7) — free scheduling
   would only add the full `|T|x|A|` `C` block on top of an already-hard
   problem, confounding "did scheduling help" with "is this now unsolvable
   for a different reason."
3. **Investigate the SCALE-Sim memory wall directly** (§6 item 8) — would
   unblock recommendation 1 without needing a different machine at all, if
   it turns out to be a real fixable inefficiency rather than a fundamental
   cost of cycle-accurate simulation at this scale. Not yet looked into.

# COSMA vs. the paper (arXiv:2311.18246) — results comparison

Source data: `run_paper_baselines.py` output, 2026-10-05 run (3 combos: `default+belady`,
`default+ilp_greedy`, `cosma_native`), on the models in `cosma/docs/paper_model_roster.md` that
have completed a full pipeline run. DenseNet (both FP32 and the INT8 `DenseNet-121` export) is
excluded from this file — `cosma_native` hit the solver time limit (`Not Solved`) on the FP32
export and needs a re-run with a higher time limit before it's trustworthy; see git history /
chat log if it's needed again later.

## Why there's no "% of paper" column

Unlike OnSRAM's paper (which prints exact per-model numbers in Fig. 7 / Table 1), the COSMA
paper **never publishes a per-model numeric table**. Its Fig. 3/4 results are bar charts only,
and the only two things stated as exact numbers in the text are:

1. "given a memory budget of MPMF (`M_P`), COSMA can always eliminate the non-compulsory
   off-chip data accesses ... [but] all other schemes incur non-compulsory data accesses
   **except for the model FCN**" (§V-B.2)
2. "under the tightest memory limit of `M_R`, COSMA can reduce on average **84%** of the
   non-compulsory data accesses for these ten DNNs" (§V-B.2)

So the honest comparison is against these two structural claims, model by model — not a
manufactured "% match" number.

## What this comparison actually validates (and what it doesn't)

Every row in table 1 falls into one of three buckets, and **none of them is the paper's actual
headline pattern**:

- **Degenerate ties (0 bytes, all three schemes)** — ResNet-50 (FP32+INT8), DeepLabV3, FCN,
  R2Plus1D-18 at `M_P`. There's no slack between `M_R` and `M_P` for these models, so there's
  nothing for any scheme to avoid in the first place. This isn't a comparison, it's an identity.
- **Baseline outright failure** — ResNeXt-50, R2Plus1D-18 at `M_P`. The TF-Lite-style linear
  allocator can't place tensors at all (fragmentation), so it's "baseline fails / COSMA
  succeeds," not a quantitative gap between two working schemes.
- **Real three-way tie, nonzero bytes** — R2Plus1D-18 at `M_R`/`M_H`. All three schemes land on
  the exact same byte count, so again there's no COSMA-vs-baseline gap to show.

What's **missing** is the paper's actual main result: a baseline that *places tensors
successfully but with real, nonzero extra spill*, with COSMA's joint ILP beating it while
itself staying nonzero — that's what most of Fig. 3's bars actually show (COSMA shorter, not
zero, vs. a baseline that's zero or failed outright). DenseNet (FP32) was the one model in this
roster that showed exactly that pattern for `default+ilp_greedy` (5.62 MB → 2.41 MB → 0 B as the
budget widened across `M_R`/`M_H`/`M_P`) — it's excluded here because `cosma_native` itself hit
the solver time limit on that model and never produced a provable result to compare against.

**Net: this file confirms two structural properties of COSMA's own ILP** (the `M_P`=0 guarantee,
and the FCN exception) — **it does not yet confirm the paper's core comparative claim** that
COSMA meaningfully beats a baseline that is itself still functioning. Getting a real test of that
claim needs either (a) fixing the DenseNet FP32 solver timeout so `cosma_native` produces a
provable result there, or (b) finding/adding more models with a genuine `M_R`-to-`M_P` range
where the baselines still place tensors (rather than failing outright) at the tighter budgets.

## 1. Non-compulsory off-chip bytes, by scheme and budget tier

`M_R` = minimum memory requirement (tightest budget). `M_H` = `(M_R+M_P)/2`. `M_P` = minimum
peak memory footprint. Several models have `M_R == M_P` (no gap to show); those rows are marked
"degenerate."

| Model (precision) | Budget tier (KB) | Default+Belady | Default+ILP-Greedy | COSMA (native ILP) |
|---|---|---|---|---|
| ResNet-50 (FP32) | M_R=M_P, degenerate (9408.0) | 0 B | 0 B | 0 B |
| ResNet-50 (INT8) | M_R=M_P, degenerate (2352.0) | 0 B | 0 B | 0 B |
| DeepLabV3 (FP32) | M_R=M_P, degenerate (39522.375) | 0 B | 0 B | 0 B |
| ResNeXt-50 (FP32) | ⚠️ labeled 9408.0 — see caveat below | FAILS (fragmentation) | FAILS (fragmentation) | 0 B |
| R2Plus1D-18 (FP32) | M_R (225792.0) | 205,520,896 B (205.52 MB) | 205,520,896 B (205.52 MB) | 205,520,896 B (205.52 MB) — tie |
| R2Plus1D-18 (FP32) | M_H (250880.0) | 205,520,896 B (205.52 MB) | 205,520,896 B (205.52 MB) | 205,520,896 B (205.52 MB) — tie, unchanged from M_R |
| R2Plus1D-18 (FP32) | M_P (275968.0) | FAILS (fragmentation) | FAILS (fragmentation) | 0 B |
| FCN (FP32) | M_R=M_P, degenerate (18816.0) | 0 B | 0 B | 0 B |

"FAILS (fragmentation)" = `TfliteArenaAllocationError` — the TFLite-style linear allocator
could not place a tensor at all, not just suboptimally. This is a *stronger* result than the
paper's own text, which only describes baselines as "incurring non-compulsory data accesses"
(implying they still place tensors, just worse).

## 2. Checking the paper's two text claims against this roster

| Paper claim | What we see |
|---|---|
| COSMA always eliminates non-compulsory accesses at `M_P` | **Holds on every model tested at `M_P`**: ResNet-50 (FP32 + INT8), DeepLabV3, ResNeXt-50, R2Plus1D-18, FCN all hit 0 bytes with `cosma_native`. |
| All other schemes incur non-compulsory accesses **except FCN** | **FCN tie confirmed exactly** — all three schemes land on 0 bytes, a direct hit on the paper's one named exception. |
| ~84% average non-compulsory reduction at `M_R` | Not reducible to one clean number from this roster: most of the models here have `M_R == M_P` (degenerate — no spill to begin with, so no reduction to measure), and the one model with a real `M_R` gap (R2Plus1D-18) ties exactly with both baselines at `M_R` (0% relative edge, though COSMA is the one *provably* optimal). ResNeXt-50's `M_R` result is unresolved (see caveat). |

## 3. Headline SCALE-Sim numbers (this project's own metric — not paper-comparable)

`dram_traffic_reduction_pct` / `speedup` compare the winning plan's total DRAM traffic / cycle
count against a synthetic "zero on-chip reuse" baseline (every tensor access goes to DRAM). This
is a different denominator than the paper's "non-compulsory accesses" metric above — don't quote
these two sections against each other as if they measure the same thing.

| Model (precision) | Budget tier (KB) | DRAM traffic reduction vs. zero-reuse | Speedup vs. zero-reuse |
|---|---|---|---|
| ResNet-50 (FP32) | 9408.0 | 95.82% | 9.54× |
| ResNet-50 (INT8) | 2352.0 | 95.62% | 14.44× |
| DeepLabV3 (FP32) | 39522.375 | 99.83% | 12.24× |
| ResNeXt-50 (FP32) | ⚠️ 9408.0 | 87.32% | 3.00× |
| R2Plus1D-18 (FP32) | 225792.0 (M_R) | 96.03% | 3.03× |
| R2Plus1D-18 (FP32) | 250880.0 (M_H) | 96.03% | 3.03× |
| R2Plus1D-18 (FP32) | 275968.0 (M_P) | 99.44% | 3.37× |
| FCN (FP32) | 18816.0 | 99.19% | 14.02× |

## 4. Caveats — verify before citing

- **ResNeXt-50's row is logged at `budget_kb=9408.0`**, but its numbers (0 bytes, 87.32%
  reduction, 3.0019× speedup) exactly match an earlier-recorded result **at `MPMF=9636KB`**, not
  at `M_R=9408KB` (which earlier showed a nonzero 3,426,564 bytes, 85.58%, 2.8625×). Either the
  budget column is mislabeled in this run, or the `M_R` result has genuinely changed since that
  earlier check. Re-run and confirm the budget before citing a specific number for this model.
- R2Plus1D-18's `M_R` and `M_H` rows are byte-identical across all three schemes (205,520,896 B,
  unchanged by widening the budget). Plausible for a model with very large per-layer CONV_3D
  tensors and a fixed default schedule (no freedom to avoid the same forced spill until `M_P`),
  but not independently re-derived by hand — worth a sanity check if this number goes in the
  paper.
- These are TFLite exports through this project's own PyTorch→ONNX→`onnx2tf`→`trim` pipeline,
  not the paper's original TensorFlow graphs, and compute cycles come from SCALE-Sim's systolic
  array rather than the paper's own hardware model — expect the same qualitative pattern, not
  identical numbers, per `cosma/docs/paper_model_roster.md` §4.

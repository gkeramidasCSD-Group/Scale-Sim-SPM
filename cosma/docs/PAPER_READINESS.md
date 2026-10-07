# What's left for the COSMA results (short version)

**Activation-only baseline-fidelity investigation: closed out,
inconclusive in that setting, recommend not chasing further there
(2026-10-07, see `ITERATION_HISTORY.md` #39-40).** In the activation-only
setting, our TFLite-style baseline allocator ties at exactly 0 bytes for
every scheme on ResNet-50/S3D/DeepLabV3/FCN, while the paper's Fig.3
shows these same models' baselines staying substantially nonzero even at
`M_P`. Four distinct hypotheses tested and **all eliminated**: missing
64-byte TFLite alignment, BatchNorm/Bias/ReLU fusion, a mis-ported TFLite
placement algorithm (re-verified against current real TFLite source), and
export-toolchain tensor granularity (tested a genuinely TF-native export,
byte-identical result to the existing QAIRT-sourced one). Confirmed the
zero gap is mathematically forced by these models' own low concurrency
(2-7 tensors live at once) in the activation-only setting specifically —
proven by contrast against a toy fixture with a real `M_R != MPMF` gap,
which immediately shows a real spill under the identical code. No public
code artifact exists for this paper to resolve the remaining gap against
ground truth in that setting.

**But the parameter-inclusive setting (`M_Rp`/`M_Hp`/`M_Pp` — the paper's
own second standard setting, Fig.3) is a different lever, and it worked
(2026-10-07, `ITERATION_HISTORY.md` #41).** `spm_common/graph_builder
.load_graph(..., include_parameters=True)` (new) tracks weight tensors
too. Result: ResNet-50 (both precisions) gain a real, non-degenerate
`M_Rp`-to-`MPMF_p` gap (784.00KB / 98.00KB) — S3D and DeepLabV3 stay
exactly degenerate even with parameters, confirmed not just untested.
ResNet-50 (INT8) is now a clean, citable "COSMA wins" result: baselines
fail outright at both tiers, COSMA solves `Optimal` both times (602,112
bytes at `M_Rp`, 0 at `MPMF_p`). Getting a *trustworthy* number out of
this surfaced and fixed three real bugs, all regression-checked against
every prior activation-only result (byte-identical after each fix):
`compute_structural_minimum_bytes()` under-counted (missing
`weight_inputs` in a node's own floor — this is what produced an initial,
bogus 25,890KB "gap" for FCN, since corrected to exactly 0, same as
S3D/DeepLabV3); Eq.5 needed a separate, looser rule for weight inputs
(`Cv(a,t) <= Cv(b,t)+P[b,t]+R[b,t]`, not the activation-input rule, to
avoid an unsatisfiable self-contradiction); and a genuine Gurobi numerical
fragility in Eq.10's big-M formulation (a 32-byte weight tensor was small
enough that the *default* `IntFeasTol` (~1e-5) induced a same-sized false
"gap," reported `Optimal` despite a real `SpmAllocator` collision — fixed
by tightening `IntFeasTol`/`FeasibilityTol` to `1e-9` in
`cosma_Ilp.solve()`, applied to every Gurobi solve). Next candidate:
extend the same `include_parameters=True` path to DenseNet/ResNeXt-50/
R2Plus1D-18/FCN's full baseline comparison (currently only re-verified
the bounds for those four, not the full replacement+allocator+ILP
pipeline).

Also worth fixing, found during the same investigation but not yet
linked to a specific discrepancy: `replacement_engine
.simulate_replacement()`'s eviction trigger is a pure byte-sum check,
blind to the allocator's real placement/fragmentation, with no feedback
loop back to replacement when placement fails.

**Fixed, 2026-10-07**: the Eq.10 pair-pruning correctness bug documented
below as item #38 (a tensor could stay "preserved" past its last real
use, invisible to the overlap constraints that assumed it never would) —
applied (`Eq10Sound_P`/`Eq10Sound_R` constraints in `build_cosma_model()`)
and regression-checked. DenseNet (FP32) no longer collides, though it
still doesn't reach a proven-optimal solution in a practical time limit
(a separate, known solve-hardness issue, see below) — the fix closed the
unsoundness, not the scale problem.


Goal, in your order: (1) make sure the ILP logic is actually correct, (2)
get real paper-comparable results, (3) justify why our numbers match the
paper's logic/claims.

**Read this first -- branch warning:** this file is being written while
on branch `OnSram`. All of this session's real COSMA work (the Eq.9 fix,
INT8 pipeline, `free_schedule` wiring, the 3 new sourced models, `cosma/
tools/`) lives on branch **`cosma2`** (local commit `8ab295d`), which was
never merged into `OnSram` and isn't even pushed to `origin/cosma2` yet
(`origin/cosma2` is one commit behind). This branch's `cosma/helpers/
cosma_Ilp.py` still has the old, pre-fix code. **Do all COSMA work on
`cosma2`, not here** -- this doc needs to end up there too (copy it over,
or ask for it to be rewritten once you're on that branch).

## 1. Make sure the logic is solid

- **Eq.9 (budget constraint)** -- already fixed and verified byte-
  identical on 2 known-good fixtures. Done.
- **Eq.13-17 (the free-schedule / MPMF equations)** -- these power
  `free_schedule=True`, wired in this session but **never given the same
  careful line-by-line reread against the paper PDF that Eq.9 got.** Do
  that before trusting any free-schedule result.
- **DenseNet-121's solve hardness** -- confirmed NOT caused by the Eq.9
  bug (diagnostic re-solve still showed a 0.0 LP bound after the fix).
  Traced to Eq.10's big-M pairwise constraint being inherently weak at
  this scale (311 tensors). Decide: accept "doesn't converge in
  reasonable time, honestly reported" as a real finding, or spend time
  tightening the formulation.
- **Activation+parameter tracking (M_Rp/M_Hp/M_Pp)** -- the paper reports
  both activation-only and activation+parameter settings; this port only
  does activation-only. Decide if your paper needs the parameter setting
  too, or if activation-only is explicitly scoped and footnoted.
- **DenseNet model identity** -- the paper's own citation for "DenseNet"
  points to a segmentation architecture (Tiramisu/FC-DenseNet), not the
  classification DenseNet this port actually uses. Unresolvable without
  the paper's own code; just needs an honest footnote in your paper.

## 2. Get real results

Run the full roster (`cosma/run_paper_roster.py --solver gurobi
--free-schedule`, sequentially, one model at a time, memory-monitored) for
all 9 currently-available models: ResNet-50 (+INT8), DenseNet-121
(+INT8), ResNeXt-50, R2Plus1D-18, S3D, FCN, DeepLabV3. Use `--free-
schedule` -- that's the paper's actual Fig. 3 "COSMA" setup for
human-designed models, not the weaker Fixed-Schedule sub-problem this
project ran by default until this session.

Known caveats to keep in mind while reading results, not blockers:
- DeepLabV3's input is 513x513, not the paper's 224x224 (externally
  sourced model) -- not directly comparable on that axis.
- R2Plus1D-18 was re-exported this session at the correct resolution
  (16x224x224) -- any older numbers from before are superseded.

## 3. Justify the match

Once results are in, the paper's own claim to check directly (§V-B.2):
COSMA reaches 0 non-compulsory bytes at M_P, baselines don't (except
FCN, the paper's own documented exception). Already confirmed on
ResNeXt-50 and DenseNet-121-INT8 locally, at their real (non-degenerate)
M_P budgets -- extend this check to the rest once the full sweep
completes, and write up the comparison against the paper's Fig. 3 bars
directly, noting the 2 caveats above where relevant.

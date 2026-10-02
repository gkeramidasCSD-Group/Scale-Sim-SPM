# What's left for the COSMA results (short version)

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

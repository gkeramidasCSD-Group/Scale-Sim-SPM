# Paper Model Roster: what's available to test against arXiv:2311.18246

Tracks which of the COSMA paper's own 14 evaluation models (§V-A1) we can
actually run, so results are checked against real availability, not
assumed. Checked directly against `trim/models/` and `cosma/_exported/` on
2026-09-10 — update this file as models get sourced/exported.

## 1. Paper's own roster (§V-A1)

10 human-designed DNNs (Figure 3) across 4 application domains, plus 4
NAS-generated DNNs (Figure 4, Table II) needing COSMA's divide-and-conquer
heuristic to solve at all.

| Model | Paper category | Available now? | Notes |
|---|---|---|---|
| **ResNet-50** | human-designed (image classification) | ✅ `_exported/_exported_resnet50-tflite-float/model.json` (auto-exported from `_exported/resnet50-tflite-float/resnet50.tflite`) | Ready to run today. **INT8 version also available, 2026-09-30**: `_exported/models_resnet50_int8_resnet50/model.json`, quantized via `cosma/tools/quantize_model.py` (full-integer PTQ, random-shape calibration) — matches the paper's own "all data are 8-bit" (§V-A2/Fig.3). Verified: input/output/weight tensors genuinely `int8` (not a boundary-only fallback), same 53 CONV2D + 16 ADD backbone as the FP32 export, tracked-tensor bytes ≈0.229× the FP32 total (a few PAD/SUB/MUL bookkeeping ops differ between the float and int8 TFLite graphs, hence not exactly 0.25×). Full `run_paper_baselines.py` pipeline confirmed clean (`Optimal` on all 3 combos) at `M_R`=2352KB |
| **DenseNet** | human-designed (image classification) | ✅ `_exported/densenet/model.json` (`tf.keras.applications.DenseNet121`) | **Open architecture-identity question, found 2026-09-30, not previously flagged this precisely**: the paper's own text lists models as "image classification (ResNet-50 [10], DenseNet [13], ResNeXt [25])" — but reference **[13] is Jégou et al., "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation" (CVPR 2017)**, not the original Huang et al. DenseNet classification paper. Tiramisu/FC-DenseNet is a fully-convolutional, U-Net-style encoder-decoder built from DenseNet's dense-block unit — a segmentation-only architecture with no classification head, structurally unrelated to `tf.keras.applications.DenseNet121`. Most likely a citation slip in the paper (DenseNet is grouped under "image classification" alongside ResNet-50/ResNeXt, while the paper already has a *separate* "semantic segmentation" category populated with FCN/L-RASPP/DeepLabV3 in the very same sentence — filing a segmentation-only architecture under classification would be an odd, inconsistent choice), but not certain from the text alone. Previously this row only said "paper doesn't specify the variant" (implying "some DenseNet classification size, just unclear which") — this is a stronger, different kind of ambiguity (possibly not even the same architecture family at all). Our `DenseNet-121` choice remains defensible (correctly-categorized, real classification DenseNet) but should not be treated as a settled 1:1 match. **INT8 version also available, 2026-09-30**: `_exported/models_densenet121_int8_densenet121/model.json`, same `quantize_model.py` pipeline. Verified: tracked-tensor bytes ratio 0.248× FP32 (very close to the ideal 0.25×), 311 vs. 312 layers (one bookkeeping op fused away by the int8 converter). `M_R`=1596.25KB, `MPMF`=2058.00KB — a **real, non-degenerate range** (corrected 2026-09-30; previously mislabeled degenerate in chat, not in this doc). Full pipeline confirmed clean (`Optimal` on all 3 combos) at `M_R`=1596.25KB: 2,609,152 bytes tied across all 3 schemes. **Also tested at `MPMF`=2058.00KB, 2026-09-30**: both baselines FAIL outright (`TfliteArenaAllocationError`, fragmentation) — same pattern as ResNeXt-50 at its own `MPMF` (see that row) — while COSMA hits the guaranteed 0 bytes, `Optimal`, 97.53% DRAM reduction, 18.3223× speedup. Second model confirming the same result: baselines can't even place tensors at `M_P`, not just spill more than COSMA |
| **DeepLabV3** | human-designed (semantic segmentation) | ⚠️ **export unblocked, full-resolution pipeline run not completed, AND input resolution does not match the paper** — `_exported/_exported_deeplabv3_DeepLabV3-Plus-MobileNet/model.json`. Confirmed 2026-09-30: `RESIZE_BILINEAR` support (added for FCN, same commit) resolves the original `builtin_code_23` blocker exactly as predicted — exports cleanly now, 98 layers (44 CONV2D, 17 DEPTHWISE_CONV2D, 19 PAD, 10 ADD, 3 RESIZE_BILINEAR, 2 CONCAT, SUB/MUL/REDUCE_MEAN), `M_R`=39522.38KB (degenerate, `M_R`==MPMF). **Same fate as FCN's full pipeline run**: the SCALE-Sim baseline pass alone pushed available memory down to ~880MB within ~240s before being killed as a precaution — this model's ASPP/decoder stage processes large spatial feature maps for the same structural reason FCN's dilated backbone does. Real SCALE-Sim performance limitation, not a correctness issue (export/bounds are solid). **Separately, input resolution audit (2026-09-30) found this model's actual input shape is `[1,513,513,3]`** (confirmed directly from `model.json`'s tensor 0) — not the paper's stated `(1,3,224,224)` for every non-video/non-Transformer model (Fig.3 caption). This `.tflite` was sourced externally (not exported by us from a controlled architecture+shape, unlike every other model in this table) and uses the standard DeepLabV3+ Pascal-VOC/Cityscapes convention (513×513), 5.3× more spatial positions than the paper's setting. Left as-is rather than re-exported at 224×224, since re-sourcing DeepLabV3 from torchvision (matching the FCN/ResNeXt/S3D pattern) would be needed for a real paper-matching comparison — not done this session; any future DeepLabV3 numbers from this exact artifact are **not** comparable to the paper's Fig.3 bars on an equal footing | Paper itself flags this one as slower to solve (6.96s/17.242s at `M_R`/`M_H`, with parameter tensors) |
| **ResNeXt** | human-designed (image classification) | ✅ sourced 2026-09-30 — `_exported/resnext50_resnext50_tflite_resnext50_float32/model.json` (torchvision `resnext50_32x4d`, random weights, PyTorch→ONNX→`onnx2tf`→`trim` exporter) | Needed two real exporter fixes, not just a re-export: (1) `exporter_core.infer_conv2d_axes()` couldn't infer axes for a grouped-conv weight tensor (TFLite's `CONV_2D` has no explicit "groups" field at all — the runtime infers grouping implicitly when the weight's own in-channel axis is smaller than the real input channel count; the exporter now does the same inference and threads `groups` through to `params`, matching what `topology_builder.py`'s `channels = in_shape[3] // groups` branch already expected); (2) `PADV2` (TFLite builtin 60) wasn't in the operator typemap at all, only plain `PAD`. Also hit and fixed an unrelated `onnx2tf` bug: its `download_test_image_data()` unconditionally fetches a calibration image from a GitHub release that returns 404 (upstream asset removed), and doesn't check the HTTP status before parsing the response as `.npy` — worked around by pre-placing a synthetic calibration `.npy` in the working directory so the (unused-by-us) download is skipped entirely. **Real, non-degenerate result at `M_R`=9408KB**: both baseline combos (`default+belady`, `default+ilp_greedy`) FAIL outright with a real `TfliteArenaAllocationError` (fragmentation) — COSMA (`cosma_native`) solves `Optimal`, 3,426,564 non-compulsory bytes, 85.58% DRAM reduction, 2.8625× speedup. Exactly the kind of case the paper's methodology is built to showcase. **Also tested at `MPMF`=9636KB, 2026-09-30**: both baselines still FAIL outright — same fragmentation error, now with a theoretically-sufficient total budget — while COSMA hits the guaranteed 0 bytes, `Optimal`, 87.32% DRAM reduction, 3.0019× speedup. Stronger than the paper's own described failure mode (paper's text says baselines "incur non-compulsory data accesses" at `M_P`, implying they can at least place tensors, just suboptimally — ours can't place them at all here). All 39 `trim` unit tests still pass after the fixes, including 2 new tests for the grouped-conv axis inference |
| **R2Plus1D-18** | human-designed (video classification) | ✅ export correct, ⚠️ **full 4-baseline comparison not completed at real scale, see note** — `_exported/r2plus1d_correct_r2plus1d_18_tflite_r2plus1d_18_float32/model.json` (torchvision `r2plus1d_18`, random weights) | **First video model to actually run, 2026-09-29.** Needed real new engine support: CONV_3D end-to-end through `trim/`'s exporter (`exporter_core.py`/`export_hooks.py`/`tflite_utils.py`) and a new 3D-conv-to-systolic-array mapping in `topology_builder.py`/`baseline.py` (temporal kernel folded into Channels, compute/ofmap scaled by T′, ifmap replaced with the real 5D tensor's element count — see `topology_builder.py`'s module docstring for the full derivation). Verified against first-principles hand-derivation on a real layer (id 3), not just "it ran" — see `cosma/docs/results_plan.md` §4. **INT8 attempted and blocked, 2026-09-30**: TFLite's `CONV_3D` builtin has no real INT8 kernel in the standard `onnx2tf`/`TFLiteConverter` toolchain — full-integer PTQ *completes without error* and produces a `.tflite` that LOOKS quantized, but only 82 of 230 tensors are actually `int8` (rest, including CONV_3D weights, stay `float32` — quantize/dequantize wrapping at the I/O boundary only); file size confirms independently (~0.5% smaller, not ~75%). Staying FP32. **Input-resolution bug found and fixed, 2026-09-30**: the `_exported/r2plus1d_18/model.json` used for every prior result in this row (92.7% DRAM reduction, 3.29× speedup, 25,690,112 bytes @ `M_R`=28224KB) actually had input shape `[1,8,112,112,3]` — half the frames, half the resolution of the paper's stated `(1,3,16,224,224)` (Fig.3 caption) — a stale artifact from an earlier session that never actually matched the paper, despite being documented as if it did. **Re-exported at the correct `[1,16,224,224,3]`** (fresh PyTorch→ONNX→`onnx2tf`→`trim` pipeline) — real, non-degenerate bounds confirmed: `M_R`=225792.00KB, `MPMF`=275968.00KB (8× the old model's byte scale, as expected: 2× frames × 4× spatial). **The old 25,690,112-byte/92.7%/3.29× numbers above are now superseded and should not be used for paper comparison** — they were measured on the wrong-resolution model. **Full 4-baseline comparison at the correct resolution attempted twice, both killed as a memory-safety precaution** — after 40 minutes, the SCALE-Sim baseline pass alone pushed available memory down to ~890MB (RSS 4.3GB) with no result yet, the same pattern now seen on FCN and DeepLabV3 (see their rows) — a real, now 3-for-3-confirmed SCALE-Sim performance/memory characteristic with large per-layer operand tensors on this machine (8× more data per layer than the old wrong-resolution export), not a correctness issue. **Also tried the lighter, single-SCALE-Sim-pass `run_cosma.py` directly (bypassing the 4-baseline script's extra passes) — killed the same way**, ~900MB available after ~50 min. This rules out "too many SCALE-Sim passes" as the cause: even one pass, at this model's real per-layer tensor size (16 frames × 224×224, CONV_3D's full 5D accounting), is enough to hit the wall. **No real, correctly-scaled R2Plus1D-18 number exists as of 2026-09-30** — export/bounds are solid, but every attempt at a real SCALE-Sim simulation at this resolution has been stopped as a memory-safety precaution. Would need either a smaller/representative test resolution, or a more powerful machine, to get a real result |
| **S3D** | human-designed (video classification) | ✅ sourced 2026-09-30 — `_exported/s3d_s3d_tflite_s3d_float32/model.json` (torchvision `s3d`, random weights) | Reused the R2Plus1D-18 CONV_3D infrastructure directly (no changes needed there), but needed one genuinely new addition: TFLite has no native `MAX_POOL_3D`/`AVERAGE_POOL_3D` builtin op at all (unlike `MAX_POOL_2D`/`AVERAGE_POOL_2D`), so `onnx2tf` falls back to the Flex (Select TF ops) delegate for S3D's 13 `MaxPool3d` + 1 `AvgPool3d` layers, emitted as `FlexMaxPool3D`/`FlexAvgPool3D` *custom* ops (not distinguishable by numeric `BuiltinCode()` alone — all TFLite CUSTOM ops share code 32; real identity is the string `CustomCode()`). Added a second, string-keyed dispatch table (`exporter_core.CUSTOM_OP_TYPEMAP`) checked only when `BuiltinCode() == CUSTOM`, mapping these two to new `MAXPOOL_3D`/`AVGPOOL_3D` layer types — deliberately given no weights, no required params (`enforce_complete_metadata()`/`infer_pool_kernel_if_missing()` are both 2D-only, exact-string, 4D-shape-assuming, so these new types pass straight through untouched), since COSMA's own non-conv costing (`_nonconv_layer_stats()`) is already shape-generic and op-name-agnostic. Exported cleanly: 185 layers (78 CONV_3D, 13 MAXPOOL_3D, 1 AVGPOOL_3D, rest RELU/CONCAT/PAD/ADD/REDUCE_MEAN). `M_R`=119168KB, degenerate (`M_R`==MPMF, same pattern as ResNet-50) — full pipeline confirmed clean (`Optimal` on all 3 combos, 98.7% DRAM reduction, 3.99× speedup) |
| **FCN** | human-designed (semantic segmentation) | ⚠️ **exported and structurally verified, full-resolution pipeline run not completed** — `_exported/fcn_fcn_tflite_fcn_float32/model.json` (torchvision `fcn_resnet50`, random weights) | The paper's own one documented exception where all 4 baselines tie with COSMA at `M_P` — worth sourcing specifically to check this, **not yet checked** (see below). Needed one real exporter fix: FCN's (and DeepLabV3's) decoder upsampling emits TFLite's `RESIZE_BILINEAR` builtin (code 23 — the exact op that separately blocks DeepLabV3, see its row), never in the operator typemap before. Added it (no weights, no required params — target H/W comes from a second constant input tensor, same treatment as `PAD`). Exported cleanly: 76 layers (55 CONV2D, 16 ADD, 3 PAD, 1 MAXPOOL, 1 RESIZE_BILINEAR), `M_R`=18816KB (degenerate, `M_R`==MPMF). **Stopped before completing a full `run_paper_baselines.py` run**: unlike ResNet-50/ResNeXt-50/S3D, FCN's backbone keeps dilated (atrous) convolutions in `layer3`/`layer4` instead of downsampling for dense per-pixel prediction, so its later bottleneck blocks run at 28×28 instead of the usual 7×7 — SCALE-Sim's cycle-accurate per-layer simulation cost scales with operand-matrix size (already documented as genuinely slow on plain full-resolution ResNet-50, see `ITERATION_HISTORY.md` item 11), and FCN's dilated layers make this materially worse. The baseline SCALE-Sim pass alone ran past 1040s with available memory dropping from ~5GB to ~400MB (swap climbing to 3.6GB) before being killed as a precaution — a real characteristic of SCALE-Sim itself at this resolution, not a COSMA/exporter bug (mirrors item 11's own conclusion). A smaller input resolution, or more patience on a more powerful machine, would likely get a real result; not attempted further this session |
| L-RASPP | human-designed (semantic segmentation) | ❌ not sourced | Would need exporting first |
| Transformer | human-designed (transformer-based) | ❌ not sourced | Would need exporting first |
| ViT | human-designed (transformer-based) | ❌ not sourced | Would need exporting first |
| PNASNet-5 | NAS-generated | ⛔ permanently out of scope | Needs the divide-and-conquer heuristic — standing project decision to never build this |
| AmoebaNet-D | NAS-generated | ⛔ permanently out of scope | Same |
| NASNet-A | NAS-generated | ⛔ permanently out of scope | Same |
| DARTS | NAS-generated | ⛔ permanently out of scope | Same |

## 2. Summary

**6 of 14 are ready to run right now**: ResNet-50, DenseNet-121, R2Plus1D-18,
ResNeXt, S3D, and FCN (FCN sourced/structurally verified; full-resolution
pipeline run still pending, see its row) — directly matching 6 of the
paper's own 10 human-designed models. Updated 2026-09-30: ResNeXt and S3D
went from not-sourced to fully working the same day, needing real new
exporter engineering each time (grouped-conv axis inference + `PADV2` for
ResNeXt; `FlexMaxPool3D`/`FlexAvgPool3D` custom-op support for S3D) — see
their rows for details. The remaining 4 human-designed models (DeepLabV3,
L-RASPP, Transformer, ViT) would need sourcing/exporting; DeepLabV3's
previously-documented blocker is now very likely resolved as a side effect
of FCN's `RESIZE_BILINEAR` fix, just not re-tested yet. The 4 NAS-generated
models are off the table regardless of availability, per the standing
decision to skip the divide-and-conquer heuristic (see `results_plan.md`).

**2026-09-30 datatype-parity update**: the paper's own evaluation uses 8-bit
tensors exclusively (Fig.3 caption) — every model here had been FP32 until
today (4× the paper's byte sizes). ResNet-50 and DenseNet-121 now have real,
verified INT8 exports (see their rows above) via the new
`cosma/tools/quantize_model.py` PTQ script. R2Plus1D-18 stays FP32 — its
CONV_3D op has no real INT8 kernel in the standard TFLite toolchain, a
genuine gap, not a shortcut (see its row for the full investigation).

## 3. How to check numbers against the paper for an available model

1. Get `M_R`/`M_H`/`M_P` first (instant, no simulation):
   ```
   PYTHONPATH=..:. python3 visualize_spm.py --model-json <path> --bounds-only --true-mpmf --solver gurobi
   ```
2. Run all 4 paper baselines + COSMA at those budgets:
   ```
   PYTHONPATH=..:. python3 run_paper_baselines.py --model-json <path> \
       --budgets-kb <M_R> <M_H> <M_P> --solver gurobi \
       --out-csv results/<name>_paper_baselines.csv
   ```
   See `run_paper_baselines.py` / `docs/baseline_construction.md`.

## 4. Caveats on "matching numbers"

- The paper reports its actual reduction numbers as bar charts (Fig 3/4),
  not a numeric table — the only exact figures given in text are the
  aggregate averages (84% for human-designed models at `M_R`, 85% for NAS
  via divide-and-conquer), not clean per-model targets. "Matching" means
  matching the qualitative pattern (COSMA ≤ baselines always, gap biggest
  at `M_R` and shrinking toward `M_P`, greedy ≤ Belady), not reproducing
  exact byte counts.
- Different array config (this project's `configs/scale.cfg` vs. the
  paper's own hardware assumptions), a different exact PyTorch→TFLite
  export pipeline, and a reimplementation of the paper's baselines from its
  textual description (not their original code) all mean exact numeric
  match was never the realistic goal — see `docs/results_plan.md` §6 for
  the other acknowledged, permanent differences (solver/hardware).
- **Resolved, 2026-09-30 (was open since 2026-09-10)**: the paper's §V-B.2
  text states two things precisely: (1) "given a memory budget of MPMF
  (`M_P` and `M_Pp`), COSMA can always eliminate the non-compulsory
  off-chip data accesses... since this memory budget equals the maximum
  sum of all live tensors at any timestep" (a structural guarantee, true
  by construction for COSMA specifically, at any model); (2) "However, all
  other schemes incur non-compulsory data accesses except for the model
  FCN" (the 4 baselines are expected to stay nonzero at `M_P`, for every
  model except FCN).
  By now testing 8 real model/precision combinations, a clear split
  emerged: **ResNet-50 (both FP32 and INT8), S3D, FCN, and DeepLabV3 all
  have `M_R` == `MPMF` exactly** (confirmed via `visualize_spm.py
  --bounds-only`'s own "No interesting range -- floor == ceiling" message,
  not just "close") — for these, `M_R` and `M_P` are the literal same
  number, so there is no budget range below `M_P` to show the interesting
  gap at all, and testing at `M_R` (this project's default budget choice)
  automatically tests at `M_P` too. This explains, and fully resolves,
  the original open question: it was never a bug, these architectures
  (at our specific input scale/config) simply have zero slack between
  their structural floor and their schedule's peak footprint, for all 5
  schemes simultaneously — consistent with property (1) above (COSMA's
  guarantee), and *not itself a contradiction* of property (2), since (2)
  only constrains what happens at a real, non-trivial `M_R`-to-`M_P` gap,
  which these 5 models simply don't have.
  **By contrast, ResNeXt-50, DenseNet-121 (both precisions), and
  R2Plus1D-18 all have a real, nontrivial `M_R` != `MPMF` gap** (e.g.
  ResNeXt-50: `M_R`=9408.00KB vs `MPMF`=9636.00KB) — these were tested
  *at* `M_R` specifically (the paper's own tightest, most informative
  budget point), not at `M_P`, so the "baselines tie at 0" pattern
  correctly does **not** apply to them: ResNeXt-50's baselines fail
  outright at `M_R` (fragmentation), DenseNet-121's show real nonzero
  spill at both `M_R` and `M_H`.
  **Now tested at `MPMF`/`M_P`, 2026-09-30, on both ResNeXt-50 and
  DenseNet-121 (INT8)** (see their own rows for full numbers): both
  confirm property (1) exactly (COSMA hits the guaranteed 0 bytes,
  `Optimal`, on both) and property (2) in an even *stronger* form than
  the paper's own text describes — rather than merely "incurring
  non-compulsory data accesses" (implying suboptimal-but-successful
  placement), both baselines fail to place tensors *at all*
  (`TfliteArenaAllocationError`, fragmentation) on both models, even
  with a theoretically-sufficient total budget. R2Plus1D-18 remains
  untested at its own `M_P` (SCALE-Sim couldn't complete a run at its
  correct, paper-matching resolution at all — see its row).

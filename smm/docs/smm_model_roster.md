# Paper Model Roster: what's available to test against the SMM paper

Tracks which of the SMM paper's own 6 evaluation models (Table 2) we can actually run, so results
are checked against real availability, not assumed. Same format/purpose as
`cosma/docs/paper_model_roster.md` and `onsram/docs/onsram_model_roster.md` — checked directly
against `cosma/_exported/` on 2026-09-16.

**Paper**: Stavroula Zouzoula, Mohammad Ali Maleki, Muhammad Waqar Azhar, Pedro Trancoso.
"Scratchpad Memory Management for Deep Learning Accelerators." ICPP '24.
https://doi.org/10.1145/3673038.3673115

**Ported (2026-10-02)** — policy-selection logic and the SCALE-Sim driver now live under
`smm/smm_helpers/` (`policy_selector.py`, `scale_sim_runner.py`, `baseline.py`), matching
COSMA/OnSRAM's layout, with a real-SCALE-Sim fidelity fix (`scalesim/memory/smm_reuse_buffers.py`)
validated against ResNet18 — see `smm/docs/smm_verification.md` for the full writeup. Entry point:
`smm/run_smm.py`. This roster still only tracks model *availability*, not which models have
actually been run end-to-end yet (only ResNet18 has, so far).

## 1. Paper's own roster (Table 2)

6 models, all CNN/image-classification, described by layer-type composition: CV (conv), DW
(depthwise conv), PW (pointwise/1×1 conv), FC (fully-connected), PL (projection layer — the 1×1
shortcut conv in a residual block).

| Model | Layers (paper) | Layer types | Available now? | Notes |
|---|---|---|---|---|
| **MobileNet** | 28 | CV, DW, PW, FC | ✅ `_exported/MobileNet/model.json` | Ready to run today — same artifact OnSRAM's roster uses as "MobileNetV1". Its exported classifier is actually a 1×1 CONV2D, not a real DENSE op (confirmed 2026-10-04) — already fully covered by the topology-row path, no DENSE-costing needed for this one specifically. |
| **MobileNetV2** | 53 | CV, DW, PW, FC | ✅ `_exported/MobileNetV2/model.json` | Ready to run today. Has 1 real DENSE classifier layer — smoke-tested 2026-10-04 with `dense_costing.py`, correct. |
| **GoogLeNet** | 64 | CV, PW, FC | ✅ `_exported/GoogLeNet/model.json` | **Correction (2026-10-04): this roster's "not sourced" claim went stale** — the export exists (82 layers: 57 CONV2D, 1 DENSE, MAXPOOL, CONCAT for the inception branches; dated 2026-09-26, 10 days after this roster's own "checked against `_exported/` on 2026-09-16" line above), presumably added for the same gap OnSRAM's/COSMA's own rosters flagged. Not yet run through `run_smm.py`. MAXPOOL/CONCAT get no topology row (shared, pre-existing gap across all three papers' ports, same treatment as everywhere else non-conv ops appear) — only the DENSE classifier needed this session's new costing. |
| **EfficientNetB0** | 82 | CV, DW, PW, FC | ⚠️ `_exported/efficient50/model.json` | An EfficientNet variant is exported (confirmed via its ops: SIGMOID+MUL Swish-activation pattern, 1 real DENSE classifier layer) — but the folder name ("efficient50") doesn't confirm it's specifically B0 vs. a different compound-scaling variant. Same "paper doesn't pin the exact variant" caveat COSMA's roster already makes for DenseNet. |
| MnasNet | 53 | CV, DW, PW, FC | ❌ not sourced | Re-checked 2026-10-04, still genuinely absent from `_exported/` (no MnasNet-named directory of any kind) — would need exporting first |
| ResNet18 | 21 | CV, PW, FC, PL | ✅ `_exported/ResNet18/model.json` | **Sourced 2026-10-05** via `spm_common/source_torchvision_model.py` (torchvision `resnet18`, random weights, PyTorch→ONNX→`onnx2tf`→trim exporter — real architecture, not hand-built: 11,689,512 params matches torchvision's published ResNet-18 param count exactly). 37 layers total (20 CONV2D, 8 ADD, 5 PAD, 1 MAXPOOL, 1 TRANSPOSE, 1 REDUCE_MEAN, 1 DENSE) — the 20 CONV2D + 1 DENSE = **21** matches the paper's own Table 2 count exactly (ADD/PAD/MAXPOOL/REDUCE_MEAN/TRANSPOSE aren't counted by the paper's CV/PW/FC/PL scheme). Input `[1,224,224,3]`, standard ImageNet resolution. Smoke-tested through `run_smm.py` directly (replacing the old hand-built `smm/topologies/resnet18_same_padded.csv` stand-in, which is now superseded — that CSV had no DENSE metadata and used non-standard "same" padding, not the real architecture). This export also lands in `cosma/_exported/`, the same cache COSMA and OnSRAM read, so it's immediately usable by both with no further work — see `spm_common/docs/cross_paper_benchmark_testbed.md` for the cross-paper feasibility caveat (COSMA/OnSRAM need the on-chip budget to exceed ResNet-18's largest tracked tensor, ~3.06MB at this resolution — infeasible below that regardless of scheduling). |

## 2. Summary

**5 of 6 ready to run right now as real model.json exports** (MobileNet, MobileNetV2, GoogLeNet,
ResNet18, and EfficientNetB0 modulo the variant caveat), **1 not sourced** (MnasNet). The "GoogLeNet
gap shared across all three paper rosters" note from the original 2026-09-16 check no longer
holds — re-verify against `cosma/_exported/` directly (`ls`) rather than trusting this file's own
prior summary line, since it's already gone stale once.

**Every one of this paper's 6 models includes FC (fully-connected) layers** per the paper's own
Table 2 — but **not every exported model.json represents that head as a real DENSE op**. Checked
directly (2026-10-04): MobileNet(v1)'s exported classifier is a 1×1 CONV2D
(`REDUCE_MEAN -> CONV2D[1024->1000] -> SOFTMAX`), not DENSE at all — it was never actually dropped,
since `topology_builder.build_topology()` already emits a row for any CONV2D regardless of spatial
size. MobileNetV2 and EfficientNetB0 (`efficient50`) each have exactly 1 real DENSE layer in their
exports. **DENSE support is now implemented** (`smm_helpers/dense_costing.py`, wired into
`run_smm.py` for any `.json` model input) — analytically, deliberately matching COSMA/OnSRAM's own
`_nonconv_layer_stats()` DENSE formula rather than real-simulating it (SMM's own policies have no
spatial-reuse distinction to offer on a layer with no spatial extent, and real-simulating a large
DENSE layer risks the same SCALE-Sim OOM COSMA's own code already documents hitting on AlexNet —
see `smm_implementation.md` §3 and `dense_costing.py`'s docstring for the full reasoning). Smoke-
tested on MobileNetV2 (2026-10-04): correctly detected 1 DENSE layer, costed it at +161,280 cycles
/ +1253.2kB (matching its known 1280→1000 classifier shape almost exactly), added identically to
every scheme (baseline/Hom/Het) in the comparison table.

## 3. Config note (for matching the paper's own baseline, once ported)

Paper's own baseline setup (§4): 16×16 PE array, **output-stationary** dataflow, 8-bit data width,
16 elements/cycle off-chip bandwidth, batch size 1, ofmap buffer fixed at 4KB with the remaining
budget split ifmap/filter at 25-75%/50-50%/75-25% (three separate baseline configs, not one) —
GLB sizes tested: 64KB, 128KB, 256KB, 512KB, 1MB. None of this matches `configs/scale.cfg`'s
current defaults (which are WS dataflow, per the earlier architecture discussion) — worth a
dedicated config file (mirroring `configs/scale_onsram.cfg`'s own precedent) once this paper is
actually ported here, not reusing COSMA/OnSRAM's shared `configs/scale.cfg`.

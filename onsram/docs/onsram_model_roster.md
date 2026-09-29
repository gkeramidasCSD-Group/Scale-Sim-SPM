# Paper Model Roster: what's available to test against the OnSRAM paper

Tracks which of the OnSRAM paper's own 12 evaluation networks (Abstract, Table 1) we can
actually run, so results are checked against real availability, not assumed. Same format/purpose
as `cosma/docs/paper_model_roster.md`. Models live in `cosma/_exported/<name>/model.json` (the
shared export cache both papers read from, see `spm_common/model_resolver.py`). Last checked
2026-09-28.

## 1. Paper's own roster

12 real networks, evaluated at 3 TFLOP / 2MB SPM / 32 GBps / batch size 1 (per
`onsram_integration_plan.md` §4's paper-fidelity check).

Every runnable model also has an `<name>_unfused` export from `spm_common/unfuse_model.py`, which
rebuilds the BatchNorm/BiasAdd/ReLU nodes TFLite folded into the convs. The unfused version is
the one to compare against the paper (see `why_our_speedups_are_lower_than_the_paper.md`).

| Model | Export name (`--model`) | Status |
|---|---|---|
| **AlexNet** | `AlexNet` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (torchvision architecture, 61.1M params, random weights; only the architecture matters here) |
| **VGG-16** | `VGG16` (no unfused export) | ⚠️ Exported, but SCALE-Sim runs out of memory on it on this machine (7 GB RAM) and can take the desktop down. Opt-in only: `run_paper_reproduction.py` skips it unless named with `--model VGG16`. Not retried since DENSE layers moved to an analytic cost (the fix that made AlexNet run), so it may or may not still fail |
| **GoogLeNet** | `GoogLeNet` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (Inception v1 with BatchNorm, as TF-slim/torchvision; 6.6M params) |
| **Inception-v3** | `_exported_inception_v3-tflite-float` (+ `_unfused`) | ✅ Runs. Used to fail in OnSRAM's placement step at float32; places fine at FP16 |
| Inception-v4 | — | ❌ Not sourced |
| **ResNet-50** | `ResNeT50` (+ `_unfused`) | ✅ Runs. (`_exported_resnet50-tflite-float` is the same network plus input-preprocessing ops; the roster uses `ResNeT50`) |
| SSD300 | — | ❌ Not sourced (detection model, would need exporting first) |
| **ResNeXt** | `ResNeXt50` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (ResNeXt-50 32x4d, 25.1M params; 16 grouped convs, `groups: 32`) |
| **MobileNetV1** | `MobileNet` (+ `_unfused`) | ✅ Runs |
| **SqueezeNet** | `squeezenet1_1` (+ `_unfused`) | ✅ Runs. **Not** `squeezenet`: that export is corrupt (all 40 layers labelled `ADD`, no convolutions; its source `.tflite` is itself degenerate). `squeezenet1_1` comes from `trim/models/squeezenet1_1.tflite`. The paper doesn't say which SqueezeNet version it used |
| PTB-LSTM | — | ⛔ Structurally blocked (see §2) |
| Multi-Head Attention | — | ⛔ Structurally blocked (see §2) |
| *MobileNetV2* | `MobileNetV2` (+ `_unfused`) | Not in the paper; run as an extra data point |

## 2. Summary

**7 of 12 run in the paper sweep** (AlexNet, GoogLeNet, Inception-v3, ResNet-50, ResNeXt,
MobileNetV1, SqueezeNet), fused and unfused, plus MobileNetV2 as an extra. VGG-16 is exported but
opt-in because of memory. Inception-v4 and SSD300 are a plain sourcing gap: export them and
they're runnable.

PTB-LSTM and Multi-Head Attention are a different kind of gap. SCALE-Sim simulates only
`CONV2D`/`DEPTHWISE_CONV2D` (`onsram_helpers/topology.py`); DENSE is costed with SCALE-Sim's fold
formula and the other non-conv ops as pure data movement (`scale_sim_runner._nonconv_layer_stats()`).
A recurrent model needs its time steps unrolled into the graph, and attention needs
activation-by-activation matmuls, which no current topology row or cost rule covers. Both need
real modelling work before they'd produce trustworthy numbers, not just an export.

## 3. How to check numbers against the paper

```bash
cd onsram
python3 run_paper_reproduction.py                        # all 7 paper models + MobileNetV2, fused and unfused, 2MB
python3 run_paper_reproduction.py --model MobileNet      # one model (by its export name), both variants
python3 run_paper_reproduction.py --variant unfused      # only the paper-comparable variant
python3 run_paper_reproduction.py --no-scale-sim         # fast: pinning + allocator replay only, no speedups
```

It uses the paper-matched config (`configs/scale_onsram.cfg`: 39×39 array, 32 B/cycle DRAM),
writes one log per run to `onsram/logs/<model>_2MB_paper.log`, and rewrites
`onsram/results/paper_reproduction.csv` after every run (`results/` is gitignored). The summary
prints each run next to the paper's Fig. 7 OnSRAM-Static value and Table 1's ∞-SPM value, plus our
own ∞-SPM ceiling and a geomean over the paper models.

Run one sweep at a time: SCALE-Sim needs several GB per process, and parallel runs would overwrite
each other's `onsram/topology.csv`. The full default sweep (16 runs) took 1 h 55 min on
2026-09-28: about 20 min per variant for Inception-v3, 13–14 min for ResNet-50/ResNeXt-50, 4–5 min
for GoogLeNet, and 1–2 min each for the rest. The script rewrites its CSV from scratch, so when
re-running a subset, pass a different `--out-csv` to keep the earlier rows.

`run_onsram.py` also works on these models, but its default `--config` is the generic
`configs/scale.cfg`, not the paper hardware; pass `--config ../configs/scale_onsram.cfg` to get
paper-config numbers from it.

## 4. Caveats on "matching numbers"

- Per-model paper values: Table 1 prints ∞-SPM and 1-Step speedups for every model. Fig. 7 prints
  only two OnSRAM-Static values (ResNeXt 3.81×, MobileNetV1 4.76×); the others were read off the
  figure, about ±0.02 (method in `why_our_speedups_are_lower_than_the_paper.md`).
- Our exports come from a different pipeline than the paper's TensorFlow graphs (TFLite, with
  `unfuse_model.py` rebuilding the separate BatchNorm/BiasAdd/ReLU nodes), and our compute times
  come from SCALE-Sim's systolic array rather than the paper's calibrated model. Expect the same
  qualitative pattern, not identical numbers; the per-model comparison and its causes are in
  `why_our_speedups_are_lower_than_the_paper.md`.

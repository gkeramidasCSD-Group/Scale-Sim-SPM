# spm_common/source_torchvision_model.py
"""
Sources real torchvision model architectures via PyTorch -> ONNX ->
onnx2tf -> trim's exporter, into cosma/_exported/<Name>/model.json -- the
same recipe already used (ad hoc, each time, not as a reusable script)
for COSMA's ResNeXt-50, R2Plus1D-18, S3D, and FCN (see
cosma/docs/paper_model_roster.md). This script is the reusable version of
that recipe, for models that torchvision already defines authoritatively
(so the architecture itself is sourced, not hand-reconstructed the way
spm_common/build_paper_models.py's AlexNet/GoogLeNet/ResNeXt-50 are).

Weights are random (weights=None): this simulator only uses the
architecture (layer shapes, ops, weight sizes), never weight values --
same convention as every other model in this repo.

Usage:
    python3 spm_common/source_torchvision_model.py --model ResNet18
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spm_common.build_paper_models import _run_exporter  # noqa: E402 -- same exporter wiring, reused not duplicated

MODELS = {
    # name -> (torchvision constructor, input shape incl. batch dim)
    'ResNet18': ('resnet18', (1, 3, 224, 224)),
}


def build(name: str) -> str:
    import torch
    import torchvision.models as tvm

    ctor_name, shape = MODELS[name]
    model = getattr(tvm, ctor_name)(weights=None)
    model.eval()

    out_dir = os.path.join(_EXPORTED, name)
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, f'{name}.onnx')
    dummy = torch.randn(*shape)
    torch.onnx.export(model, dummy, onnx_path, input_names=['input'],
                       output_names=['output'], opset_version=18)

    import onnx2tf
    tf_dir = os.path.join(out_dir, 'tf_saved_model')
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=tf_dir,
                     non_verbose=True, disable_model_save=False)

    tflite_candidates = [f for f in os.listdir(tf_dir) if f.endswith('.tflite')]
    if not tflite_candidates:
        raise FileNotFoundError(f"onnx2tf produced no .tflite in {tf_dir}")
    # Prefer the plain float32 one if more than one variant got written.
    tflite_candidates.sort(key=lambda f: 0 if 'float32' in f.lower() else 1)
    tflite_path = os.path.join(tf_dir, tflite_candidates[0])

    _run_exporter(tflite_path, out_dir)
    print(f"[+] {name}: {sum(p.numel() for p in model.parameters()):,} params "
          f"-> {out_dir}/model.json")
    return os.path.join(out_dir, 'model.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=list(MODELS), action='append',
                        help="model(s) to source (default: all)")
    args = parser.parse_args()
    for name in args.model or list(MODELS):
        build(name)


if __name__ == '__main__':
    main()

# spm_common/build_paper_models_tf_native.py
"""
Builds a real tf.keras.applications architecture and converts it with the
REAL tf.lite.TFLiteConverter.from_keras_model() -- a genuinely TF-native
TFLite export path, as opposed to every other COSMA ResNet-50 export so far,
which is either externally-sourced (`cosma/_exported/resnet50-tflite-float/
resnet50.tflite` -- its own metadata.json shows `tool_versions: {"qairt":
...}`, i.e. downloaded pre-converted from Qualcomm AI Hub, not TF-native and
not onnx2tf either) or PyTorch->ONNX->onnx2tf->trim-sourced (ResNeXt-50/S3D/
FCN/R2Plus1D-18, per cosma/docs/paper_model_roster.md).

Mirrors cosma/tools/quantize_model.py's build_keras_model() (same
tf.keras.applications name resolution, same frozen-batch-size-1 pattern,
itself mirroring trim/python_scripts/download_tf_model.py's model
construction) but skips post-training quantization entirely -- plain FP32
conversion, no converter.optimizations set, matching
spm_common/build_paper_models.py's own build() exactly -- so this is an
unconfounded FP32-vs-FP32 structural comparison against the existing
ResNet-50 baseline, not FP32-vs-INT8 (quantize_model.py already gives that,
confounded by quantization's own extra PAD/SUB/MUL bookkeeping ops -- see
cosma/docs/ITERATION_HISTORY.md's entry on this session's baseline-fidelity
investigation).

Usage:
    python3 spm_common/build_paper_models_tf_native.py --model resnet50
    python3 spm_common/build_paper_models_tf_native.py --model resnet50 --weights none
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from spm_common import model_resolver  # noqa: E402

_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')


def _resolve_keras_app(name: str):
    """Same lookup as cosma/tools/quantize_model.py's build_keras_model()
    (duplicated, not imported, since that file lives under cosma/tools/ as
    a standalone script, not an importable package -- same precedent
    build_paper_models.py already set for its own 3 models)."""
    model_map = {}
    for attr_name in dir(tf.keras.applications):
        attr = getattr(tf.keras.applications, attr_name)
        if callable(attr) and not attr_name.startswith('_') and attr_name[0].isupper():
            model_map[attr_name.lower()] = (attr_name, attr)
    key = name.lower().replace('-', '').replace('_', '')
    if key not in model_map:
        raise ValueError(f"Unrecognized Keras model {name!r} -- available: "
                          f"{sorted(n for n, _ in model_map.values())}")
    return model_map[key]


def build(name: str, weights: str = 'imagenet') -> str:
    official_name, model_fn = _resolve_keras_app(name)
    kwargs = {'weights': None if weights == 'none' else weights}
    print(f"[*] Loading tf.keras.applications.{official_name} (weights={kwargs['weights']})...")
    base_model = model_fn(**kwargs)  # include_top=True default -- matches the paper's classifier head

    input_shape = base_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    static_shape = (1,) + input_shape[1:]  # freeze batch to 1, same as quantize_model.py
    static_input = tf.keras.Input(batch_shape=static_shape)
    static_model = tf.keras.Model(static_input, base_model(static_input), name=official_name)

    out_dir = os.path.join(_EXPORTED, f'{name}_tf_native')
    os.makedirs(out_dir, exist_ok=True)
    tflite_path = os.path.join(out_dir, f'{official_name}_tf_native.tflite')
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)  # plain FP32, no optimizations
    with open(tflite_path, 'wb') as f:
        f.write(converter.convert())
    print(f"[+] {official_name}: {static_model.count_params():,} params, "
          f"input {static_shape} -> {tflite_path}")

    model_json_path = model_resolver.resolve_model_json(tflite_path, mode='fp32')
    print(f"[+] Exported -> {model_json_path}")
    return model_json_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='tf.keras.applications name, e.g. resnet50')
    ap.add_argument('--weights', default='imagenet', choices=['imagenet', 'none'],
                     help="'imagenet' (default, already cached locally, no download) or "
                          "'none' for random weights (architecture-only -- COSMA never reads "
                          "weight values, see build_paper_models.py's own docstring).")
    args = ap.parse_args()
    build(args.model, args.weights)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# cosma/tools/quantize_model.py
"""
Post-training INT8 quantization for the COSMA paper-comparison model set, to
match the paper's own evaluation datatype (arXiv:2311.18246 Fig.3 caption:
"All data are of 8-bit datatype"). Every model this project has exported so
far used FP32 (4x the paper's tensor byte sizes) -- see
cosma/docs/paper_model_roster.md and results_plan.md for the gap this fixes.

trim/python_scripts/export_model.py's own `--mode int8` is already fully
implemented end-to-end (quant metadata, per-channel scales -- see
lib/export_hooks.py's `_is_int8` branches) -- it just never had a real INT8
.tflite to consume, because nothing in either repo produced one. This
script produces that INT8 .tflite; trim's existing exporter takes it from
there unchanged.

Source path covered here: tf.keras.applications models (ResNet50,
DenseNet121, ...) -- mirrors trim/python_scripts/download_tf_model.py's own
model construction (imagenet weights, batch size frozen to 1) so the
quantized model matches the same architecture our existing FP32 exports
already use, then applies TFLiteConverter's own full-integer
post-training-quantization (PTQ) API on top.

Calibration data: COSMA never inspects tensor *values*, only shapes and
structure (same precedent already established for R2Plus1D-18's
random-weight CONV_3D export -- see paper_model_roster.md) -- so a
representative_dataset of random inputs at the model's real input shape is
sufficient to calibrate activation ranges. No real ImageNet/CIFAR data is
needed, and none is used.

Usage:
  python3 quantize_model.py --model resnet50 \
      --out /home/george/Desktop/trim/models/resnet50_int8/resnet50.tflite
  python3 quantize_model.py --model densenet121 \
      --out /home/george/Desktop/trim/models/densenet121_int8/densenet121.tflite

Does NOT cover R2Plus1D-18/S3D (CONV_3D, PyTorch->ONNX->onnx2tf sourced, not
a tf.keras.applications model) -- TFLite's CONV_3D builtin op's INT8 kernel
support is unconfirmed; if/when attempted, that's a separate quantization
path (onnx2tf's own `-oiqt`/calibration flags on the existing ONNX export),
not this script. See cosma/docs/paper_model_roster.md for that model's
current status.
"""
import argparse

import numpy as np
import tensorflow as tf


def _representative_dataset(input_shape, num_samples: int = 100):
    """Random inputs at the model's real shape -- see module docstring for
    why real calibration data isn't needed here."""
    def gen():
        for _ in range(num_samples):
            yield [np.random.uniform(-1, 1, size=input_shape).astype(np.float32)]
    return gen


def build_keras_model(name: str, input_hw=None):
    """
    Builds a frozen-batch-size-1 tf.keras.applications model, exactly
    mirroring trim/python_scripts/download_tf_model.py's own
    get_keras_model_map()/model-construction logic (not imported directly
    -- trim/ is a sibling repo invoked via subprocess everywhere else in
    this project, never imported in-process -- see
    spm_common/model_resolver.py's own DEFAULT_EXPORTER), so the quantized
    model matches the same architecture/weights our existing FP32 exports
    already use.

    Returns (static_model, static_input_shape).
    """
    model_map = {}
    for attr_name in dir(tf.keras.applications):
        attr = getattr(tf.keras.applications, attr_name)
        if callable(attr) and not attr_name.startswith('_') and attr_name[0].isupper():
            model_map[attr_name.lower()] = (attr_name, attr)

    key = name.lower().replace('-', '').replace('_', '')
    if key not in model_map:
        raise ValueError(
            f"Unrecognized Keras model {name!r} -- available: "
            f"{sorted(m[0] for m in model_map.values())}")
    official_name, model_fn = model_map[key]

    kwargs = {'weights': 'imagenet'}
    if input_hw:
        kwargs['input_shape'] = (input_hw[0], input_hw[1], 3)
    print(f"[*] Loading tf.keras.applications.{official_name} (imagenet weights)...")
    base_model = model_fn(**kwargs)

    input_shape_tuple = base_model.input_shape
    if isinstance(input_shape_tuple, list):
        input_shape_tuple = input_shape_tuple[0]
    static_shape = (1,) + input_shape_tuple[1:]

    static_input = tf.keras.Input(batch_shape=static_shape)
    static_model = tf.keras.Model(inputs=static_input, outputs=base_model(static_input))
    return static_model, static_shape


def quantize_keras_model(model, input_shape, out_path: str) -> str:
    """
    Full-integer post-training quantization: every op (not just weights)
    is INT8, including the model's own input/output tensors -- matching
    the paper's "All data are of 8-bit datatype" as closely as TFLite's
    standard converter allows. Raises if any op in the graph can't be
    quantized this way (e.g. falls back to a float op) -- surfacing that
    loudly here is better than silently producing a mixed-precision
    .tflite that would misreport COSMA's own byte-size math downstream.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[*] Converting with full-integer INT8 post-training quantization "
          "(this calibrates over 100 random representative inputs)...")
    tflite_model = converter.convert()

    import os
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                     help='tf.keras.applications model name, e.g. resnet50, densenet121')
    ap.add_argument('--out', required=True, help='Output path for the INT8 .tflite file.')
    ap.add_argument('--input-shape', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                     help='Override input H/W (default: the architecture\'s own default).')
    args = ap.parse_args()

    model, static_shape = build_keras_model(args.model, args.input_shape)
    print(f"[*] Built {args.model} with static input shape {static_shape}")
    out_path = quantize_keras_model(model, static_shape, args.out)
    print(f"[+] Wrote INT8 model to {out_path}")


if __name__ == '__main__':
    main()

# spm_common/build_paper_models.py
"""
Builds the OnSRAM paper models that have no ready-made TFLite file --
AlexNet, GoogLeNet (Inception v1) and ResNeXt-50 (32x4d) -- as Keras models,
converts them to TFLite (static batch 1, like trim's download_tf_model.py)
and exports them with trim's exporter to cosma/_exported/<Name>/model.json.

Weights are random: this simulator only uses the architecture (layer
shapes, ops, weight sizes), never the weight values, so no pretrained
download is needed. Architectures follow the standard definitions:
  - AlexNet: torchvision's version (the "one weird trick" single-tower
    variant: 64-192-384-256-256 convs, 4096-4096-1000 dense).
  - GoogLeNet: Inception v1 with BatchNorm after every conv, as in both
    TF-slim's inception_v1 and torchvision (whose "5x5" branch is a 3x3
    conv; kept here), no auxiliary classifiers (inference).
  - ResNeXt-50 32x4d: torchvision's resnext50_32x4d (bottleneck width
    2x planes, 32-group 3x3 convs, stages 3-4-6-3).

Usage:
    python3 spm_common/build_paper_models.py                  # all three
    python3 spm_common/build_paper_models.py --model AlexNet
"""
import argparse
import os
import runpy
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')
_TRIM_ROOT = '/home/george/Desktop/trim'
_EXPORTER = os.path.join(_TRIM_ROOT, 'python_scripts', 'export_model.py')


def _run_exporter(tflite_path: str, out_dir: str) -> None:
    """
    Runs trim's export_model.py in this process, with grouped-conv support
    patched in (trim's own files are not modified). TFLite stores a grouped
    conv's filter as [outC, kh, kw, inC/groups]; trim's exporter only
    accepts [.., inC] and fails on ResNeXt ("Cannot infer Conv2D axes").
    The patch calls trim's own conv export with the per-group input count
    (so its axis inference and (K, outC) reshape work unchanged), then
    records the real group count in the layer's params["groups"].
    """
    # Same import paths export_model.py sets up for itself (its repo root,
    # and python_scripts/ for trim's generated `tflite` package).
    for path in (os.path.join(_TRIM_ROOT, 'python_scripts'), _TRIM_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from python_scripts.lib import export_hooks as eh

    if not getattr(eh.WeightExporter, '_grouped_conv_patch', False):
        original = eh.WeightExporter._export_conv2d

        def _export_conv2d(self, model, inputs_raw, outputs_raw, in_shape, out_shape,
                           layer_id, out_dir, meta):
            groups = 1
            if len(inputs_raw) >= 2 and in_shape:
                w_shape = [int(d) for d in eh.shape_of(model, inputs_raw[1])]
                in_c = int(in_shape[-1])
                if len(w_shape) == 4 and 0 < w_shape[3] < in_c and in_c % w_shape[3] == 0:
                    groups = in_c // w_shape[3]
                    in_shape = list(in_shape[:-1]) + [w_shape[3]]
            original(self, model, inputs_raw, outputs_raw, in_shape, out_shape,
                     layer_id, out_dir, meta)
            if groups > 1:
                meta['params']['groups'] = groups

        eh.WeightExporter._export_conv2d = _export_conv2d
        eh.WeightExporter._grouped_conv_patch = True

    argv = sys.argv
    sys.argv = [_EXPORTER, '--model', tflite_path, '--out', out_dir, '--mode', 'fp32']
    try:
        runpy.run_path(_EXPORTER, run_name='__main__')
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = argv


def alexnet():
    x = inp = layers.Input((224, 224, 3), batch_size=1)
    x = layers.ZeroPadding2D(2)(x)
    x = layers.Conv2D(64, 11, strides=4, activation='relu')(x)          # 55x55
    x = layers.MaxPool2D(3, 2)(x)                                        # 27x27
    x = layers.Conv2D(192, 5, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(3, 2)(x)                                        # 13x13
    x = layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(3, 2)(x)                                        # 6x6
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return tf.keras.Model(inp, x, name='AlexNet')


def _conv_bn(x, filters, kernel, strides=1, groups=1, relu=True):
    x = layers.Conv2D(filters, kernel, strides=strides, padding='same',
                      groups=groups, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x) if relu else x


def googlenet():
    def inception(x, c1, c3r, c3, c5r, c5, cp):
        b1 = _conv_bn(x, c1, 1)
        b2 = _conv_bn(_conv_bn(x, c3r, 1), c3, 3)
        b3 = _conv_bn(_conv_bn(x, c5r, 1), c5, 3)   # "5x5" branch is 3x3 (slim/torchvision)
        b4 = _conv_bn(layers.MaxPool2D(3, 1, padding='same')(x), cp, 1)
        return layers.Concatenate()([b1, b2, b3, b4])

    x = inp = layers.Input((224, 224, 3), batch_size=1)
    x = _conv_bn(x, 64, 7, strides=2)                                    # 112
    x = layers.MaxPool2D(3, 2, padding='same')(x)                        # 56
    x = _conv_bn(x, 64, 1)
    x = _conv_bn(x, 192, 3)
    x = layers.MaxPool2D(3, 2, padding='same')(x)                        # 28
    x = inception(x, 64, 96, 128, 16, 32, 32)                            # 3a
    x = inception(x, 128, 128, 192, 32, 96, 64)                          # 3b
    x = layers.MaxPool2D(3, 2, padding='same')(x)                        # 14
    x = inception(x, 192, 96, 208, 16, 48, 64)                           # 4a
    x = inception(x, 160, 112, 224, 24, 64, 64)                          # 4b
    x = inception(x, 128, 128, 256, 24, 64, 64)                          # 4c
    x = inception(x, 112, 144, 288, 32, 64, 64)                          # 4d
    x = inception(x, 256, 160, 320, 32, 128, 128)                        # 4e
    x = layers.MaxPool2D(3, 2, padding='same')(x)                        # 7
    x = inception(x, 256, 160, 320, 32, 128, 128)                        # 5a
    x = inception(x, 384, 192, 384, 48, 128, 128)                        # 5b
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return tf.keras.Model(inp, x, name='GoogLeNet')


def resnext50_32x4d():
    def bottleneck(x, planes, stride):
        width = planes * 2            # int(planes * 4/64) * 32 groups
        out = _conv_bn(x, width, 1)
        out = _conv_bn(out, width, 3, strides=stride, groups=32)
        out = _conv_bn(out, planes * 4, 1, relu=False)
        if stride != 1 or x.shape[-1] != planes * 4:
            x = _conv_bn(x, planes * 4, 1, strides=stride, relu=False)
        return layers.ReLU()(layers.Add()([out, x]))

    x = inp = layers.Input((224, 224, 3), batch_size=1)
    x = _conv_bn(x, 64, 7, strides=2)
    x = layers.MaxPool2D(3, 2, padding='same')(x)
    for planes, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
        for b in range(blocks):
            x = bottleneck(x, planes, stride if b == 0 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return tf.keras.Model(inp, x, name='ResNeXt50')


MODELS = {'AlexNet': alexnet, 'GoogLeNet': googlenet, 'ResNeXt50': resnext50_32x4d}


def build(name: str) -> str:
    out_dir = os.path.join(_EXPORTED, name)
    os.makedirs(out_dir, exist_ok=True)
    model = MODELS[name]()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_path = os.path.join(out_dir, f'{name}.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(converter.convert())
    _run_exporter(tflite_path, out_dir)
    print(f"[+] {name}: {model.count_params():,} params -> {out_dir}/model.json")
    return os.path.join(out_dir, 'model.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=list(MODELS), action='append',
                        help="model(s) to build (default: all)")
    args = parser.parse_args()
    for name in args.model or list(MODELS):
        build(name)


if __name__ == '__main__':
    main()

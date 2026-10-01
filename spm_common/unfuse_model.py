# spm_common/unfuse_model.py
"""
Writes an "unfused" copy of a TFLite-exported model.json: every fused
conv/dense layer is split back into the separate nodes a TensorFlow graph
has -- Conv -> BatchNorm (or BiasAdd) -> ReLU -- and a fused activation on
any other op (e.g. ResNet-50's ADD+ReLU) becomes its own ReLU node.

Why: the OnSRAM paper's headline results (Table 1, Fig. 7) run TensorFlow
graphs where BatchNorm/ReLU/BiasAdd are separate, strongly memory-bound
nodes; with layer fusion the paper reports much smaller speedups
(Sec. 7.1.6: 1.01-2.17x). TFLite exports are fused. This lets both
COSMA and OnSRAM run the unfused graph, unchanged: the inserted nodes are
ordinary non-conv layers, which both runners cost as pure data movement.

Port of load_graph_from_json_unfused() in
/home/george/trim/spm_management/onsram_bw/onsram_unfused.py, with these
corrections:
  - ReLU/ReLU6 is added only where model.json records a fused activation
    (the reference added ReLU after every conv; e.g. MobileNetV2's 17
    linear-bottleneck convs and ResNet-50's 20 pre-ADD convs have none),
    and also after non-conv ops that had one (ResNet-50's 16 ADD+ReLU).
  - BatchNorm is added only for networks that actually have it (MobileNet
    V1/V2, ResNets, ResNeXt, Inception/GoogLeNet, DenseNet, EfficientNet --
    not SqueezeNet, VGG or AlexNet, which the reference also gave
    BatchNorm). A network without it gets BiasAdd instead, as in its
    TensorFlow graph. model.json can't say
    which: TFLite folds BatchNorm into the conv weights, so this is chosen
    by model name, overridable with --batchnorm yes|no.
  - Dense layers get BiasAdd (never BatchNorm); a BatchNorm network's
    final classifier conv with no activation (e.g. MobileNet V1's) gets
    BiasAdd too.
  - New tensor ids start above the model's highest id (the reference used a
    fixed 1000, which can collide on large models), layers are renumbered
    0..N-1 in order, and inputs_from / topo_sort are rebuilt to match.

The inserted nodes have no weights (BatchNorm's 4*C parameters are tiny
and ignored). Usage:

    python3 spm_common/unfuse_model.py MobileNet
    python3 spm_common/unfuse_model.py path/to/model.json --batchnorm no

Writes cosma/_exported/<name>_unfused/model.json by default.
"""
import argparse
import copy
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')

CONV_LIKE = ('CONV2D', 'DEPTHWISE_CONV2D', 'CONV_3D', 'DENSE')

# Substrings of the model name -> whether the original network has BatchNorm.
_NO_BATCHNORM = ('squeezenet', 'vgg', 'alexnet')
_BATCHNORM = ('mobilenet', 'resnet', 'resnext', 'inception', 'googlenet', 'densenet', 'efficient')


def has_batchnorm_by_name(name: str):
    """True/False from the model name, or None if unknown."""
    n = name.lower()
    if any(k in n for k in _NO_BATCHNORM):
        return False
    if any(k in n for k in _BATCHNORM):
        return True
    return None


def _activation(layer: dict):
    params = layer.get('params') if isinstance(layer.get('params'), dict) else {}
    act = params.get('activation', layer.get('activation'))
    if act is None or str(act).upper() in ('NONE', ''):
        return None
    return str(act).upper()


def _clear_activation(layer: dict) -> None:
    if isinstance(layer.get('params'), dict) and 'activation' in layer['params']:
        layer['params']['activation'] = None
    if 'activation' in layer:
        layer['activation'] = None


def _has_bias(layer: dict) -> bool:
    bias = layer.get('bias') or {}
    return bool(bias.get('elements', bias.get('size', 0)))


def unfuse(model: dict, has_batchnorm: bool) -> dict:
    """Returns an unfused deep copy of model (see module docstring)."""
    model = copy.deepcopy(model)
    tensors = {t['id']: t for t in model['tensors']}
    next_tid = max(tensors) + 1
    layers = model['layers']
    last_conv_like = max((i for i, l in enumerate(layers) if l['op'] in CONV_LIKE), default=-1)

    new_layers = []
    added = {}
    for idx, layer in enumerate(layers):
        op = layer['op']
        act = _activation(layer)
        post_ops = []
        if op in CONV_LIKE:
            if op != 'DENSE' and has_batchnorm and not (idx == last_conv_like and act is None):
                post_ops.append('BATCH_NORM')
            elif _has_bias(layer):
                post_ops.append('BIAS_ADD')
        if act:
            post_ops.append(act)

        if not post_ops or not layer.get('outputs'):
            new_layers.append(layer)
            continue

        final_out = layer['outputs'][0]
        out_info = tensors[final_out]
        out_shape = out_info['shape']

        def new_tensor():
            nonlocal next_tid
            tid = next_tid
            next_tid += 1
            model['tensors'].append({'id': tid, 'shape': list(out_shape), 'dtype': out_info['dtype']})
            return tid

        current = new_tensor()
        layer['outputs'] = [current] + layer['outputs'][1:]
        _clear_activation(layer)
        new_layers.append(layer)
        for k, post_op in enumerate(post_ops):
            out = final_out if k == len(post_ops) - 1 else new_tensor()
            new_layers.append({
                'id': None,  # renumbered below
                'op': post_op,
                'inputs': [current],
                'outputs': [out],
                'input_shape': list(out_shape),
                'output_shape': list(out_shape),
                'params': {},
                'unfused_from': layer['id'],
            })
            added[post_op] = added.get(post_op, 0) + 1
            current = out

    # Renumber 0..N-1 in (topological) list order and rebuild inputs_from.
    for i, layer in enumerate(new_layers):
        layer['id'] = i
    producer = {tid: layer['id'] for layer in new_layers for tid in layer.get('outputs', [])}
    for layer in new_layers:
        layer['inputs_from'] = [producer.get(tid, -1) for tid in layer.get('inputs', [])]
    model['layers'] = new_layers
    if isinstance(model.get('topo_sort'), dict):
        model['topo_sort'] = {'enabled': True, 'order': list(range(len(new_layers))),
                              'mapping': {str(i): i for i in range(len(new_layers))}}
    model['unfused'] = {'batchnorm': has_batchnorm, 'added_layers': added}
    return model


def _resolve(model_arg: str):
    if os.path.isfile(model_arg):
        path = model_arg
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    else:
        path = os.path.join(_EXPORTED, model_arg, 'model.json')
        name = model_arg
    if not os.path.isfile(path):
        sys.exit(f"model.json not found: {path}")
    return path, name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="model.json path, or a name under cosma/_exported/")
    parser.add_argument('--batchnorm', choices=('yes', 'no'), default=None,
                        help="whether the original network has BatchNorm (default: from the model name)")
    parser.add_argument('--out', default=None,
                        help="output model.json (default: cosma/_exported/<name>_unfused/model.json)")
    args = parser.parse_args()

    path, name = _resolve(args.model)
    if args.batchnorm is not None:
        has_bn = args.batchnorm == 'yes'
    else:
        has_bn = has_batchnorm_by_name(name)
        if has_bn is None:
            sys.exit(f"don't know whether '{name}' has BatchNorm -- pass --batchnorm yes|no")

    with open(path, 'r') as f:
        model = json.load(f)
    unfused = unfuse(model, has_bn)
    unfused['unfused']['source'] = os.path.relpath(path, _REPO_ROOT)

    out = args.out or os.path.join(_EXPORTED, f"{name}_unfused", 'model.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(unfused, f, indent=2)
    print(f"{name}: {len(model['layers'])} -> {len(unfused['layers'])} layers "
          f"(added {unfused['unfused']['added_layers']}, batchnorm={has_bn}) -> {out}")


if __name__ == '__main__':
    main()

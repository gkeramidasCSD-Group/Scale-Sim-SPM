#!/usr/bin/env python3
# cosma/tools/compare_model_structures.py
"""
Read-only structural comparison across ResNet-50 exports from different
toolchains, to check whether M_R==MPMF (zero spill/retrieve range) is a
toolchain artifact or a genuine property of the architecture. Built to
check the hypothesis in cosma/docs/ITERATION_HISTORY.md's baseline-fidelity
investigation entry: does a genuinely different real TFLite-conversion
toolchain (native tf.keras.applications.ResNet50 + real TFLiteConverter,
see spm_common/build_paper_models_tf_native.py) produce a different
M_R/MPMF relationship than the existing QAIRT-sourced export?

Does not modify any existing files or data -- pure read/report, reusing
cosma_Ilp.py's and graph_builder.py's own trusted, unmodified functions.

Usage:
    PYTHONPATH=..:. python3 tools/compare_model_structures.py
"""
import collections
import difflib
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'cosma')):
    if p not in sys.path:
        sys.path.insert(0, p)

from spm_common import graph_builder
from helpers import cosma_Ilp

_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')

ROWS = [
    ('QAIRT FP32 (fused, current baseline)',
     '_exported_resnet50-tflite-float/model.json'),
    ('QAIRT FP32 (unfused)',
     '_exported_resnet50-tflite-float_unfused/model.json'),
    ('native-TF-Keras INT8 (existing)',
     'models_resnet50_int8_resnet50/model.json'),
    ('native-TF-Keras FP32 (new, this session)',
     '_exported_resnet50_tf_native_ResNet50_tf_native/model.json'),
]


def max_concurrency_width(nodes, tensors):
    windows = cosma_Ilp._liveness_windows(tensors)
    T = sorted(nodes.keys())
    best_t, best_w = None, 0
    for t in T:
        w = sum(1 for (s, e) in windows.values() if s <= t <= e)
        if w > best_w:
            best_w, best_t = w, t
    return best_w, best_t


def op_sequence(nodes):
    return [nodes[t].op for t in sorted(nodes.keys())]


def main():
    loaded = {}
    print(f"{'Row':42} {'layers':>7} {'tensors':>8} {'M_R (B)':>12} {'MPMF (B)':>12} {'gap':>10} {'width':>6}")
    print('-' * 100)
    for label, rel_path in ROWS:
        path = os.path.join(_EXPORTED, rel_path)
        if not os.path.exists(path):
            print(f"{label:42} -- MISSING: {path}")
            continue
        nodes, tensors = graph_builder.load_graph(path)
        mr, mr_t = cosma_Ilp.compute_structural_minimum_bytes(nodes, tensors)
        mp, mp_t = cosma_Ilp.compute_mpmf_bytes(nodes, tensors)
        width, width_t = max_concurrency_width(nodes, tensors)
        loaded[label] = (nodes, tensors)
        print(f"{label:42} {len(nodes):7d} {len(tensors):8d} {mr:12,d} {mp:12,d} {mp-mr:10,d} {width:6d}")

    for label, rel_path in ROWS:
        path = os.path.join(_EXPORTED, rel_path)
        if not os.path.exists(path):
            continue
        nodes, tensors = loaded[label]
        hist = collections.Counter(nodes[t].op for t in nodes)
        print(f"\n--- {label} op histogram ---")
        for op, count in hist.most_common():
            print(f"  {op:20s} {count}")

    fp32_labels = [l for l, _ in ROWS if 'FP32' in l]
    if all(l in loaded for l in fp32_labels):
        a_label, b_label = fp32_labels[0], fp32_labels[-1]
        a_ops = op_sequence(loaded[a_label][0])
        b_ops = op_sequence(loaded[b_label][0])
        print(f"\n--- op-sequence diff: {a_label!r} vs {b_label!r} ---")
        diff = list(difflib.unified_diff(a_ops, b_ops, lineterm='',
                                          fromfile=a_label, tofile=b_label))
        if not diff:
            print("  (identical op sequences)")
        else:
            for line in diff:
                print(f"  {line}")


if __name__ == '__main__':
    main()

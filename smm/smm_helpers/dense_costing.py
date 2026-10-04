# smm/smm_helpers/dense_costing.py
"""
Analytical cost of DENSE (fully-connected, batch 1) layers -- a
self-contained copy of the SAME formula cosma/helpers/baseline.py's and
onsram/onsram_helpers/scale_sim_runner.py's own `_nonconv_layer_stats()`
already use for DENSE, not a reinterpretation. Duplicated here rather than
imported, for the same reason OnSRAM keeps its own copy instead of
importing COSMA's: a change to one paper's DENSE costing must never change
another's simulated numbers (see onsram_helpers/resident_buffers.py's
docstring for the same reasoning applied to a different shared mechanism).

Why DENSE is costed analytically here too, instead of through real
SCALE-Sim simulation like every conv layer in this port: SCALE-Sim's own
per-weight address bookkeeping scales with weight COUNT, and a dense
layer's weights (n_in * n_out, no spatial reuse at batch 1) can be huge --
COSMA's own code documents a real, already-hit OOM on AlexNet's 9216x4096
dense layer (>5GB). SMM's own policies (P1-P5) also have nothing to
contribute to a dense layer specifically: every policy's distinction is
about SPATIAL reuse in a 2D conv (sliding windows, per-channel streaming),
and a dense layer has no spatial extent at all (it IS representable as a
1x1 conv with IH=IW=OH=OW=1, at which point every policy's formula
collapses to the same "load weights once, move a small vector" cost --
there is no real per-policy choice happening for this layer shape). So
real-simulating DENSE for SMM specifically, while COSMA/OnSRAM cost it
analytically, would measure simulation fidelity on a layer type none of
the three papers' own algorithms actually act on -- a confound in a
cross-paper benchmark, not a result. Matching COSMA/OnSRAM's existing,
already-proven approach keeps the comparison apples-to-apples and avoids
the OOM risk entirely (decided with the user 2026-10-04; see
smm/docs/smm_implementation.md).

Formula (identical to cosma/onsram's own): a batch-1 dense layer produces
one output vector, so its compute time is SCALE-Sim's own weight-stationary
fold formula, ceil(n_in/rows) * ceil(n_out/cols) * (rows + cols), which
COSMA's own code claims matches SCALE-Sim's measured cycles to ~1% on
convs. Its DRAM traffic is the weight matrix once (no reuse possible --
every weight is used exactly once to produce the single output vector)
plus the input and output vectors each once.

Precision: BYTES_PER_ELEMENT = 1, matching the paper's own Sec. 4 hardware
("The data width is 8-bits") and smm_helpers.policy_selector.HwParams's
own default -- NOT OnSRAM's FP16 rescaling or COSMA's raw model.json
float32 sizes, since those are each paper's own hardware assumption, not
SMM's.
"""
from __future__ import annotations

import json
import math
from typing import Tuple

BYTES_PER_ELEMENT = 1


def cost_dense_layer(n_in: int, n_out: int, array_dims: Tuple[int, int],
                      weight_elems: int = None, bytes_per_element: int = BYTES_PER_ELEMENT) -> dict:
    """
    compute_cycles / ifmap_dram_bytes / filter_dram_bytes / ofmap_dram_bytes
    for one DENSE layer. weight_elems defaults to n_in*n_out (no bias) when
    not given explicitly (e.g. from model.json's own weights+bias element
    counts, which may include a bias vector the n_in*n_out product doesn't).
    """
    rows, cols = array_dims
    if weight_elems is None:
        weight_elems = n_in * n_out
    return dict(
        compute_cycles=math.ceil(n_in / rows) * math.ceil(n_out / cols) * (rows + cols),
        ifmap_dram_bytes=n_in * bytes_per_element,
        filter_dram_bytes=weight_elems * bytes_per_element,
        ofmap_dram_bytes=n_out * bytes_per_element,
    )


def cost_dense_layers_in_model(model_json_path: str, array_dims: Tuple[int, int],
                                bytes_per_element: int = BYTES_PER_ELEMENT) -> dict:
    """
    Sums cost_dense_layer() over every DENSE layer in a model.json. Returns
    a dict with the same 4 keys as cost_dense_layer, aggregated, plus
    'num_dense_layers' and 'total_dram_bytes' for convenience.
    """
    with open(model_json_path, 'r') as f:
        model = json.load(f)

    total = dict(compute_cycles=0, ifmap_dram_bytes=0, filter_dram_bytes=0,
                 ofmap_dram_bytes=0, num_dense_layers=0)
    for layer in model['layers']:
        if layer.get('op') != 'DENSE':
            continue
        n_in = layer['input_shape'][-1]
        n_out = layer['output_shape'][-1]
        weight_elems = sum(part.get('elements', 0) for part in
                            (layer.get('weights', {}), layer.get('bias', {}))) or None
        stats = cost_dense_layer(n_in, n_out, array_dims, weight_elems, bytes_per_element)
        for k in ('compute_cycles', 'ifmap_dram_bytes', 'filter_dram_bytes', 'ofmap_dram_bytes'):
            total[k] += stats[k]
        total['num_dense_layers'] += 1

    total['total_dram_bytes'] = (total['ifmap_dram_bytes'] + total['filter_dram_bytes']
                                  + total['ofmap_dram_bytes'])
    return total

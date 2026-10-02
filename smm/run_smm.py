# smm/run_smm.py
"""
SMM (Zouzoula et al., ICPP '24) entry point -- runs the paper's own
baseline (3 fixed ifmap/filter partitions, smm_helpers/baseline.py) and
the Hom/Het policy-managed schemes (smm_helpers/scale_sim_runner.py)
through REAL, cycle-accurate SCALE-Sim simulation for a given model and
GLB size, so SMM sits on the same shared simulator as cosma/ and onsram/
for a fair benchmark comparison -- see smm/docs/smm_verification.md for
why this needed scalesim/memory/smm_reuse_buffers.py (the paper's own
methodology only ever simulates the baseline; Hom/Het are evaluated
analytically in the paper itself).

Model input: either a SCALE-Sim topology CSV directly (e.g.
topologies/conv_nets/Resnet18.csv), or an exported model.json (e.g.
cosma/_exported/MobileNet/model.json) -- converted to a topology CSV via
cosma/helpers/topology_builder.build_topology(), the same converter
COSMA/OnSRAM already use, so this doesn't duplicate that logic. Only
CONV2D/DEPTHWISE_CONV2D/CONV_3D layers are simulated either way -- FC
layers are dropped (same known gap already flagged in
smm/docs/smm_model_roster.md and shared with cosma/onsram's own ports).

Usage:
  python3 smm/run_smm.py --model topologies/conv_nets/Resnet18.csv --glb_kb 64
  python3 smm/run_smm.py --model cosma/_exported/MobileNet/model.json --glb_kb 64 128 256
  python3 smm/run_smm.py --model ... --glb_kb 64 --objective latency --skip-baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smm.smm_helpers.baseline import run_baseline, BASELINE_RATIOS
from smm.smm_helpers.scale_sim_runner import SMMScaleSimRunner

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, 'configs', 'scale_smm.cfg')


def _resolve_topology_csv(model_arg: str, scratch_dir: str) -> str:
    """model_arg is either a .csv topology already, or a model.json to
    convert via cosma's topology_builder (CONV2D/DEPTHWISE_CONV2D/CONV_3D
    layers only -- see module docstring)."""
    if model_arg.endswith('.csv'):
        return model_arg
    if model_arg.endswith('.json'):
        from cosma.helpers.topology_builder import build_topology
        csv_path = os.path.join(scratch_dir, 'smm_topology.csv')
        build_topology(model_arg, csv_path)
        return csv_path
    raise ValueError(f"--model must be a .csv topology or a model.json, got: {model_arg}")


def _run_one_glb(topology_csv: str, config_file: str, glb_kb: int, objective: str,
                  skip_baseline: bool, out_dir: str):
    print(f"\n{'#' * 90}\n# GLB = {glb_kb} kB, objective = {objective}\n{'#' * 90}")

    rows = []

    if not skip_baseline:
        for ratio_name in BASELINE_RATIOS:
            res = run_baseline(topology_csv, config_file, glb_kb, ratio_name)
            rows.append((ratio_name, res.total_cycles, res.total_dram_bytes))

    for homogeneous, label in [(False, 'Het'), (True, 'Hom')]:
        runner = SMMScaleSimRunner(
            topology_file=topology_csv, config_file=config_file, glb_size_kb=glb_kb,
            objective=objective, homogeneous=homogeneous, allow_prefetch=True,
            output_dir=os.path.join(out_dir, f'{label}_{glb_kb}kb'),
            verbose=False, save_ifmap_trace=False, save_filter_trace=False,
            save_ofmap_trace=False,
        )
        runner.run()
        totals = runner.get_actual_totals()
        rows.append((f'{label}_{objective}', totals['total_cycles'], totals['total_dram_bytes']))

    print(f"\n{'scheme':<18} {'cycles':>14} {'dram_bytes':>14} {'dram_MB':>10}")
    best_baseline_bytes = min((b for name, _, b in rows if name.startswith('sa_')), default=None)
    for name, cycles, dram_bytes in rows:
        mb = dram_bytes / (1024 * 1024)
        red = (f"  ({(1 - dram_bytes / best_baseline_bytes) * 100:+.1f}% vs best baseline)"
               if best_baseline_bytes and not name.startswith('sa_') else '')
        print(f"{name:<18} {cycles:>14} {dram_bytes:>14} {mb:>10.2f}{red}")

    return rows


def main():
    p = argparse.ArgumentParser(description='SMM paper benchmark on real SCALE-Sim')
    p.add_argument('--model', required=True, help='Topology CSV or exported model.json')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='SCALE-Sim config (.cfg)')
    p.add_argument('--glb_kb', type=int, nargs='+', default=[64],
                    help='GLB size(s) in kB to sweep (default: 64)')
    p.add_argument('--objective', choices=['accesses', 'latency'], default='accesses')
    p.add_argument('--skip-baseline', action='store_true',
                    help='Skip the 3 fixed-partition baselines (Hom/Het only, faster)')
    p.add_argument('--out', default=os.path.join(HERE, 'results', 'run_smm_out'))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        topology_csv = _resolve_topology_csv(args.model, scratch)
        for glb_kb in args.glb_kb:
            _run_one_glb(topology_csv, args.config, glb_kb, args.objective,
                         args.skip_baseline, args.out)


if __name__ == '__main__':
    main()

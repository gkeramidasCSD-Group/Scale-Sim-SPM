# cosma/run_paper_roster.py
"""
Sweeps every currently-available model from the COSMA paper's own roster
(§V-A1, arXiv:2311.18246) through the real paper-comparable pipeline, one
model at a time, and writes one combined results CSV so numbers are
comparable side by side.

For each model this runs:
  1. Instant bounds (M_R, MPMF-proxy) -- cosma_Ilp.compute_structural_minimum_bytes()
     / compute_mpmf_bytes() via visualize_spm.print_budget_bounds(), same
     functions --bounds-only uses. Free, no ILP solve.
  2. (--true-mpmf only) The paper's real M_P -- a genuine free-schedule ILP
     solve (cosma_Ilp.compute_true_mpmf_bytes()) -- and M_H = (M_R+M_P)/2.
     Off by default: this is the expensive, sometimes very slow part (see
     docs/results_plan.md §6 item 1) -- without it, budgets are
     [M_R, MPMF-proxy] instead of the paper's true [M_R, M_H, M_P].
  3. run_paper_baselines.py, as a **separate subprocess**, at those budgets
     -- the paper's own 4 TFLite-linear x {default,MPMF} x {Belady,
     ILP-greedy} comparison baselines (or just the 2 'default'-schedule
     ones, this script's own default -- see --schedules below), plus the
     cosma_native row. Each
     model's own combined CSV (from run_paper_baselines.py itself) is kept
     under --results-dir, and this script's own output is everything
     concatenated with a header (model/budget_kb/combo/status/
     total_non_compulsory_access_bytes/dram_traffic_reduction_pct/speedup/
     error -- run_paper_baselines.py's own schema, unmodified).

Runs models STRICTLY SEQUENTIALLY, one full subprocess at a time --
**never parallelize multiple models' ILP solves against each other.**
Confirmed 2026-09-29 on a 7GB-RAM machine: running ResNet-50 and
DenseNet-121's solves at once crashed the machine outright; even a single
DenseNet-121 solve alone (1.4M rows / 768K columns before presolve) pushed
memory into ~1.7GB of swap by itself. On a more powerful machine this
restriction can likely be relaxed (run models in parallel yourself if you
trust the RAM headroom), but this script doesn't attempt to detect that
automatically or parallelize on its own -- sequential is always correct,
just not maximally fast. Each model is a fresh subprocess specifically so
one model's ILP/solver memory is fully released before the next starts.

A failure on any one model (export failure, infeasible budget, solver
time-out) is caught, logged, and recorded as an error row -- the sweep
continues to the next model, same resilience pattern as run_experiments.py.

**--schedules defaults to 'default' here (not run_paper_baselines.py's own
'both' default), on purpose.** run_paper_baselines.py's 'mpmf' schedule
combos require a real free-schedule ILP solve internally
(cosma_Ilp.compute_true_mpmf_bytes()) *regardless* of this script's own
--true-mpmf flag above, which only governs which BUDGETS get tested, not
which schedule combos run against them -- confirmed the hard way
2026-09-29: a ResNet-50-alone smoke test with the (then-nonexistent)
--schedules stuck at 'both' burned 30+ min of CPU and ~1.5GB of swap for a
model whose real work (the fixed-schedule solve) takes 0.4s, because
ResNet-50's M_R==MPMF-proxy already (see --list/paper_model_roster.md) so
the real MPMF-schedule ILP could only ever reproduce the default schedule
-- pure wasted solve. Pass --schedules both once you deliberately want the
real 4-combo paper comparison and have the compute budget for it (expect
this to be the slow part on every model, not just ResNet-50).

--true-mpmf (this script's bounds step) and --schedules mpmf/both
(run_paper_baselines.py's own combos) both pay for the *same* underlying
ILP problem independently -- there's no result-sharing between them yet.
Using both together pays for two separate solves of it; that's accepted
here rather than piping a precomputed schedule through, since
run_paper_baselines.py has no such external-schedule input today.

Portability to another machine
-------------------------------
This script and everything it calls need:
  - This repo (cosma/, spm_common/, scalesim/).
  - The trim project's exporter, for any model still given as .tflite --
    spm_common/model_resolver.py's DEFAULT_EXPORTER is hardcoded to
    /home/george/Desktop/trim/python_scripts/export_model.py on THIS
    machine; pass --exporter explicitly on another machine, or export
    every roster model to model.json here first and copy cosma/_exported/
    across instead (no exporter needed at all on the far side then).
  - A Gurobi license, only if using --solver gurobi (recommended -- see
    docs/results_plan.md's Solver row: ~0.4s on ResNet-50 here, matching
    the paper's own 0.296s average, vs. CBC being meaningfully slower on
    anything this size). Gurobi's WLS (Web License Service) credentials
    are portable across machines -- copy ~/gurobi.lic (or set
    WLSACCESSID/WLSSECRET/LICENSEID as env vars) on the new machine; no
    machine-locking involved. Falls back to --solver cbc (bundled with
    PuLP, no license needed) otherwise.

Usage
-----
    python3 run_paper_roster.py                    # every available model, cheap: M_R + MPMF-proxy budgets, default-schedule combos only
    python3 run_paper_roster.py --list              # show the full roster (incl. blocked/not-sourced) and exit
    python3 run_paper_roster.py --models resnet50 densenet121
    python3 run_paper_roster.py --solver gurobi     # recommended whenever a license is available (~0.4s vs. CBC's much slower on anything this size)
    python3 run_paper_roster.py --schedules both --true-mpmf   # the real, full paper comparison -- expect this to be slow (see above)
    python3 run_paper_roster.py --time-limit 1200 --true-mpmf-time-limit 1200

See docs/paper_model_roster.md for what's available/blocked/not-sourced and why.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spm_common import graph_builder, model_resolver
import visualize_spm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')
DEFAULT_LOGS_DIR = os.path.join(HERE, 'logs')

# The paper's own 10 human-designed models (§V-A1) plus DenseNet-121's exact
# variant assumption -- see docs/paper_model_roster.md for the full rationale
# and how each status was determined. Kept here (not just in the .md) because
# the script needs real paths to run; update both together.
ROSTER = [
    {'name': 'resnet50', 'category': 'image classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', '_exported_resnet50-tflite-float', 'model.json')},
    {'name': 'resnet50_int8', 'category': 'image classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'models_resnet50_int8_resnet50', 'model.json'),
     'note': 'INT8, quantized via cosma/tools/quantize_model.py -- matches the paper\'s own 8-bit setting.'},
    {'name': 'densenet121', 'category': 'image classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'densenet', 'model.json'),
     'note': ('Paper\'s own citation for "DenseNet" ([13]) is Jegou et al.\'s Tiramisu/'
              'FC-DenseNet (a segmentation architecture), not classification DenseNet -- '
              'likely a citation slip, but unresolved. This is tf.keras.applications.'
              'DenseNet121, our best reasonable guess, not a confirmed match.')},
    {'name': 'densenet121_int8', 'category': 'image classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'models_densenet121_int8_densenet121', 'model.json'),
     'note': 'INT8 variant of the above -- same DenseNet-identity caveat applies.'},
    {'name': 'deeplabv3', 'category': 'semantic segmentation', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', '_exported_deeplabv3_DeepLabV3-Plus-MobileNet', 'model.json'),
     'note': ('Export unblocked 2026-09-30 (RESIZE_BILINEAR support added). Input is '
              '[1,513,513,3], NOT the paper\'s (1,3,224,224) -- externally-sourced .tflite, '
              'not directly comparable to the paper\'s own numbers on this axis. Also very '
              'memory-heavy for SCALE-Sim\'s real simulation on a 7GB machine (killed as a '
              'precaution here after ~240s, available memory dropping to ~880MB) -- a more '
              'powerful machine is exactly what this note is for.')},
    {'name': 'resnext', 'category': 'image classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'resnext50_resnext50_tflite_resnext50_float32', 'model.json'),
     'note': ('Real result already obtained on this machine, both M_R=9408KB and '
              'MPMF=9636KB: baselines fail outright (fragmentation), COSMA solves Optimal. '
              'See docs/paper_model_roster.md.')},
    {'name': 'r2plus1d', 'category': 'video classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'r2plus1d_correct_r2plus1d_18_tflite_r2plus1d_18_float32', 'model.json'),
     'note': ('Correct-resolution export ([1,16,224,224,3], matching the paper exactly) as '
              'of 2026-09-30 -- the earlier _exported/r2plus1d_18/ was wrong-resolution '
              '([1,8,112,112,3]) and should not be used. M_R=225792KB, MPMF=275968KB. '
              'SCALE-Sim could not complete a real simulation at this resolution on this '
              '7GB machine (killed twice as a memory-safety precaution, ~900MB/~470MB '
              'available after 40-50 min each) -- this is the primary reason a more '
              'powerful machine is worth using.')},
    {'name': 's3d', 'category': 'video classification', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 's3d_s3d_tflite_s3d_float32', 'model.json'),
     'note': 'Real result already obtained here (degenerate M_R==MPMF case, Optimal, 98.7%, 3.99x).'},
    {'name': 'fcn', 'category': 'semantic segmentation', 'status': 'available',
     'model_path': os.path.join(HERE, '_exported', 'fcn_fcn_tflite_fcn_float32', 'model.json'),
     'note': ('The paper\'s one documented exception where all 4 baselines tie COSMA at '
              'M_P -- worth checking directly, NOT yet done (export/bounds only). '
              'SCALE-Sim could not complete a real run on this 7GB machine (dilated '
              'backbone layers, killed as a precaution after ~1040s, available memory '
              'dropping to ~400MB) -- exactly the kind of model a more powerful machine '
              'is for.')},
    {'name': 'l_raspp', 'category': 'semantic segmentation', 'status': 'not_sourced'},
    {'name': 'transformer', 'category': 'transformer-based', 'status': 'not_sourced'},
    {'name': 'vit', 'category': 'transformer-based', 'status': 'not_sourced'},
]


def print_roster() -> None:
    by_status = {'available': [], 'blocked': [], 'not_sourced': []}
    for m in ROSTER:
        by_status[m['status']].append(m)
    print(f"COSMA paper roster: {len(ROSTER)} human-designed models "
          f"({len(by_status['available'])} available, {len(by_status['blocked'])} blocked, "
          f"{len(by_status['not_sourced'])} not sourced). "
          f"4 NAS-generated models (PNASNet-5/AmoebaNet-D/NASNet-A/DARTS) are permanently "
          f"out of scope (no divide-and-conquer heuristic) and aren't listed here.\n")
    for status, label in (('available', 'AVAILABLE'), ('blocked', 'BLOCKED'),
                          ('not_sourced', 'NOT SOURCED')):
        print(f"-- {label} --")
        for m in by_status[status]:
            print(f"  {m['name']:14s} ({m['category']})"
                  + (f" -- {m['note']}" if m.get('note') else ""))
        print()


def _roster_by_name(names) -> list:
    lookup = {m['name']: m for m in ROSTER}
    selected = []
    for n in names:
        if n not in lookup:
            raise ValueError(f"Unknown model {n!r} -- known names: {sorted(lookup)}")
        m = lookup[n]
        if m['status'] != 'available':
            raise ValueError(f"{n!r} is not available ({m['status']}"
                              + (f": {m['note']}" if m.get('note') else "") + ")")
        selected.append(m)
    return selected


def _budgets_for(model_path: str, exporter: str, export_dir: str, force_export: bool,
                  true_mpmf: bool, true_mpmf_time_limit, solver: str) -> tuple:
    """Returns (resolved_model_json_path, [budget_kb, ...])."""
    resolved = model_resolver.resolve_model_json(model_path, exporter=exporter,
                                                   export_dir=export_dir,
                                                   force_export=force_export)
    nodes, tensors = graph_builder.load_graph(resolved)
    bounds = visualize_spm.print_budget_bounds(nodes, tensors)
    m_r = bounds['structural_minimum_bytes']
    mpmf_proxy = bounds['mpmf_bytes']

    if not true_mpmf:
        budgets = sorted({m_r, mpmf_proxy})
        return resolved, [b / 1024 for b in budgets]

    true_mp = visualize_spm.print_true_mpmf(nodes, tensors, m_r,
                                             time_limit_sec=true_mpmf_time_limit,
                                             solver=solver)
    budgets = sorted({m_r, true_mp['m_h_bytes'], true_mp['true_mpmf_bytes']})
    return resolved, [b / 1024 for b in budgets]


def run_one_model(model: dict, args) -> dict:
    """Returns {'name', 'ok', 'csv_path' or 'error'}."""
    name = model['name']
    print(f"\n{'=' * 70}\n{name} ({model['category']})\n{'=' * 70}")
    try:
        resolved, budgets_kb = _budgets_for(
            model['model_path'], args.exporter, args.export_dir, args.force_export,
            args.true_mpmf, args.true_mpmf_time_limit, args.solver)
    except Exception as e:
        print(f"[{name}] FAILED during export/bounds: {e}")
        return {'name': name, 'ok': False, 'error': f'export/bounds: {e}'}

    print(f"[{name}] budgets (KB): {budgets_kb}")
    out_csv = os.path.join(args.results_dir, f"{name}_paper_roster.csv")
    log_path = os.path.join(args.logs_dir, f"{name}_paper_roster.log")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    cmd = [sys.executable, os.path.join(HERE, 'run_paper_baselines.py'),
           '--model-json', resolved,
           '--budgets-kb', *[str(b) for b in budgets_kb],
           '--solver', args.solver,
           '--schedules', args.schedules,
           '--time-limit', str(args.time_limit),
           '--out-csv', out_csv,
           '--logs-dir', args.logs_dir]
    if args.no_plots:
        cmd.append('--no-plots')
    if args.free_schedule:
        cmd.append('--free-schedule')

    t0 = time.time()
    with open(log_path, 'w') as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[{name}] subprocess exit {result.returncode} in {elapsed:.1f}s "
          f"-- full log: {log_path}")

    if result.returncode != 0 or not os.path.exists(out_csv):
        return {'name': name, 'ok': False,
                'error': f'run_paper_baselines.py exit {result.returncode}, see {log_path}'}
    return {'name': name, 'ok': True, 'csv_path': out_csv}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=None,
                         help="Roster names to run (default: every 'available' model). "
                              "See --list for names.")
    parser.add_argument('--list', action='store_true',
                         help='Print the full roster (available/blocked/not-sourced) and exit.')
    parser.add_argument('--config', default=None,
                         help="Passed through to run_paper_baselines.py's own default if unset.")
    parser.add_argument('--exporter', default=model_resolver.DEFAULT_EXPORTER)
    parser.add_argument('--export-dir', default=model_resolver.DEFAULT_EXPORT_DIR)
    parser.add_argument('--force-export', action='store_true')
    parser.add_argument('--true-mpmf', action='store_true',
                         help='Solve the real M_P/M_H per model (slow, see module docstring). '
                              'Default: use [M_R, MPMF-proxy] only.')
    parser.add_argument('--true-mpmf-time-limit', type=float, default=900,
                         help='Time limit (seconds) for the true-M_P bounds solve. Default: 900.')
    parser.add_argument('--time-limit', type=float, default=900,
                         help="Passed to run_paper_baselines.py's own --time-limit "
                              "(MPMF schedule + cosma_native solves). Default: 900.")
    parser.add_argument('--solver', choices=['cbc', 'gurobi'], default='gurobi',
                         help="ILP solver backend (default: gurobi, matching every other "
                              "entry point's CLI default -- run_cosma.py/run_paper_baselines.py/"
                              "run_experiments.py/visualize_spm.py. Falls back to cbc "
                              "automatically only if you pass --solver cbc explicitly; no "
                              "license needed for cbc.")
    parser.add_argument('--schedules', choices=['default', 'mpmf', 'both'], default='default',
                         help="Forwarded to run_paper_baselines.py's own --schedules. Default "
                              "here is 'default' (cheap, skips the real MPMF-schedule ILP "
                              "solve) -- see module docstring for why this differs from "
                              "run_paper_baselines.py's own 'both' default.")
    parser.add_argument('--no-plots', action='store_true', default=True,
                         help='On by default here (a roster sweep produces a lot of plots '
                              'otherwise); pass --plots to re-enable.')
    parser.add_argument('--plots', action='store_false', dest='no_plots')
    parser.add_argument('--free-schedule', action='store_true',
                         help="Forwarded to run_paper_baselines.py's own --free-schedule -- lets "
                              "each model's cosma_native row choose its own operator schedule "
                              "(the paper's full joint optimization) instead of the Fixed-"
                              "Schedule sub-problem this project ran by default until now. Off "
                              "by default. Never validated at roster scale before this flag "
                              "existed -- expect real solve-time cost on larger models.")
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--logs-dir', default=DEFAULT_LOGS_DIR)
    parser.add_argument('--out-csv', default=os.path.join(DEFAULT_RESULTS_DIR, 'paper_roster_summary.csv'))
    args = parser.parse_args()

    if args.list:
        print_roster()
        return

    if args.config:
        os.environ.setdefault('COSMA_SCALE_CFG', args.config)  # informational only; run_paper_baselines.py takes --config directly if ever wired through here

    models = _roster_by_name(args.models) if args.models else [
        m for m in ROSTER if m['status'] == 'available']
    if not models:
        print("No available models to run -- pass --models explicitly or see --list.")
        return

    print(f"Running {len(models)} model(s) sequentially: {[m['name'] for m in models]}")
    print("(one model's full subprocess at a time -- see module docstring on why)")

    outcomes = [run_one_model(m, args) for m in models]

    combined_rows = []
    fieldnames = None
    for o in outcomes:
        if not o['ok']:
            continue
        with open(o['csv_path'], newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            combined_rows.extend(reader)

    if combined_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"\nWrote {len(combined_rows)} combined rows to {args.out_csv}")

    print("\n--- Summary ---")
    for o in outcomes:
        if o['ok']:
            print(f"  {o['name']:14s} OK   -> {o['csv_path']}")
        else:
            print(f"  {o['name']:14s} FAIL -> {o['error']}")


if __name__ == '__main__':
    main()

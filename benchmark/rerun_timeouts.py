#!/usr/bin/env python3
"""Re-run only the sweep runs that timed out, without the time limit, so every
(model, config) pair gets a real speedup instead of a lower bound.

Put this file in the sweep's benchmark/ folder (next to run_sweep.py): it calls
run_sweep.execute_task(), so each re-run uses exactly the same command, config
template, COMPUTE_REPORT parsing and trace cleanup as the original sweep.

What it runs: every (model, combo, version, repeat) whose LATEST row in the sweep
CSV has status "timeout" -- normally vanilla, plus the few configs where the
optimized version timed out too (those run first, since they are shorter).
Vanilla runs are ordered by how long the optimized run of the same config took,
cheapest first, so results arrive steadily.

Resumable: results go to <out-root>/results_rerun.csv; a run already recorded
there with status "ok" is skipped when you start the script again.
At the end (or with --merge-only) it writes <out-root>/results_merged.csv: the
original sweep with each timed-out row replaced by its finished re-run, ready for
paper/make_sweep_figures.py.

These runs are long: every vanilla run here already took more than the sweep's
limit (2 h), and the slowest may need a day or more. Use --dry-run first.

Example (server):
  cd /data/grizos/Scale-Sim-SPM
  python3 benchmark/rerun_timeouts.py \
      --results-csv    benchmarks/results_new/results_raw.csv \
      --vanilla-repo   /data/grizos/SCALE-Sim \
      --optimized-repo /data/grizos/Scale-Sim-SPM \
      --vanilla-venv-python   /data/grizos/SCALE-Sim/venv/bin/python \
      --optimized-venv-python /data/grizos/Scale-Sim-SPM/venv/bin/python \
      --jobs 8 --dry-run
  # then the same command without --dry-run, under nohup or tmux
"""
import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sweep  # noqa: E402  (same folder: benchmark/)

INT_FIELDS = {"array_size", "sram_kb", "ifmap_offset", "filter_offset", "ofmap_offset", "rng_seed"}


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def latest_rows(csv_path):
    """(model, combo_id, version, repeat_idx) -> the most recent row for it."""
    rows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["combo_id"], r["version"], r["repeat_idx"])
            if key not in rows or r["timestamp"] > rows[key]["timestamp"]:
                rows[key] = r
    return rows


def build_tasks(rows, statuses, only):
    tasks = []
    for (model, combo_id, version, rep), r in rows.items():
        if r["status"] not in statuses:
            continue
        other_version = "optimized" if version == "vanilla" else "vanilla"
        other = rows.get((model, combo_id, other_version, rep), {})
        task = dict(
            phase=r["phase"], model=model, combo_id=combo_id, version=version,
            repeat_idx=int(rep), run_id=run_sweep.make_run_id(model, combo_id, version, rep),
            combo={k: int(r[k]) if k in INT_FIELDS else r[k] for k in run_sweep.COMBO_FIELDS},
            prev_seconds=float(r["wall_seconds"] or 0),
            other_status=other.get("status", "?"),
            other_seconds=float(other.get("wall_seconds") or 0),
        )
        if only and not run_sweep.task_matches(task, only):
            continue
        tasks.append(task)
    # optimized re-runs first (shorter, and their vanilla partner needs them anyway),
    # then vanilla, cheapest first by the optimized run of the same config
    tasks.sort(key=lambda t: (t["version"] != "optimized",
                              t["other_seconds"] if t["other_status"] == "ok" else float("inf")))
    return tasks


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def merge(results_csv, rerun_csv, merged_csv):
    """Original sweep with each re-run that finished replacing the row it re-ran."""
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        original = list(reader)
    finished = {}
    if os.path.isfile(rerun_csv):
        for r in latest_rows(rerun_csv).values():
            if r["status"] == "ok":
                finished[(r["model"], r["combo_id"], r["version"], r["repeat_idx"])] = r
    latest = {}  # key -> index of that run's most recent row in `original`
    for i, r in enumerate(original):
        key = (r["model"], r["combo_id"], r["version"], r["repeat_idx"])
        if key not in latest or r["timestamp"] > original[latest[key]]["timestamp"]:
            latest[key] = i
    out, replaced = [], 0
    for i, r in enumerate(original):
        key = (r["model"], r["combo_id"], r["version"], r["repeat_idx"])
        if latest[key] != i:
            continue                      # older duplicate of this run: drop it
        if key in finished and r["status"] != "ok":
            out.append({k: finished[key].get(k, "") for k in fieldnames})
            replaced += 1
        else:
            out.append(r)
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    still = sum(1 for r in out if r["status"] != "ok")
    log(f"merged: {replaced} timed-out rows replaced, {still} rows still not ok -> {merged_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-csv", required=True, help="the sweep's results CSV (results_raw.csv)")
    ap.add_argument("--vanilla-repo", required=True)
    ap.add_argument("--optimized-repo", required=True)
    ap.add_argument("--vanilla-venv-python", required=True)
    ap.add_argument("--optimized-venv-python", required=True)
    ap.add_argument("--out-root", default=None,
                    help="where re-run cfgs/runs/logs and CSVs go [<results-csv dir>/rerun_timeouts]")
    ap.add_argument("--jobs", type=int, default=4, help="runs at the same time [4]")
    ap.add_argument("--timeout-h", type=float, default=0,
                    help="per-run cap in hours; 0 = no limit [0]")
    ap.add_argument("--min-free-mb", type=int, default=16000,
                    help="wait to START a run until this much RAM is available [16000]; "
                         "running jobs are never killed")
    ap.add_argument("--include-failed", action="store_true",
                    help="also re-run rows with status failed/parse_error")
    ap.add_argument("--only", default=None,
                    help="filter, same syntax as run_sweep.py: e.g. model=vgg16,version=vanilla")
    ap.add_argument("--dry-run", action="store_true", help="list what would run and exit")
    ap.add_argument("--merge-only", action="store_true", help="only (re)write results_merged.csv")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(os.path.dirname(os.path.abspath(args.results_csv)),
                                             "rerun_timeouts")
    os.makedirs(out_root, exist_ok=True)
    rerun_csv = os.path.join(out_root, "results_rerun.csv")
    merged_csv = os.path.join(out_root, "results_merged.csv")
    if args.merge_only:
        merge(args.results_csv, rerun_csv, merged_csv)
        return

    statuses = {"timeout"} | ({"failed", "parse_error"} if args.include_failed else set())
    only = run_sweep.parse_only(args.only) if args.only else None
    tasks = build_tasks(latest_rows(args.results_csv), statuses, only)
    done = run_sweep.load_done_run_ids(rerun_csv)
    todo = [t for t in tasks if t["run_id"] not in done]

    log(f"{len(tasks)} timed-out runs found, {len(tasks) - len(todo)} already re-run, "
        f"{len(todo)} to go; jobs={args.jobs}, "
        f"limit={'none' if not args.timeout_h else f'{args.timeout_h} h'}")
    for t in todo:
        partner = (f"{t['other_seconds']:7.0f} s" if t["other_status"] == "ok" else f"{t['other_status']:>9s}")
        print(f"  {t['version']:9s} {t['model']:12s} {t['combo_id']:20s} "
              f"was > {t['prev_seconds'] / 3600:.1f} h | other version: {partner}")
    if args.dry_run or not todo:
        if not todo:
            merge(args.results_csv, rerun_csv, merged_csv)
        return

    repos = {"vanilla": args.vanilla_repo, "optimized": args.optimized_repo}
    venvs = {"vanilla": args.vanilla_venv_python, "optimized": args.optimized_venv_python}
    static_meta = {v: run_sweep.capture_static_meta(venvs[v]) for v in ("vanilla", "optimized")}
    timeout_s = args.timeout_h * 3600 if args.timeout_h else None

    write_lock, start_lock = threading.Lock(), threading.Lock()
    new_file = not (os.path.isfile(rerun_csv) and os.path.getsize(rerun_csv) > 0)
    f = open(rerun_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=run_sweep.FIELDNAMES)
    if new_file:
        writer.writeheader()
        f.flush()

    stop = threading.Event()  # set on Ctrl-C: no new runs start after that

    def work(task):
        with start_lock:  # one start at a time, so the RAM check sees the previous job's growth
            while not stop.is_set():
                free = mem_available_mb()
                if free is None or free >= args.min_free_mb:
                    break
                log(f"waiting for RAM before {task['run_id']} ({free} MB free < {args.min_free_mb} MB)")
                stop.wait(60)
            if stop.is_set():
                return task, None
            log(f"start  {task['run_id']}")
            time.sleep(5)
        row = run_sweep.execute_task(task, repos, venvs, out_root, timeout_s, static_meta)
        if row["returncode"] in (-1, -2, -15):
            # killed by hang-up / Ctrl-C / SIGTERM, not a simulator failure; retried next start
            row["status"] = "interrupted"
        with write_lock:
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        return task, row

    # No `with` block: on Python 3.8 its exit waits for every queued run, so Ctrl-C would
    # keep starting new runs. Queued runs are cancelled explicitly instead.
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    futures = [pool.submit(work, t) for t in todo]
    try:
        for n, fut in enumerate(as_completed(futures), start=1):
            task, row = fut.result()
            if row is None:
                continue
            extra = ""
            if row["status"] == "ok" and task["version"] == "vanilla" and task["other_status"] == "ok":
                extra = f"  speedup {float(row['wall_seconds']) / task['other_seconds']:.2f}x"
            log(f"[{n}/{len(todo)}] done {task['run_id']}: {float(row['wall_seconds']) / 3600:.2f} h, "
                f"status={row['status']}{extra}")
    except KeyboardInterrupt:
        stop.set()
        for fut in futures:
            fut.cancel()
        log("Ctrl-C: no new runs will start; waiting for the running ones to exit "
            "(Ctrl-C in a terminal stops them too). Start the script again to resume.")
    pool.shutdown(wait=True)
    f.close()
    merge(args.results_csv, rerun_csv, merged_csv)


if __name__ == "__main__":
    main()

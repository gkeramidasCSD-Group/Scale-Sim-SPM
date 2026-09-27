#!/usr/bin/env python3
"""Compare profiles/<prefix>vanilla.pstats against profiles/<prefix>optimized.pstats.

usage: compare.py              -> the 3-layer MobileNet profiles (vanilla.pstats / optimized.pstats)
       compare.py <name>_      -> a run_network.sh run, e.g. compare.py googlenet_a16_s16_calc_

Prints a table of every function the optimization touched (plus the callers/callees whose
numbers move because of it), and writes the full cProfile text reports, sorted by tottime and by
cumtime, to profiles/<version>_{tottime,cumtime}.txt.

Functions are matched by (file relative to scalesim/, function name), not by line number,
because the edits shifted line numbers between the two checkouts.
"""
import os
import pstats
import sys

D = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(D, "profiles")

# (fix id, file suffix, function) -- fix ids match CPROFILE_VANILLA_VS_OPTIMIZED.md
ROWS = [
    ("#2", "compute/systolic_compute_ws.py", "create_ifmap_prefetch_mat"),
    ("#3", "memory/read_buffer_estimate_bw.py", "check_hit"),
    ("#3", "memory/read_buffer_estimate_bw.py", "manage_prefetches"),
    ("#3", "memory/read_buffer_estimate_bw.py", "service_reads"),
    ("#5", "memory/read_buffer_estimate_bw.py", "prefetch"),
    ("#5", "memory/read_buffer_estimate_bw.py", "_finalize_trace_matrix"),
    ("#4a", "memory/read_buffer.py", "active_buffer_hit"),
    ("#4a", "memory/read_buffer.py", "<genexpr>"),
    ("#4a", "memory/read_buffer.py", "prepare_hashed_buffer"),
    ("#4b", "memory/read_buffer.py", "set_fetch_matrix"),
    ("#1", "memory/read_buffer.py", "service_reads"),
    ("#5", "memory/read_buffer.py", "new_prefetch"),
    ("#5", "memory/read_buffer.py", "_finalize_trace_matrix"),
    ("#1", "memory/write_buffer.py", "service_writes"),
    ("#6", "memory/write_buffer.py", "append_to_trace_mat"),
    ("#6", "memory/write_buffer.py", "_append_chunk_to_trace_matrix"),
    ("#1", "tqdm/std.py", "__init__"),
    ("--", "memory/double_buffered_scratchpad_mem.py", "service_memory_requests"),
]


def agg(stats, file_suffix, func):
    nc = tt = ct = 0
    for (f, _line, fn), (_cc, n, t, c, _callers) in stats.stats.items():
        if fn == func and f.endswith(file_suffix):
            nc += n; tt += t; ct += c
    return nc, tt, ct


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    st = {v: pstats.Stats(os.path.join(P, f"{prefix}{v}.pstats")) for v in ("vanilla", "optimized")}
    van, opt = st["vanilla"], st["optimized"]
    print(f"vanilla   : {van.total_tt:7.2f}s  {van.total_calls:>12,} calls")
    print(f"optimized : {opt.total_tt:7.2f}s  {opt.total_calls:>12,} calls")
    print(f"speedup   : {van.total_tt / opt.total_tt:.2f}x\n")

    hdr = f"{'fix':4s} {'function':60s} | {'calls':>10s} {'tot':>6s} {'cum':>6s} | {'calls':>10s} {'tot':>6s} {'cum':>6s}"
    print(f"{'':65s} | {'----- vanilla -----':^24s} | {'---- optimized ----':^24s}")
    print(hdr)
    print("-" * len(hdr))
    for fix, fs, fn in ROWS:
        a, b = agg(van, fs, fn), agg(opt, fs, fn)
        name = f"{fs.split('/')[-1]}::{fn}"
        print(f"{fix:4s} {name:60s} | {a[0]:>10,} {a[1]:6.2f} {a[2]:6.2f} | {b[0]:>10,} {b[1]:6.2f} {b[2]:6.2f}")

    for v, s in st.items():
        for key, sort in (("tottime", "tottime"), ("cumtime", "cumulative")):
            path = os.path.join(P, f"{prefix}{v}_{key}.txt")
            with open(path, "w") as f:
                pstats.Stats(os.path.join(P, f"{prefix}{v}.pstats"), stream=f).sort_stats(sort).print_stats(40)
    print(f"\nfull reports written to {P}/{prefix}<version>_{{tottime,cumtime}}.txt")


if __name__ == "__main__":
    main()

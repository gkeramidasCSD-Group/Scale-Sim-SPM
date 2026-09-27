#!/usr/bin/env python3
"""Percentage breakdown of where a SCALE-Sim run spends its time, vanilla vs. optimized.

usage: breakdown.py <vanilla.pstats> <optimized.pstats> [--top N]

Raw cProfile tottime scatters library time (numpy, dict/set builtins, savetxt internals) over
hundreds of rows. Here it is charged to the SCALE-Sim function that called it, using each
function's "exclusive" time:

    exclusive(F) = cumtime(F) - sum of cumtime of the tracked functions F called directly

"Tracked" means code in scalesim/, tqdm, or the profiling driver. Exclusive times of all tracked
functions add up to the whole profiled run, and each one is grouped into a simulator-level
category (the table in CATEGORIES below).
"""
import argparse
import os
import pstats

# (category, file basename, function name or None for "any function in that file")
# First match wins, so specific rows come before whole-file rows.
CATEGORIES = [
    ("Per-cycle memory loop + SRAM trace assembly", "double_buffered_scratchpad_mem.py", "service_memory_requests"),
    ("Read hit/miss + prefetch decisions (CALC)", "read_buffer_estimate_bw.py", "service_reads"),
    ("Read hit/miss + prefetch decisions (CALC)", "read_buffer_estimate_bw.py", "manage_prefetches"),
    ("Read hit/miss + prefetch decisions (CALC)", "read_buffer_estimate_bw.py", "check_hit"),
    ("Read hit/miss (USER)", "read_buffer.py", "service_reads"),
    ("Read hit/miss (USER)", "read_buffer.py", "active_buffer_hit"),
    ("Read hit/miss (USER)", "read_buffer.py", "<genexpr>"),
    ("Read hit/miss (USER)", "read_buffer.py", "prefetch_active_buffer"),
    ("USER read-buffer setup", "read_buffer.py", "set_fetch_matrix"),
    ("USER read-buffer setup", "read_buffer.py", "prepare_hashed_buffer"),
    ("Write-buffer servicing", "write_buffer.py", "service_writes"),
    ("Write-buffer servicing", "write_buffer.py", "empty_drain_buf"),
    ("Write-buffer servicing", "write_buffer.py", "empty_all_buffers"),
    ("DRAM trace building (in memory)", "read_buffer_estimate_bw.py", "prefetch"),
    ("DRAM trace building (in memory)", "read_buffer_estimate_bw.py", "complete_all_prefetches"),
    ("DRAM trace building (in memory)", "read_buffer_estimate_bw.py", "_finalize_trace_matrix"),
    ("DRAM trace building (in memory)", "read_buffer.py", "new_prefetch"),
    ("DRAM trace building (in memory)", "read_buffer.py", "_finalize_trace_matrix"),
    ("DRAM trace building (in memory)", "write_buffer.py", "store_to_trace_mat_cache"),
    ("DRAM trace building (in memory)", "write_buffer.py", "append_to_trace_mat"),
    ("DRAM trace building (in memory)", "write_buffer.py", "_append_chunk_to_trace_matrix"),
    ("Trace CSV writing (savetxt)", None, "print_trace"),
    ("Trace CSV writing (savetxt)", "double_buffered_scratchpad_mem.py", "print_ifmap_sram_trace"),
    ("Trace CSV writing (savetxt)", "double_buffered_scratchpad_mem.py", "print_filter_sram_trace"),
    ("Trace CSV writing (savetxt)", "double_buffered_scratchpad_mem.py", "print_ofmap_sram_trace"),
    ("Trace CSV writing (savetxt)", "single_layer_sim.py", "save_traces"),
    ("Prefetch-matrix diagonal flatten", None, "create_ifmap_prefetch_mat"),
    ("Prefetch-matrix diagonal flatten", None, "create_filter_prefetch_mat"),
    ("Operand/demand matrix generation", "operand_matrix.py", None),
    ("Operand/demand matrix generation", "systolic_compute_ws.py", None),
    ("Operand/demand matrix generation", "systolic_compute_os.py", None),
    ("Operand/demand matrix generation", "systolic_compute_is.py", None),
    ("Report statistics (start/stop cycles, counts)", "single_layer_sim.py", "calc_report_data"),
    ("Report statistics (start/stop cycles, counts)", "double_buffered_scratchpad_mem.py", None),
    ("Report statistics (start/stop cycles, counts)", "read_buffer_estimate_bw.py", None),
    ("Report statistics (start/stop cycles, counts)", "read_buffer.py", None),
    ("Report statistics (start/stop cycles, counts)", "write_buffer.py", None),
    ("tqdm progress-bar objects", "@tqdm", None),
]
OTHER = "Other (config/topology parsing, report files, driver)"


def tracked(filename):
    return "/scalesim/" in filename or "/tqdm/" in filename or "cprofile_compare" in filename


def categorize(filename, func):
    base = os.path.basename(filename)
    for cat, fbase, fname in CATEGORIES:
        if fbase == "@tqdm":
            if "/tqdm/" in filename:
                return cat
            continue
        if fbase is not None and fbase != base:
            continue
        if fname is not None and fname != func:
            continue
        if "/scalesim/" not in filename:
            continue
        return cat
    return OTHER


def exclusive_times(stats):
    """{(file, line, func): exclusive seconds} for every tracked function."""
    excl = {}
    for key, (_cc, _nc, _tt, ct, _callers) in stats.stats.items():
        if tracked(key[0]):
            excl[key] = ct

    def subtract_up(caller, amount, depth=0):
        # A tracked function can be reached through library frames, e.g.
        # active_buffer_hit -> builtins.any -> <genexpr>, or tqdm.__hash__ called by set.add.
        # Walk up through untracked frames, splitting by cumtime share, to the tracked caller(s).
        if caller in excl:
            excl[caller] -= amount
            return
        if depth > 6 or caller not in stats.stats:
            return
        up = stats.stats[caller][4]
        total = sum(cs[3] for cs in up.values())
        if total <= 0:
            return
        for gc, cs in up.items():
            subtract_up(gc, amount * cs[3] / total, depth + 1)

    for key, (_cc, _nc, _tt, _ct, callers) in stats.stats.items():
        if not tracked(key[0]):
            continue
        for caller, cstat in callers.items():
            if caller != key:
                subtract_up(caller, cstat[3])  # cumtime of calls from this caller
    return excl


def summarize(path):
    st = pstats.Stats(path)
    excl = exclusive_times(st)
    cats, funcs = {}, {}
    for (f, line, fn), t in excl.items():
        c = categorize(f, fn)
        cats[c] = cats.get(c, 0.0) + t
        if "/tqdm/" in f:
            label = "tqdm (all functions)"
        elif fn == "<genexpr>" and os.path.basename(f) == "read_buffer.py":
            label = "read_buffer.py::active_buffer_hit"  # its any(...) generator, the only one there
        else:
            label = f"{os.path.basename(f)}::{fn}"
        funcs[label] = funcs.get(label, 0.0) + t
    return st.total_tt, cats, funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vanilla")
    ap.add_argument("optimized")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    v_tot, v_cat, v_fn = summarize(args.vanilla)
    o_tot, o_cat, o_fn = summarize(args.optimized)
    print(f"vanilla total {v_tot:.1f}s, optimized total {o_tot:.1f}s, speedup {v_tot/o_tot:.2f}x")
    print(f"(categories cover {sum(v_cat.values())/v_tot*100:.1f}% / {sum(o_cat.values())/o_tot*100:.1f}% of each run)\n")

    order = [c for c, _, _ in CATEGORIES] + [OTHER]
    order = sorted(dict.fromkeys(order), key=lambda c: -v_cat.get(c, 0.0))
    saved_total = v_tot - o_tot
    print("| Category | Vanilla s | Vanilla % | Optimized s | Optimized % | Saved s | Share of saving |")
    print("|---|---|---|---|---|---|---|")
    for c in order:
        a, b = v_cat.get(c, 0.0), o_cat.get(c, 0.0)
        if a < 0.005 and b < 0.005:
            continue
        print(f"| {c} | {a:.1f} | {a/v_tot*100:.1f}% | {b:.1f} | {b/o_tot*100:.1f}% | {a-b:.1f} | {(a-b)/saved_total*100:.1f}% |")
    print(f"| **Total** | **{v_tot:.1f}** | 100% | **{o_tot:.1f}** | 100% | **{saved_total:.1f}** | 100% |")

    for name, tot, fn in (("vanilla", v_tot, v_fn), ("optimized", o_tot, o_fn)):
        print(f"\nTop {args.top} functions by exclusive time -- {name}")
        print("| Function | Seconds | % of run |")
        print("|---|---|---|")
        for label, t in sorted(fn.items(), key=lambda x: -x[1])[:args.top]:
            print(f"| `{label}` | {t:.1f} | {t/tot*100:.1f}% |")


if __name__ == "__main__":
    main()

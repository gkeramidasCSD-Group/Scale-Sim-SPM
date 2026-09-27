#!/usr/bin/env python3
"""Build the paper figures and LaTeX tables for the SCALE-Sim performance work.

Inputs (all already in the repo):
  ../profiles/googlenet_a16_s16_calc_{vanilla,optimized}.pstats   CALC profile (laptop)
  ../profiles/googlenet_a32_s32_user_{vanilla,optimized}.pstats   USER profile (server)
  ../../../results.csv                                            un-profiled server sweep

Outputs (written next to this script):
  fig_time_breakdown.{pdf,png}   where the time goes, vanilla vs. optimized, per mode
  fig_speedup.{pdf,png}          end-to-end speedup across models and configurations
  tables.tex                     tab:optimizations, tab:breakdown, tab:speedup, tab:walltime
  numbers.txt                    every number the text quotes, for checking

usage: python3 make_paper_assets.py
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(HERE, "..", "profiles")
RESULTS = os.path.join(HERE, "..", "..", "..", "results.csv")
sys.path.insert(0, os.path.join(HERE, ".."))
import breakdown  # noqa: E402

WORKLOADS = [
    ("calc", "googlenet_a16_s16_calc", "CALC", "GoogLeNet, 16×16 array, 16 KB SRAMs"),
    ("user", "googlenet_a32_s32_user", "USER", "GoogLeNet, 32×32 array, 32 KB SRAMs"),
]

# breakdown.py's detailed categories, folded into the six a reader needs.
PAPER_CATEGORIES = [
    ("Hit/miss checks", ["Read hit/miss + prefetch decisions (CALC)", "Read hit/miss (USER)"]),
    ("DRAM trace construction", ["DRAM trace building (in memory)"]),
    ("Trace file output", ["Trace CSV writing (savetxt)"]),
    ("Cycle loop & write buffer", ["Per-cycle memory loop + SRAM trace assembly", "Write-buffer servicing"]),
    ("Progress-bar objects", ["tqdm progress-bar objects"]),
    ("Setup & other", ["USER read-buffer setup", "Prefetch-matrix diagonal flatten",
                       "Operand/demand matrix generation",
                       "Report statistics (start/stop cycles, counts)", breakdown.OTHER]),
]

# Which functions each optimization touched (exclusive-time labels from breakdown.summarize).
# read_buffer.service_reads / write_buffer.service_writes changed only by losing tqdm (#1).
FIXES = [
    ("1", "Per-cycle progress-bar objects", ["tqdm (all functions)", "read_buffer.py::service_reads",
                                              "write_buffer.py::service_writes"]),
    ("2", "Prefetch-order diagonal flatten", ["systolic_compute_ws.py::create_ifmap_prefetch_mat",
                                              "systolic_compute_os.py::create_ifmap_prefetch_mat",
                                              "systolic_compute_os.py::create_filter_prefetch_mat",
                                              "systolic_compute_is.py::create_filter_prefetch_mat"]),
    ("3", "CALC hit/miss lookup", ["read_buffer_estimate_bw.py::check_hit",
                                   "read_buffer_estimate_bw.py::manage_prefetches"]),
    ("4a", "USER hit/miss lookup", ["read_buffer.py::active_buffer_hit",
                                    "read_buffer.py::prepare_hashed_buffer"]),
    ("4b", "USER fetch-matrix layout", ["read_buffer.py::set_fetch_matrix"]),
    ("5", "Read DRAM trace construction", ["read_buffer_estimate_bw.py::prefetch",
                                           "read_buffer_estimate_bw.py::_finalize_trace_matrix",
                                           "read_buffer.py::new_prefetch",
                                           "read_buffer.py::_finalize_trace_matrix"]),
    ("6", "Write DRAM trace construction", ["write_buffer.py::append_to_trace_mat",
                                            "write_buffer.py::_append_chunk_to_trace_matrix"]),
]

# Validated categorical palette, slots 1-6 in fixed order (dataviz reference palette, light mode),
# plus a hatch per slot so the figure survives grayscale printing.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
HATCHES = ["", "////", "", "\\\\\\\\", "....", "xxxx"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, as IEEE/ACM checkers require
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "hatch.linewidth": 0.5, "hatch.color": "#ffffff",
})

out_numbers = []


def note(line):
    out_numbers.append(line)


def fold(cats):
    return {name: sum(cats.get(c, 0.0) for c in members) for name, members in PAPER_CATEGORIES}


def fix_savings(v_fn, o_fn, saved):
    rows, covered = [], 0.0
    for fid, name, funcs in FIXES:
        s = sum(v_fn.get(f, 0.0) - o_fn.get(f, 0.0) for f in funcs)
        rows.append((fid, name, s, s / saved * 100))
        covered += s
    return rows, saved - covered


# ---------------------------------------------------------------- profiles
prof = {}
for key, name, mode, desc in WORKLOADS:
    v_tot, v_cat, v_fn = breakdown.summarize(os.path.join(PROF, f"{name}_vanilla.pstats"))
    o_tot, o_cat, o_fn = breakdown.summarize(os.path.join(PROF, f"{name}_optimized.pstats"))
    fixes, residual = fix_savings(v_fn, o_fn, v_tot - o_tot)
    prof[key] = dict(name=name, mode=mode, desc=desc, v_tot=v_tot, o_tot=o_tot,
                     v=fold(v_cat), o=fold(o_cat), fixes=fixes, residual=residual)
    note(f"[{mode}] {desc}: vanilla {v_tot:.1f}s, optimized {o_tot:.1f}s (profiled), "
         f"speedup {v_tot/o_tot:.2f}x, optimized = {o_tot/v_tot*100:.1f}% of vanilla")
    for c, _ in PAPER_CATEGORIES:
        a, b = prof[key]["v"][c], prof[key]["o"][c]
        note(f"    {c:28s} vanilla {a:7.1f}s {a/v_tot*100:5.1f}%   optimized {b:7.1f}s {b/o_tot*100:5.1f}%")
    for fid, fname, s, pct in fixes:
        note(f"    fix #{fid:3s} {fname:32s} saved {s:7.1f}s = {pct:5.1f}% of saving")
    note(f"    unchanged code (run-to-run variation)   {residual:7.1f}s = {residual/(v_tot-o_tot)*100:5.1f}%")

# ---------------------------------------------------------------- figure 1: breakdown
LEGEND_NAME = {"Progress-bar objects": "Progress-bar objects", "Cycle loop & write buffer": "Cycle loop & write buf."}
fig, ax = plt.subplots(figsize=(3.5, 2.55))
bars = []  # (y, label, workload key, version)
y = 0
yticks, ylabels, group_mid = [], [], []
for key, *_ in WORKLOADS:
    group_mid.append(y - 0.5)
    for ver in ("v", "o"):
        bars.append((y, key, ver))
        yticks.append(y)
        ylabels.append("Vanilla" if ver == "v" else "Optimized")
        y -= 1
    y -= 0.7

for yy, key, ver in bars:
    p = prof[key]
    left = 0.0
    for i, (c, _) in enumerate(PAPER_CATEGORIES):
        w = p[ver][c] / p["v_tot"] * 100  # % of this workload's vanilla time
        ax.barh(yy, w, left=left, height=0.72, color=COLORS[i], hatch=HATCHES[i],
                edgecolor="#fcfcfb", linewidth=0.8,
                label=LEGEND_NAME.get(c, c) if (yy == bars[0][0]) else None)
        if w >= 9:
            ax.text(left + w / 2, yy, f"{w:.0f}", ha="center", va="center", fontsize=5.8,
                    color="#ffffff" if i in (0, 5) else INK)
        left += w
    total = p["o_tot"] / p["v_tot"] * 100 if ver == "o" else 100.0
    tag = f"{total:.0f}%" if ver == "v" else f"{total:.0f}%  ({p['v_tot']/p['o_tot']:.2f}× faster)"
    ax.text(left + 1.5, yy, tag, va="center", ha="left", fontsize=7, color=INK2)

for (key, name, mode, desc), ym in zip(WORKLOADS, group_mid):
    ax.text(-1, ym + 1.05, f"{mode} mode: {desc}", ha="left", va="bottom", fontsize=6,
            fontweight="bold", transform=ax.transData)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xlim(0, 128)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Profiled execution time (% of vanilla)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_ylim(y + 0.9, 1.25)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.0, columnspacing=0.6, handletextpad=0.3, fontsize=6.6)
fig.subplots_adjust(left=0.19, right=0.98, top=0.83, bottom=0.155)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_time_breakdown.{ext}"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- sweep (un-profiled)
d = pd.read_csv(RESULTS)
d = d[d.phase.isin(["main_grid", "dataflow_addendum"])]
TIMEOUT = 5400.0
CONFIGS = [  # (combo_id, label, mode)
    ("anchor", "32×32, 64 KB (default)", "CALC"),
    ("grid_a16_s16_calc", "16×16, 16 KB", "CALC"),
    ("grid_a16_s256_calc", "16×16, 256 KB", "CALC"),
    ("grid_a64_s16_calc", "64×64, 16 KB", "CALC"),
    ("grid_a64_s256_calc", "64×64, 256 KB", "CALC"),
    ("grid_a16_s16_user", "16×16, 16 KB", "USER"),
    ("grid_a16_s256_user", "16×16, 256 KB", "USER"),
    ("grid_a64_s16_user", "64×64, 16 KB", "USER"),
    ("grid_a64_s256_user", "64×64, 256 KB", "USER"),
]
MODEL_KIND = {"alexnet": "conv, 5", "googlenet": "conv, 58", "mobilenet": "conv, 27",
              "vit_b": "GEMM, 5", "transformer": "GEMM, 54"}  # layer type, layer count
MODELS = [("alexnet", "AlexNet"), ("googlenet", "GoogLeNet"), ("mobilenet", "MobileNet"),
          ("vit_b", "ViT-B"), ("transformer", "Transformer")]


def cell(model, combo):
    """(kind, speedup, vanilla_mean, vanilla_sd, opt_mean, opt_sd); kind: ok | lower | none."""
    r = d[(d.model == model) & (d.combo_id == combo)]
    v, o = r[r.version == "vanilla"], r[r.version == "optimized"]
    vok, ook = v[v.status == "ok"], o[o.status == "ok"]
    if len(vok) and len(ook):
        return ("ok", vok.wall_seconds.mean() / ook.wall_seconds.mean(), vok.wall_seconds.mean(),
                vok.wall_seconds.std(), ook.wall_seconds.mean(), ook.wall_seconds.std())
    if len(ook) and (v.status == "timeout").all() and len(v):
        return ("lower", TIMEOUT / ook.wall_seconds.mean(), None, None,
                ook.wall_seconds.mean(), ook.wall_seconds.std())
    return ("none", None, None, None, None, None)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


grid = {(m, c): cell(m, c) for m, _ in MODELS for c, _, _ in CONFIGS}
ok_pairs = [v for v in grid.values() if v[0] == "ok"]
cvs = [s / m for k, sp, m, s, om, os_ in ok_pairs for (m, s) in ((m, s), (om, os_))]
cyc = d[d.status == "ok"].groupby(["model", "combo_id", "version"])[
    ["total_cycles", "total_cycles_incl_prefetch"]].first().unstack("version")
both = cyc.dropna()
cycles_match = ((both["total_cycles"]["vanilla"] == both["total_cycles"]["optimized"]) &
                (both["total_cycles_incl_prefetch"]["vanilla"] == both["total_cycles_incl_prefetch"]["optimized"]))
note("")
note(f"sweep: {len(ok_pairs)} model/config pairs where both finished; "
     f"geomean speedup {geomean([p[1] for p in ok_pairs]):.2f}x, "
     f"min {min(p[1] for p in ok_pairs):.2f}x, max {max(p[1] for p in ok_pairs):.2f}x")
note(f"sweep: max coefficient of variation over 3 runs {max(cvs)*100:.1f}%, median {sorted(cvs)[len(cvs)//2]*100:.1f}%")
note(f"sweep: simulated cycle counts identical in {int(cycles_match.sum())}/{len(cycles_match)} pairs where both finished")
for mode in ("CALC", "USER"):
    xs = [grid[(m, c)][1] for m, _ in MODELS for c, _, md in CONFIGS if md == mode and grid[(m, c)][0] == "ok"]
    note(f"sweep: {mode} geomean {geomean(xs):.2f}x over {len(xs)} pairs, range {min(xs):.2f}-{max(xs):.2f}x")
lows = [(m, c, grid[(m, c)][1]) for m, _ in MODELS for c, _, _ in CONFIGS if grid[(m, c)][0] == "lower"]
note(f"sweep: {len(lows)} pairs where vanilla hit the {TIMEOUT/60:.0f}-min timeout but optimized finished: "
     + ", ".join(f"{m}/{c} >={s:.1f}x" for m, c, s in lows))
for c, lab, md in CONFIGS:
    xs = [grid[(m, c)][1] for m, _ in MODELS if grid[(m, c)][0] == "ok"]
    if len(xs) >= 3:
        note(f"    {md} {lab:24s} geomean {geomean(xs):.2f}x (n={len(xs)})")

# ---------------------------------------------------------------- figure 2: speedup dot plot
fig, ax = plt.subplots(figsize=(3.5, 2.6))
MARK = {"alexnet": "o", "googlenet": "s", "mobilenet": "D", "vit_b": "^", "transformer": "v"}
xpos = []
for i, (c, lab, md) in enumerate(CONFIGS):
    x = i + (0.6 if md == "USER" else 0)
    xpos.append(x)
    xs = []
    for j, (m, mname) in enumerate(MODELS):
        kind, sp, *_ = grid[(m, c)]
        if kind == "none":
            continue
        xo = x + (j - 2) * 0.09
        if kind == "ok":
            xs.append(sp)
            ax.plot(xo, sp, MARK[m], ms=4.2, mfc=COLORS[0], mec="#fcfcfb", mew=0.5, zorder=3)
        else:
            ax.plot(xo, sp, MARK[m], ms=4.2, mfc="#fcfcfb", mec=COLORS[0], mew=0.8, zorder=3)
            ax.plot([xo, xo], [sp * 1.1, sp * 1.35], color=COLORS[0], lw=0.8, zorder=2)  # "at least"
    if len(xs) >= 3:  # a mean over one or two models would mislead
        g = geomean(xs)
        ax.plot([x - 0.3, x + 0.3], [g, g], color=INK, lw=1.2, zorder=4)
ax.axhline(1, color=MUTED, lw=0.6)
ax.set_yscale("log")
ax.set_ylim(0.9, 32)
ax.set_yticks([1, 2, 4, 8, 16])
ax.set_yticklabels(["1×", "2×", "4×", "8×", "16×"])
ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="major")
ax.set_axisbelow(True)
ax.set_xticks(xpos)
ax.set_xticklabels([lab.replace(" (default)", "\n(default)").replace(", ", "\n") for _, lab, _ in CONFIGS],
                   fontsize=6.3)
ax.set_xlim(-0.6, xpos[-1] + 0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_ylabel("Speedup over vanilla")
calc_mid = sum(xpos[:5]) / 5
user_mid = sum(xpos[5:]) / 4
for xm, t in ((calc_mid, "CALC bandwidth mode"), (user_mid, "USER bandwidth mode")):
    ax.text(xm, 29, t, ha="center", va="top", fontsize=7.5, fontweight="bold")
ax.axvline((xpos[4] + xpos[5]) / 2, color=GRID, lw=0.8)
handles = [plt.Line2D([], [], ls="", marker=MARK[m], ms=4.2, mfc=COLORS[0], mec="#fcfcfb", label=n)
           for m, n in MODELS]
handles += [plt.Line2D([], [], color=INK, lw=1.2, label="Geometric mean"),
            plt.Line2D([], [], ls="", marker="o", ms=4.2, mfc="#fcfcfb", mec=COLORS[0],
                       label="Lower bound (vanilla > 90 min)")]
handles[-1] = plt.Line2D([], [], color=COLORS[0], lw=0.8, marker="o", ms=4.2, mfc="#fcfcfb", mec=COLORS[0],
                         markevery=[0], label="Lower bound (vanilla > 90 min)")
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 1.0),
           handlelength=1.3, columnspacing=0.8, handletextpad=0.3, fontsize=6.5)
fig.subplots_adjust(left=0.13, right=0.99, top=0.79, bottom=0.18)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_speedup.{ext}"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- LaTeX tables
calc, user = prof["calc"], prof["user"]
T = []
T.append(r"""% Generated by benchmark/cprofile_compare/paper/make_paper_assets.py -- do not edit by hand.
% Needs \usepackage{booktabs}; tab:optimizations also \usepackage{tabularx}.

% ------------------------------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{Optimizations applied to SCALE-Sim. Each one leaves the simulated behavior unchanged:
reports and all per-layer SRAM/DRAM traces are identical to the unmodified simulator. The
last two columns give each optimization's share of the total time saved in the two profiled
GoogLeNet runs (Fig.~\ref{fig:breakdown}).}
\label{tab:optimizations}
\small
\begin{tabularx}{\textwidth}{@{}l l X X r r@{}}
\toprule
\# & Component & Bottleneck in the original code & Change & \multicolumn{2}{c}{Share of saving} \\
\cmidrule(l){5-6}
 & & & & CALC & USER \\
\midrule""")
DESC = {
    "1": (r"A \texttt{tqdm} progress-bar object (display disabled) was constructed on every simulated cycle in the read- and write-servicing loops.",
          r"Plain loops; the object is no longer created."),
    "2": (r"Per-element Python loop with scalar indexing to reorder the operand matrix along anti-diagonals, $O(MN)$ interpreted steps per layer.",
          r"One vectorized NumPy gather per diagonal."),
    "3": (r"Every address lookup scanned all sets in the active window: $O(W)$ per address, per cycle.",
          r"Map from address to most recent set id; $O(1)$ window test."),
    "4a": (r"Every address lookup scanned all lines in the (wrapping) active window.",
           r"Static reverse index from address to line ids, built once per layer."),
    "4b": (r"Element-by-element copy with index arithmetic to reshape the fetch matrix.",
           r"Single pad-and-reshape."),
    "5": (r"Each DRAM prefetch concatenated onto the whole trace so far: $O(n^2)$ copying over $n$ prefetches.",
          r"Chunks collected in a list, joined once when first read."),
    "6": (r"Each write-buffer drain concatenated onto the whole trace so far: $O(n^2)$ copying.",
          r"Geometrically growing pre-allocated buffer, amortized $O(n)$."),
}
for (fid, name, s_c, p_c), (_, _, s_u, p_u) in zip(calc["fixes"], user["fixes"]):
    bott, chg = DESC[fid]
    fmt = lambda p: "--" if abs(p) < 0.05 else f"{p:.1f}\\%"
    T.append(f"{fid} & {name} & {bott} & {chg} & {fmt(p_c)} & {fmt(p_u)} \\\\")
rc = calc["residual"] / (calc["v_tot"] - calc["o_tot"]) * 100
ru = user["residual"] / (user["v_tot"] - user["o_tot"]) * 100
T.append(r"\midrule")
T.append(f"-- & Unchanged code & \\multicolumn{{2}}{{l}}{{Run-to-run variation in code not modified}} & {rc:.1f}\\% & {ru:.1f}\\% \\\\")
T.append(r"""\bottomrule
\end{tabularx}
\end{table*}
""")

T.append(r"""% ------------------------------------------------------------------------------------------------
\begin{table}[t]
\centering
\caption{Share of execution time per simulator component, before and after optimization
(cProfile; library time is charged to the simulator function that called it).
Workloads as in Fig.~\ref{fig:breakdown}.}
\label{tab:breakdown}
\small
\begin{tabular}{@{}l rr rr@{}}
\toprule
 & \multicolumn{2}{c}{CALC} & \multicolumn{2}{c}{USER} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Component & Vanilla & Opt. & Vanilla & Opt. \\
\midrule""")
for c, _ in PAPER_CATEGORIES:
    vals = [calc["v"][c] / calc["v_tot"], calc["o"][c] / calc["o_tot"],
            user["v"][c] / user["v_tot"], user["o"][c] / user["o_tot"]]
    T.append(f"{c.replace('&', chr(92) + '&')} & " + " & ".join(f"{x*100:.1f}\\%" for x in vals) + r" \\")
T.append(r"\midrule")
T.append(f"Profiled time (s) & {calc['v_tot']:.0f} & {calc['o_tot']:.0f} & {user['v_tot']:.0f} & {user['o_tot']:.0f} \\\\")
T.append(f"Speedup (profiled) & \\multicolumn{{2}}{{c}}{{{calc['v_tot']/calc['o_tot']:.2f}$\\times$}} & "
         f"\\multicolumn{{2}}{{c}}{{{user['v_tot']/user['o_tot']:.2f}$\\times$}} \\\\")
T.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

T.append(r"""% ------------------------------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{End-to-end speedup of the optimized simulator over the unmodified one (wall-clock time
without profiling, mean of three runs; weight-stationary dataflow unless noted).
$\geq$: the unmodified simulator exceeded the 90-minute limit in all three runs, so the value is a
lower bound. --: neither finished, or fewer than three models for a geometric mean. Second header
row: layer type and number of layers (ViT-B is the 5-layer GEMM slice from the ISPASS'25 topology
set; Transformer is SCALE-Sim's \texttt{transformer\_fwd} topology). Simulated cycle counts are
identical in every case where both finished.}
\label{tab:speedup}
\small
\begin{tabular}{@{}l l """ + "r" * len(MODELS) + r""" r@{}}
\toprule
Mode & Array, SRAM & """ + " & ".join(n for _, n in MODELS) + r""" & Geomean \\
 & & """ + " & ".join(MODEL_KIND[m] for m, _ in MODELS) + r""" & \\
\midrule""")
prev = None
for c, lab, md in CONFIGS:
    if prev and md != prev:
        T.append(r"\midrule")
    cells, xs = [], []
    for m, _ in MODELS:
        kind, sp, *_ = grid[(m, c)]
        if kind == "ok":
            cells.append(f"{sp:.2f}")
            xs.append(sp)
        elif kind == "lower":
            cells.append(f"$\\geq${sp:.1f}")
        else:
            cells.append("--")
    g = geomean(xs) if len(xs) >= 3 else None
    T.append(f"{md if md != prev else ''} & {lab} & " + " & ".join(cells) + f" & {g:.2f} \\\\" if g else
             f"{md if md != prev else ''} & {lab} & " + " & ".join(cells) + " & -- \\\\")
    prev = md
T.append(r"\midrule")
for c, lab in (("anchor_dfos", "32×32, 64 KB, OS"), ("anchor_dfis", "32×32, 64 KB, IS")):
    cells, xs = [], []
    for m, _ in MODELS:
        kind, sp, *_ = cell(m, c)
        cells.append(f"{sp:.2f}" if kind == "ok" else "n/a")
        if kind == "ok":
            xs.append(sp)
    T.append(f"CALC & {lab} & " + " & ".join(cells) + " & -- \\\\")  # only 2 models ran these
T.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

T.append(r"""% ------------------------------------------------------------------------------------------------
\begin{table}[t]
\centering
\caption{Wall-clock time in the default configuration (32$\times$32, 64\,KB SRAMs, CALC,
weight-stationary), mean $\pm$ standard deviation over three runs.}
\label{tab:walltime}
\small
\begin{tabular}{@{}l rr r@{}}
\toprule
Model & Vanilla (s) & Optimized (s) & Speedup \\
\midrule""")
for m, n in MODELS:
    kind, sp, vm, vs, om, os_ = grid[(m, "anchor")]
    T.append(f"{n} & {vm:.1f} $\\pm$ {vs:.1f} & {om:.1f} $\\pm$ {os_:.1f} & {sp:.2f}$\\times$ \\\\")
T.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
tex = "\n".join(T).replace("×", r"$\times$").replace("$\\times$$\\times$", "$\\times$")
with open(os.path.join(HERE, "tables.tex"), "w") as f:
    f.write(tex)
with open(os.path.join(HERE, "numbers.txt"), "w") as f:
    f.write("\n".join(out_numbers) + "\n")
print("\n".join(out_numbers))
print("\nwrote fig_time_breakdown.{pdf,png}, fig_speedup.{pdf,png}, tables.tex, numbers.txt")

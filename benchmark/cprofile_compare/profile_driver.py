#!/usr/bin/env python3
"""Profile one SCALE-Sim checkout with cProfile.

Workload: MobileNet layers Conv1-Conv3 (mobilenet3.csv), ws dataflow, 32x32 array, 64KB SRAMs.
It runs twice inside a single cProfile session: once with InterfaceBandwidth=CALC (goes through
read_buffer_estimate_bw.py) and once with USER (goes through read_buffer.py). Every changed
function in the optimized version therefore shows up in one .pstats file per version.

usage: profile_driver.py <repo_root> <tag> <out_dir>
Imports scalesim from <repo_root>, not from whatever pip installed, and asserts that it did.
"""
import cProfile, os, sys, time

repo, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)
import scalesim
assert os.path.realpath(scalesim.__file__).startswith(os.path.realpath(repo)), scalesim.__file__
print(f"[{tag}] scalesim imported from {scalesim.__file__}", flush=True)
from scalesim.scale_sim import scalesim as Sim

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out, exist_ok=True)  # simulator.py uses os.mkdir, so the parent must exist
pr = cProfile.Profile()
for bw in ("CALC", "USER"):
    s = Sim(save_disk_space=False, verbose=False,
            config=os.path.join(here, f"scale_{bw}.cfg"),
            topology=os.path.join(here, "mobilenet3.csv"),
            layout=os.path.join(repo, "layouts/conv_nets/test.csv"),
            input_type_gemm=False)
    t0 = time.perf_counter()
    pr.enable(); s.run_scale(top_path=os.path.join(out, bw)); pr.disable()
    print(f"[{tag}] {bw}: {time.perf_counter()-t0:.1f}s (under cProfile)", flush=True)
pr.dump_stats(os.path.join(out, f"{tag}.pstats"))

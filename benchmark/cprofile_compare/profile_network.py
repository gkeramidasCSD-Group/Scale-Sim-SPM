#!/usr/bin/env python3
"""Profile one SCALE-Sim checkout on a full network with cProfile, on a low-RAM machine.

usage: profile_network.py <repo_root> <tag> <out_dir> <config.cfg> <topology.csv> [conv|gemm]

Stock SCALE-Sim keeps every layer's simulation objects (demand matrices, SRAM/DRAM trace
matrices) alive until the whole network finishes, so peak RAM grows with the network's total
cycle count. A full network can exceed 7GB. To stay safe, this driver wraps
single_layer_sim.save_traces the same way for both versions. After a layer's traces are written:
  1. calc_report_data() runs (profiled; stock code runs it later, in generate_reports, anyway)
  2. with the profiler paused: md5 each trace CSV into <out_dir>/trace_md5.txt, delete the CSVs
     (about 4GB of disk per run otherwise), drop the layer's memory_system/compute_system, and
     log RSS
generate_reports() only reads the numbers cached by calc_report_data(), so the reports are
unchanged. The simulation code itself is not modified.
"""
import cProfile, gc, hashlib, os, sys, time

repo, tag, out, cfg, topo = sys.argv[1:6]
input_type = sys.argv[6] if len(sys.argv) > 6 else "conv"  # gemm: topology has an M,N,K header
sys.path.insert(0, repo)
import scalesim
assert os.path.realpath(scalesim.__file__).startswith(os.path.realpath(repo)), scalesim.__file__
from scalesim.scale_sim import scalesim as Sim
from scalesim.single_layer_sim import single_layer_sim

os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "layers.log"), "w", buffering=1)
md5_out = open(os.path.join(out, "trace_md5.txt"), "w")
pr = cProfile.Profile()
t_start = time.perf_counter()


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024


def emit(msg):
    print(msg, flush=True)
    log.write(msg + "\n")


_orig_save_traces = single_layer_sim.save_traces


def save_traces_and_free(self, top_path, *args, **kwargs):
    _orig_save_traces(self, top_path, *args, **kwargs)
    if not self.report_items_ready:
        self.calc_report_data()
    pr.disable()
    layer_dir = os.path.join(top_path, "layer" + str(self.layer_id))
    peak = rss_mb()
    for name in sorted(os.listdir(layer_dir)):
        path = os.path.join(layer_dir, name)
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        md5_out.write(f"layer{self.layer_id}/{name} {h.hexdigest()}\n")
        os.remove(path)
    md5_out.flush()
    self.memory_system = None
    self.compute_system = None
    gc.collect()
    emit(f"[{tag}] layer {self.layer_id:2d} done  t={time.perf_counter()-t_start:7.1f}s  "
         f"rss_before_free={peak}MB rss_after_free={rss_mb()}MB")
    pr.enable()


single_layer_sim.save_traces = save_traces_and_free

emit(f"[{tag}] scalesim imported from {scalesim.__file__} (input type: {input_type})")
s = Sim(save_disk_space=False, verbose=False, config=cfg, topology=topo,
        layout=os.path.join(repo, "layouts/conv_nets/test.csv"), input_type_gemm=(input_type == "gemm"))
pr.enable()
s.run_scale(top_path=out)
pr.disable()
emit(f"[{tag}] TOTAL {time.perf_counter()-t_start:.1f}s wall (under cProfile)")
pr.dump_stats(os.path.join(out, f"{tag}.pstats"))

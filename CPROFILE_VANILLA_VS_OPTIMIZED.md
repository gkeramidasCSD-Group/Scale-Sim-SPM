# cProfile: vanilla SCALE-Sim vs. optimized (`sim-opt`)

This profiles upstream SCALE-Sim and this repo on the same workloads and compares them. It shows
what share of the runtime each part of the simulator takes before and after the performance
changes, and which changed function saved how much.

**Two workloads:**

| Run | Workload | Vanilla | Optimized | Speedup | Output |
|---|---|---|---|---|---|
| **A: full network** | GoogLeNet, 58 layers, `ws` 16×16, 16 KB SRAMs, `CALC` | **1260 s (21.0 min)** | **704 s (11.7 min)** | **1.79×** | reports byte-identical, 348/348 trace CSVs identical (md5) |
| B: coverage run | MobileNet Conv1–3, `ws` 32×32, 64 KB SRAMs, `CALC` **and** `USER` | 98.7 s | 55.3 s | 1.78× | 44/44 CSVs byte-identical |

Times are profiled time under cProfile.
- **Run A** answers "where does the time go on a real network". It only uses `CALC` mode, the
  default config.
- **Run B** is small, but it also runs `USER` mode. That's the only way to exercise the
  `read_buffer.py` changes (#4a, #4b).

Everything needed to rerun both is in [benchmark/cprofile_compare/](benchmark/cprofile_compare/).

---

## 1. Where the time goes: before and after (Run A, GoogLeNet)

**Method.** Raw cProfile `tottime` scatters library time (numpy, `dict`/`set` builtins,
`savetxt` internals) over hundreds of rows. [breakdown.py](benchmark/cprofile_compare/breakdown.py)
charges each library call to the SCALE-Sim function that made it, then groups those functions
into simulator-level categories. The categories add up to exactly 100% of each run.

### By simulator category

| Category | Vanilla s | **Vanilla %** | Optimized s | **Optimized %** | Saved s | Share of total saving |
|---|---|---|---|---|---|---|
| Read hit/miss + prefetch decisions (`check_hit`, `manage_prefetches`, `service_reads`) | 399.2 | **31.7%** | 226.6 | **32.2%** | 172.6 | 31.0% |
| DRAM trace building in memory (`prefetch`, `store_to_trace_mat_cache`, `append_to_trace_mat`) | 263.3 | **20.9%** | 127.0 | **18.0%** | 136.2 | 24.5% |
| `tqdm` progress-bar objects (never displayed) | 193.0 | **15.3%** | 1.0 | **0.1%** | 192.0 | **34.5%** |
| Trace CSV writing (`savetxt`) | 142.7 | **11.3%** | 142.8 | **20.3%** | 0 | 0% |
| Per-cycle memory loop + SRAM trace assembly (`service_memory_requests`) | 136.1 | **10.8%** | 124.3 | **17.7%** | 11.9 | 2.1% |
| Write-buffer servicing (`service_writes`, `empty_drain_buf`) | 103.8 | **8.2%** | 65.6 | **9.3%** | 38.2 | 6.9% |
| Prefetch-matrix diagonal flatten | 8.6 | 0.7% | 3.4 | 0.5% | 5.2 | 0.9% |
| Other (config/topology parsing, report files) | 7.9 | 0.6% | 7.5 | 1.1% | 0.4 | 0.1% |
| Operand/demand matrix generation | 4.2 | 0.3% | 4.5 | 0.6% | −0.2 | 0% |
| Report statistics | 1.4 | 0.1% | 1.3 | 0.2% | 0.2 | 0% |
| **Total** | **1260.3** | 100% | **704.0** | 100% | **556.3** | 100% |

**What this shows:**
- **About a third of vanilla's saving was pure waste removed.** Vanilla built a `tqdm` progress
  bar with `disable=True` on every simulated cycle in `write_buffer.service_writes`. That's
  6.56 M objects, and none was ever displayed. Removing them saved 192 s, the single biggest win.
- **The hit/miss path is still the largest category (~32%)**, even though it's 1.76× faster.
  `check_hit` is now O(1), and what's left is the cost of calling Python functions 91 M times.
- **The DRAM-trace fixes (#5, #6) matter on a real network.** With 16 KB SRAMs there are about
  10.5k prefetch and drain events, and vanilla re-copied the whole trace at every one. That
  category fell from 263 s to 127 s.
- **Trace CSV writing didn't change (143 s).** Its share grew from 11% to 20% only because
  everything around it got faster. It's now the second-largest category.

### Top functions (exclusive time, library calls included)

| # | Vanilla | s | % | Optimized | s | % |
|---|---|---|---|---|---|---|
| 1 | `read_buffer_estimate_bw::check_hit` | 245.5 | 19.5% | `double_buffered_scratchpad_mem::service_memory_requests` | 124.3 | 17.7% |
| 2 | `tqdm` (all functions) | 193.0 | 15.3% | `write_buffer::store_to_trace_mat_cache` | 116.0 | 16.5% |
| 3 | `double_buffered_scratchpad_mem::service_memory_requests` | 136.1 | 10.8% | `read_buffer_estimate_bw::service_reads` | 88.1 | 12.5% |
| 4 | `write_buffer::store_to_trace_mat_cache` | 132.8 | 10.5% | `read_buffer_estimate_bw::check_hit` | 72.1 | 10.2% |
| 5 | `write_buffer::service_writes` | 100.1 | 7.9% | `read_buffer_estimate_bw::manage_prefetches` | 66.5 | 9.4% |
| 6 | `read_buffer_estimate_bw::service_reads` | 93.2 | 7.4% | `write_buffer::service_writes` | 62.0 | 8.8% |
| 7 | `read_buffer_estimate_bw::prefetch` | 73.4 | 5.8% | `read_buffer_estimate_bw::print_trace` | 48.0 | 6.8% |
| 8 | `read_buffer_estimate_bw::manage_prefetches` | 60.5 | 4.8% | `write_buffer::print_trace` | 30.6 | 4.3% |
| 9 | `write_buffer::append_to_trace_mat` | 57.1 | 4.5% | `double_buffered_scratchpad_mem::print_ofmap_sram_trace` | 22.5 | 3.2% |
| 10 | `read_buffer_estimate_bw::print_trace` | 47.8 | 3.8% | `double_buffered_scratchpad_mem::print_ifmap_sram_trace` | 21.9 | 3.1% |

## 2. Where the time goes: Run B (MobileNet Conv1–3, `CALC` + `USER`)

| Category | Vanilla s | **Vanilla %** | Optimized s | **Optimized %** | Share of saving |
|---|---|---|---|---|---|
| Read hit/miss (`USER`: `active_buffer_hit`, `service_reads`) | 20.5 | **20.8%** | 12.9 | **23.4%** | 17.6% |
| Read hit/miss + prefetch decisions (`CALC`) | 18.4 | **18.6%** | 10.8 | **19.6%** | 17.4% |
| `tqdm` progress-bar objects | 16.9 | **17.1%** | 0.1 | **0.1%** | **38.8%** |
| `USER` read-buffer setup (`set_fetch_matrix`, `prepare_hashed_buffer`) | 10.2 | **10.4%** | 5.4 | **9.7%** | 11.2% |
| Trace CSV writing (`savetxt`) | 8.8 | **8.9%** | 8.7 | **15.7%** | 0.1% |
| Prefetch-matrix diagonal flatten | 6.4 | **6.5%** | 1.4 | **2.5%** | 11.7% |
| Per-cycle memory loop + SRAM trace assembly | 6.1 | **6.2%** | 6.0 | **10.9%** | 0.2% |
| DRAM trace building in memory | 5.2 | **5.3%** | 5.2 | **9.5%** | −0.1% |
| Write-buffer servicing | 4.6 | **4.7%** | 3.4 | **6.1%** | 2.9% |
| Operand/demand matrix generation, report stats, other | 1.5 | 1.5% | 1.5 | 2.6% | 0% |
| **Total** | **98.7** | 100% | **55.3** | 100% | 100% |

Only 3 layers with 64 KB SRAMs means just 140 prefetch events, so the DRAM-trace fixes (#5, #6)
barely show up here. Compare that with Run A. On the other hand, the per-layer setup costs (the
diagonal flatten and `set_fetch_matrix`) take a larger share, because there are few cycles per
layer.

---

## 3. What was compared, and how

| | Vanilla | Optimized |
|---|---|---|
| Path | `/home/george/Music/SCALE-Sim` | `/home/george/Desktop/SCALE-Sim` (this repo, `sim-opt` @ `c8c9a86`) |
| Code | upstream `scalesim-project/SCALE-Sim` `main` @ `9f98c43` | vanilla + the changes in section 5 |

**Is the Music copy really vanilla? Yes.** Its `HEAD` is `9f98c43`, the same commit
`git ls-remote origin HEAD` reports for upstream `main`, and `git status` shows **no changes
under `scalesim/`**. The only local edit is `configs/scale.cfg` (array set to 16×16), which
isn't used here; both versions get identical configs from `benchmark/cprofile_compare/`. A
recursive diff of the two `scalesim/` packages finds exactly 8 differing files, all covered in
section 5.

**Right code imported:** the system `python3` has this repo editable-installed, so a plain
`import scalesim` would load the *optimized* code even from the vanilla folder. Both drivers put
the target repo first on `sys.path` and **assert** that `scalesim.__file__` lives inside it.

**Environment:** Python 3.10.12, numpy 1.26.4, BLAS threads pinned to 1. The two versions ran
**one after the other**, never in parallel.

### Running a full network on a 7 GB laptop (Run A)

Stock SCALE-Sim keeps every layer's simulation objects (demand matrices plus SRAM/DRAM trace
matrices) in memory until the whole network finishes. So peak RAM grows with the network's
total cycle count, which for GoogLeNet at 16×16 is 6.5 M cycles, well over 7 GB.
[profile_network.py](benchmark/cprofile_compare/profile_network.py) prevents this by wrapping
`single_layer_sim.save_traces` **the same way for both versions**. After each layer writes its
traces:

1. `calc_report_data()` runs. This is profiled; stock code runs it later in `generate_reports`
   anyway.
2. **With the profiler paused**, the driver:
   - records the md5 of each trace CSV, then deletes it (otherwise that's about 4 GB of disk per run);
   - drops the layer's `memory_system` and `compute_system`.

`generate_reports()` only reads the numbers cached in step 1, so every report is unchanged.
**No SCALE-Sim code is modified.** A 700 MB free-RAM watchdog kills the run if memory gets
tight. It never fired.

- The largest layer, Conv2 (3×3, 64→192, about 1.4 M cycles), peaked at 2.05 GB in vanilla and 2.45 GB in optimized.
- Peak RSS for the whole process was 2.80 GB (vanilla) and 2.90 GB (optimized).
- The optimized version uses slightly more peak RAM because of #6's doubling buffer and #5's list of pending trace chunks.

The md5 check does the correctness job that a plain byte diff can't do once the CSVs are deleted.

**Why GoogLeNet 16×16 / 16 KB `CALC`:** it's a real 58-layer network (6.56 M simulated
cycles). The earlier server sweep (`results.csv`) put it at 742 s vanilla without cProfile, so it
was expected to run about 25–30 min on this laptop under cProfile. It came in at 21 min. The
sweep's longer configs (for example MobileNet 64×64/16 KB `USER`, 4192 s) were ruled out: at
roughly 5–7× longer they would run for hours, and their larger per-layer cycle counts risk the
RAM limit.

### Run B setup

MobileNet Conv1–Conv3 (`mobilenet3.csv`), `ws` 32×32, 64 KB SRAMs, run once with
`InterfaceBandwidth: CALC` and once with `USER` inside one cProfile session per version. The two
modes use different read-buffer classes (`read_buffer_estimate_bw.py` and `read_buffer.py`), so
this is the only run that covers fixes #4a and #4b. It's kept to 3 layers because of the RAM limit.

### Caveats

- cProfile adds a fixed cost to every Python function call. Functions called tens of millions of
  times (`check_hit`, `manage_prefetches`) look slower than they are in a normal run. Compare the
  ratios, not the absolute seconds. The speedups still line up with the un-profiled server sweep
  (~1.7× for `CALC`).
- **Unchanged functions varied by up to ~10% between runs.** For example,
  `service_memory_requests` went 136 s → 124 s and `store_to_trace_mat_cache` went 133 s → 116 s,
  with no code changes. That's likely memory-allocator and cache effects. Treat differences of
  that size as noise.

### Reproduce

```bash
cd /home/george/Desktop/SCALE-Sim
# Run A: GoogLeNet 16x16 / 16 KB / CALC (the defaults), ~33 min total, optimized first then vanilla.
# Writes profiles/googlenet_a16_s16_calc_{vanilla,optimized}.pstats and _report.md
benchmark/cprofile_compare/run_network.sh

# Rebuild the tables from the saved profiles without re-running
python3 benchmark/cprofile_compare/breakdown.py \
    benchmark/cprofile_compare/profiles/googlenet_a16_s16_calc_vanilla.pstats \
    benchmark/cprofile_compare/profiles/googlenet_a16_s16_calc_optimized.pstats   # section 1
python3 benchmark/cprofile_compare/compare.py googlenet_a16_s16_calc_               # section 4

# Run B: MobileNet Conv1-3, CALC + USER, ~2.5 min (laptop-sized)
benchmark/cprofile_compare/run_both.sh
```

### Running it on the server (e.g. `USER` mode with small arrays/SRAMs)

`run_network.sh` is configured entirely through environment variables. The full list is in the
header of [run_network.sh](benchmark/cprofile_compare/run_network.sh).

| Variable | Sets |
|---|---|
| `VANILLA_REPO`, `OPTIMIZED_REPO` | the two checkouts |
| `VANILLA_PYTHON`, `OPTIMIZED_PYTHON` | the Python for each, e.g. each repo's venv |
| `TOPO` | the topology CSV |
| `ARRAY`, `SRAM_KB`, `BW` (`CALC` or `USER`), `DATAFLOW` | the config, generated into `configs/<name>.cfg` |
| `PARALLEL=1` | run both versions at once; fine when RAM isn't the limit |

Example, run from the server copy of this repo:

```bash
cd /data/grizos/Scale-Sim-SPM
VANILLA_REPO=/data/grizos/SCALE-Sim \
VANILLA_PYTHON=/data/grizos/SCALE-Sim/venv/bin/python \
OPTIMIZED_PYTHON=/data/grizos/Scale-Sim-SPM/venv/bin/python \
TOPO=topologies/conv_nets/mobilenet.csv ARRAY=16 SRAM_KB=16 BW=USER PARALLEL=1 \
nohup benchmark/cprofile_compare/run_network.sh > cprofile_mobilenet_a16_s16_user.log 2>&1 &

tail -f /tmp/scalesim_cprofile/mobilenet_a16_s16_user/*/layers.log        # per-layer progress
cat benchmark/cprofile_compare/profiles/mobilenet_a16_s16_user_report.md  # when it's done
```

Each run writes these to `profiles/`:

| File | Contents |
|---|---|
| `<name>_report.md` | correctness check, % breakdown, changed-function table (the same tables as sections 1 and 4) |
| `<name>_{vanilla,optimized}.pstats` | raw profiles |
| `<name>_*_{tottime,cumtime}.txt` | top-40 cProfile text reports |

`<name>` looks like `mobilenet_a16_s16_user`, so different configs never overwrite each other.
To look at the results on the laptop, `scp` the `profiles/` folder back. The `.pstats` files also
open in `python3 -m pstats` or snakeviz.

**Things to know for the server:**
- Scripts are checked against Python 3.8, the server's version. The driver asserts that each
  run imported `scalesim` from the right repo, so pointing both at the same venv is also safe.
- `USER` mode with small arrays/SRAMs can be **very long under cProfile**. The sweep's
  MobileNet 64×64/16 KB `USER` run took 4192 s vanilla *without* cProfile, so expect 2–3 h
  with it. Start with `nohup` (as above) or `tmux` so a dropped SSH session doesn't kill it.
- RAM: layers are freed as they finish, so peak RAM is set by the largest single layer, not
  the whole network. With `PARALLEL=1` both processes count against the machine's RAM. The
  watchdog (`MIN_FREE_MB`, default 700) stops both runs rather than let the machine swap.
- Trace CSVs are hashed and then deleted per layer, so disk use stays small. `OUT` (default
  `/tmp/scalesim_cprofile/<name>`) keeps the logs, the reports, and `trace_md5.txt`.

---

## 4. The changed functions: numbers per run

**tot** (tottime) is time in the function's own code. **cum** (cumtime) also includes
everything it calls. All values are in seconds.

### Run A: GoogLeNet (`CALC`, so `read_buffer.py` changes don't run)

| Fix | Function | Calls | Vanilla tot / cum | Optimized tot / cum |
|---|---|---|---|---|
| #1 | `tqdm.__init__` (from `write_buffer.service_writes`) | 6,564,623 → 59 | 20.2 / **132.8** | 0.00 / 0.01 |
| #1 | `write_buffer.py::service_writes` | 6,564,506 | 81.7 / **484.0** | 54.3 / **182.9** |
| #2 | `systolic_compute_ws.py::create_ifmap_prefetch_mat` | 58 | **8.38** / 9.44 | **2.71** / 3.33 |
| #3 | `read_buffer_estimate_bw.py::check_hit` | 91,459,476 | **232.6** / 245.5 | **43.4** / 72.1 |
| #3 | `read_buffer_estimate_bw.py::manage_prefetches` | 91,459,476 | 53.8 / 380.4 | 58.6 / 147.6 |
| #3 | `read_buffer_estimate_bw.py::service_reads` | 13,129,012 | 93.2 / 473.6 | 88.1 / 235.6 |
| #5 | `read_buffer_estimate_bw.py::prefetch` | 10,558 | **70.0** / 73.4 | **6.1** / 9.4 |
| #5 | `read_buffer_estimate_bw.py::_finalize_trace_matrix` (new) | 0 → 232 | — | 1.0 / 1.1 |
| #6 | `write_buffer.py::append_to_trace_mat` | 10,421 | **57.1** / 57.1 | **0.02** / 0.59 |
| #6 | `write_buffer.py::_append_chunk_to_trace_matrix` (new) | 0 → 10,421 | — | 0.56 / 0.57 |
| — | `service_memory_requests` (unchanged; calls all of the above) | 58 | 111.4 / 1095.6 | 100.1 / 544.3 |

Total Python function calls fell from 1,072 M to 692 M (−35%).

### Run B: MobileNet Conv1–3 (`CALC` + `USER`)

| Fix | Function | Calls | Vanilla tot / cum | Optimized tot / cum |
|---|---|---|---|---|
| #1 | `tqdm.__init__` | 590,653 → 7 | 1.73 / **11.34** | 0.00 / 0.00 |
| #1 | `read_buffer.py::service_reads` | 295,320 | 4.55 / **28.88** | 3.39 / **13.10** |
| #1 | `write_buffer.py::service_writes` | 295,320 | 3.78 / **16.96** | 2.97 / **8.04** |
| #2 | `systolic_compute_ws.py::create_ifmap_prefetch_mat` | 6 | **6.38** / 7.12 | **1.12** / 1.36 |
| #3 | `read_buffer_estimate_bw.py::check_hit` | 4,627,491 | **11.72** / 12.34 | **2.40** / 3.79 |
| #3 | `read_buffer_estimate_bw.py::manage_prefetches` | 4,627,491 | 2.62 / 15.85 | 3.21 / 7.72 |
| #4a | `read_buffer.py::active_buffer_hit` (+ its `any()` generator) | 4,627,631 | **14.96** / 14.96 | **3.43** / 9.06 |
| #4a | `read_buffer.py::prepare_hashed_buffer` | 6 | 3.57 / 4.02 | 4.31 / 5.37 |
| #4b | `read_buffer.py::set_fetch_matrix` | 6 | **5.18** / 10.31 | **0.00** / 5.38 |
| #5 | `read_buffer_estimate_bw.py::prefetch` | 140 | 0.40 / 0.56 | 0.18 / 0.36 |
| #5 | `read_buffer.py::new_prefetch` | 140 | 0.37 / 0.38 | 0.16 / 0.17 |
| #6 | `write_buffer.py::append_to_trace_mat` | 239 | 0.09 / 0.09 | 0.00 / 0.01 |
| — | `service_memory_requests` (unchanged) | 6 | 5.04 / 71.01 | 4.93 / 38.41 |

## 5. What each change is and what it affects

### #1 Removed per-cycle `tqdm` progress bars

- **Where:** `write_buffer.py::service_writes`, `read_buffer.py::service_reads` (both loops),
  and the diagonal-flatten loops in `systolic_compute_{ws,os,is}.py`.
- **Change:** `for i in tqdm(range(n), disable=True)` → `for i in range(n)`. The unused
  `pbar = tqdm(...)` / `pbar.update(1)` / `pbar.close()` calls were also removed.
- **Affects:** the per-cycle servicing loops.
  - `service_writes` handles **ofmap writes in every mode**, and it's called once per simulated
    cycle.
  - `read_buffer.service_reads` handles ifmap/filter reads in `USER` mode.
  - `disable=True` meant nothing was ever displayed, but every call still built, registered, and
    garbage-collected a `tqdm` object.
- **Evidence:**
  - Run A: **193 s → 1 s, 15.3% of vanilla's runtime and 34.5% of the total saving.** This is the
    largest single win.
  - Vanilla also made 10.8 M `pbar.update` calls from `create_ifmap_prefetch_mat`.
  - The 59 tqdm objects left in the optimized run are one per layer, from
    `service_memory_requests`. That code is unchanged and harmless.
- **Output change:** none.

### #2 Vectorized the diagonal flatten of the prefetch matrix

- **Where:**
  - `systolic_compute_ws.py::create_ifmap_prefetch_mat`
  - `systolic_compute_os.py::create_ifmap_prefetch_mat` and `create_filter_prefetch_mat`
  - `systolic_compute_is.py::create_filter_prefetch_mat`
- **Change:** the per-element inner loop (`matrix[row][col]` scalar indexing plus
  `pbar.update(1)`) became one numpy fancy-index copy per anti-diagonal.
- **Affects:** the per-layer, pre-simulation build of the **DRAM prefetch order**. Which operand
  gets flattened depends on the dataflow: ifmap for `ws`, ifmap and filter for `os`, filter for
  `is`. Its cost grows with the operand matrix size.
- **Evidence:** own time 8.4 s → 2.7 s (Run A) and 6.4 s → 1.1 s (Run B). Only the `ws` variant
  ran; the `os` and `is` variants are the same edit.
- **Output change:** none.

### #3 O(1) dictionary lookup in `check_hit` (`CALC` mode)

- **Where:** `read_buffer_estimate_bw.py`.
  - `check_hit` was rewritten.
  - `manage_prefetches` maintains `addr_last_set_id`.
  - `__init__` creates it.
- **Change:** `check_hit` used to scan every set in the active window. Now it looks up the most
  recently finalized set that contains the address and checks whether that set is inside the
  window. Sets are only ever added with increasing IDs, so this gives the same answer.
- **Affects:** the **hit/miss decision for every ifmap and filter address, every cycle, in `CALC`
  mode** (the default `InterfaceBandwidth`). A miss triggers a DRAM prefetch, so this code decides
  fetch timing and the estimated bandwidth.
- **Evidence (Run A):**
  - `check_hit` own time: **232.6 s → 43.4 s (5.4×)** over 91 M calls.
  - `manage_prefetches` own time rose by 4.8 s because it now maintains the dictionary. Its
    cumulative time still fell 380 s → 148 s.
- **Output change:** none.

### #4a Reverse index for `active_buffer_hit` (`USER` mode, Run B only)

- **Where:** `read_buffer.py`.
  - `active_buffer_hit` was rewritten.
  - `prepare_hashed_buffer` builds `addr_to_lines`.
  - `__init__` and `reset` initialize it.
- **Change:** `hashed_buffer` never changes after it's built. So an address → line-IDs map is
  built once per layer, and the hit test checks only those lines against the window (including
  wrap-around). Before, it scanned every line in the window.
- **Affects:** the **hit/miss decision per address per cycle in `USER` mode**.
- **Evidence:**
  - 14.96 s → 9.06 s cumulative.
  - What's left is mostly the `any(...)` generator: about 5 steps per call, 22.9 M generator steps.
  - Building the index costs +1.35 s once per run, in `prepare_hashed_buffer`. It's the only
    changed function that got slower.
- **Output change:** none.

### #4b Vectorized `set_fetch_matrix` (`USER` mode, Run B only)

- **Where:** `read_buffer.py::set_fetch_matrix`.
- **Change:** the per-element copy loop (four `math.floor` calls per element) became
  `np.full` + `reshape`.
- **Affects:** the per-layer layout of the operand fetch matrix into rows of `req_gen_bandwidth`.
- **Evidence:** own time 5.18 s → 0.00 s. `math.floor` calls went from 17.2 M to 56.
- **Output change:** none.

### #5 Deferred assembly of the read-buffer DRAM trace

- **Where:**
  - `read_buffer_estimate_bw.py::prefetch` and `read_buffer.py::new_prefetch`
  - new `_finalize_trace_matrix` in both classes, called by `get_trace_matrix`,
    `get_external_access_start_stop_cycles`, and `print_trace`
- **Change:** each prefetch used to `np.concatenate` onto the whole `trace_matrix` so far, and in
  `CALC` mode also re-padded its columns. That's O(n²) copying per layer. Now chunks are collected
  in a list and joined once, on first read.
- **Affects:** the **IFMAP/FILTER DRAM read trace**, which feeds `*_DRAM_TRACE.csv` and the DRAM
  start/stop cycles in `BANDWIDTH_REPORT.csv`.
- **Evidence:**
  - Run A: **70.0 s → 6.1 s own time (11×)** over 10,558 prefetches, plus 1.0 s to finalize.
  - Run B: negligible, because there are only 140 prefetches.
  - The benefit grows with the number of prefetches per layer, which means **smaller SRAMs**
    (as in the SMM/SPM policies) and longer layers.
- **Output change:** none.

### #6 Growable backing array for the write-buffer DRAM trace

- **Where:** `write_buffer.py::append_to_trace_mat` and the new `_append_chunk_to_trace_matrix`.
- **Change:** the per-drain `np.concatenate` became a pre-allocated array that doubles when full.
  `trace_matrix` stays a live view of it, because `empty_drain_buf()` reads it mid-simulation.
- **Affects:** the **OFMAP DRAM write trace** (`OFMAP_DRAM_TRACE.csv` and the write start/stop
  cycles).
- **Evidence:** Run A: **57.1 s → 0.6 s** over 10,421 drains. Run B: negligible (239 drains).
- **Output change:** none.

### Other differences in the diff (not speed changes)

| Where | Change | Effect on results |
|---|---|---|
| `read_buffer_estimate_bw.py::set_params` | `num_items_per_set = max(1, floor(total_size_elems/100))` | Only when a read buffer holds **< 100 elements**. Vanilla computed 0 there and crashed in `complete_all_prefetches`. |
| `write_buffer.py::__init__`, `set_params` | `drain_buf_size = max(1, total - active)` | Only when the drain half rounds to 0 elements. Vanilla deadlocked on the first write. |
| `single_layer_sim.py::save_traces` | new `save_{ifmap,filter,ofmap}_trace=True` arguments | Defaults keep vanilla behaviour. |
| `scale_config.py` | removed a duplicate `get_bandwidths_as_list` definition | None. |

The first two are **robustness fixes for tiny buffers**, needed for the SMM/SPM policy work. At
16 KB and 64 KB they never fire, which is why both runs match vanilla exactly.

---

## 6. What to optimize next (not implemented)

From Run A's optimized profile:

1. **`write_buffer.store_to_trace_mat_cache`: 16.5%, now the #2 function, untouched so far.**
   `service_writes` calls it once **per written ofmap element** (85 M calls in Run A). Each call
   does a `.shape` check and a numpy scalar store into `current_line`, and every full line is
   `np.concatenate`d onto `trace_matrix_cache`. Two fixes would remove most of it:
   - write a whole cycle's row with one slice assignment instead of element by element;
   - pre-allocate the cache to `max_cache_lines` rows with a fill index, the same idea as #6.
2. **Trace CSV writing: 20.3%.** `np.savetxt(fmt='%s')` writes floats one row at a time. Integer
   formatting or a bulk writer is much faster, **but** it changes the CSV bytes. Correctness
   would then need a numeric comparison instead of a byte diff.
3. **Per-cycle Python overhead** in `service_memory_requests`, `service_reads`, and
   `manage_prefetches`. That's about 40% combined: 6.56 M cycles × 2 read buffers (13.1 M
   `service_reads` calls) × about 7 addresses each (91 M `manage_prefetches` calls).
   This is the structural limit described in `CORE_SIMULATION_PERFORMANCE.md`, and it needs
   batching or Cython.
4. (`USER` mode, Run B) Replace the `any()` generator in `active_buffer_hit` with a plain loop.
   That's about 5.6 s of 55 s.

## 7. Files

| File | Purpose |
|---|---|
| `benchmark/cprofile_compare/profile_network.py` | Run A driver: full network, frees each layer after its traces are written, md5s and deletes the trace CSVs, logs RSS per layer |
| `benchmark/cprofile_compare/run_network.sh` | Full-network runner: any topology, array, SRAM, `CALC`/`USER`, dataflow, repos, pythons; sequential or parallel with a RAM watchdog; writes `profiles/<name>_report.md` |
| `benchmark/cprofile_compare/configs/<name>.cfg` | Configs generated by `run_network.sh` (`googlenet_a16_s16_calc.cfg` = Run A) |
| `benchmark/cprofile_compare/profile_driver.py`, `run_both.sh` | Same for Run B (`CALC` + `USER`, 3 layers) |
| `benchmark/cprofile_compare/scale_{CALC,USER}.cfg`, `mobilenet3.csv` | Run B workload (these configs fix the `FilteexporrSRAMBankNum` typo in `configs/scale.cfg`) |
| `benchmark/cprofile_compare/breakdown.py` | Category % breakdown and top functions, vanilla vs. optimized (sections 1 and 2) |
| `benchmark/cprofile_compare/compare.py` | Per-changed-function table (section 4) and the top-40 text reports; `compare.py googlenet_a16_s16_calc_` for Run A |
| `benchmark/cprofile_compare/profiles/googlenet_a16_s16_calc_{vanilla,optimized}.pstats` | Run A raw profiles |
| `benchmark/cprofile_compare/profiles/{vanilla,optimized}.pstats` | Run B raw profiles |
| `benchmark/cprofile_compare/profiles/*_{tottime,cumtime}.txt` | Top-40 cProfile text reports per profile |

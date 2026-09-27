# cProfile: googlenet_a32_s32_user

- topology: `/data/grizos/Scale-Sim-SPM/topologies/conv_nets/Googlenet.csv`
- config: array 32x32, SRAM 32 KB, InterfaceBandwidth USER, dataflow ws (`/data/grizos/Scale-Sim-SPM/benchmark/cprofile_compare/configs/googlenet_a32_s32_user.cfg`)
- vanilla: `/data/grizos/SCALE-Sim`; optimized: `/data/grizos/Scale-Sim-SPM`; parallel=1; host wil; 2026-09-26T17:05:20+03:00
- vanilla: 1760.0s wall (under cProfile), peak RSS 2408 MB
- optimized: 426.7s wall (under cProfile), peak RSS 2588 MB

## Same simulated result?

- COMPUTE_REPORT.csv: identical
- BANDWIDTH_REPORT.csv: identical
- DETAILED_ACCESS_REPORT.csv: identical
- trace CSVs: identical (348 files, md5)

## Where the time goes

vanilla total 1747.6s, optimized total 413.8s, speedup 4.22x
(categories cover 100.0% / 100.0% of each run)

| Category | Vanilla s | Vanilla % | Optimized s | Optimized % | Saved s | Share of saving |
|---|---|---|---|---|---|---|
| DRAM trace building (in memory) | 1199.6 | 68.6% | 70.0 | 16.9% | 1129.6 | 84.7% |
| Trace CSV writing (savetxt) | 168.9 | 9.7% | 174.5 | 42.2% | -5.5 | -0.4% |
| Read hit/miss (USER) | 148.2 | 8.5% | 98.0 | 23.7% | 50.3 | 3.8% |
| tqdm progress-bar objects | 137.0 | 7.8% | 0.3 | 0.1% | 136.8 | 10.3% |
| Per-cycle memory loop + SRAM trace assembly | 30.7 | 1.8% | 26.3 | 6.4% | 4.5 | 0.3% |
| Write-buffer servicing | 27.5 | 1.6% | 19.2 | 4.6% | 8.3 | 0.6% |
| USER read-buffer setup | 25.0 | 1.4% | 19.0 | 4.6% | 6.0 | 0.4% |
| Prefetch-matrix diagonal flatten | 5.6 | 0.3% | 1.5 | 0.4% | 4.1 | 0.3% |
| Operand/demand matrix generation | 2.5 | 0.1% | 2.8 | 0.7% | -0.3 | -0.0% |
| Other (config/topology parsing, report files, driver) | 1.5 | 0.1% | 1.2 | 0.3% | 0.3 | 0.0% |
| Report statistics (start/stop cycles, counts) | 1.1 | 0.1% | 1.1 | 0.3% | -0.0 | -0.0% |
| **Total** | **1747.6** | 100% | **413.8** | 100% | **1333.8** | 100% |

Top 15 functions by exclusive time -- vanilla
| Function | Seconds | % of run |
|---|---|---|
| `read_buffer.py::new_prefetch` | 1122.1 | 64.2% |
| `tqdm (all functions)` | 137.0 | 7.8% |
| `read_buffer.py::print_trace` | 123.7 | 7.1% |
| `read_buffer.py::active_buffer_hit` | 97.4 | 5.6% |
| `write_buffer.py::store_to_trace_mat_cache` | 62.5 | 3.6% |
| `read_buffer.py::service_reads` | 50.7 | 2.9% |
| `double_buffered_scratchpad_mem.py::service_memory_requests` | 30.7 | 1.8% |
| `write_buffer.py::service_writes` | 26.8 | 1.5% |
| `read_buffer.py::set_fetch_matrix` | 18.2 | 1.0% |
| `write_buffer.py::print_trace` | 16.5 | 0.9% |
| `write_buffer.py::append_to_trace_mat` | 15.0 | 0.9% |
| `double_buffered_scratchpad_mem.py::print_ofmap_sram_trace` | 10.3 | 0.6% |
| `double_buffered_scratchpad_mem.py::print_filter_sram_trace` | 10.3 | 0.6% |
| `double_buffered_scratchpad_mem.py::print_ifmap_sram_trace` | 7.9 | 0.5% |
| `read_buffer.py::prepare_hashed_buffer` | 6.7 | 0.4% |

Top 15 functions by exclusive time -- optimized
| Function | Seconds | % of run |
|---|---|---|
| `read_buffer.py::print_trace` | 127.6 | 30.8% |
| `read_buffer.py::active_buffer_hit` | 64.2 | 15.5% |
| `write_buffer.py::store_to_trace_mat_cache` | 59.2 | 14.3% |
| `read_buffer.py::service_reads` | 33.6 | 8.1% |
| `double_buffered_scratchpad_mem.py::service_memory_requests` | 26.3 | 6.4% |
| `read_buffer.py::prepare_hashed_buffer` | 18.9 | 4.6% |
| `write_buffer.py::service_writes` | 18.5 | 4.5% |
| `write_buffer.py::print_trace` | 16.8 | 4.1% |
| `double_buffered_scratchpad_mem.py::print_filter_sram_trace` | 10.6 | 2.6% |
| `double_buffered_scratchpad_mem.py::print_ofmap_sram_trace` | 10.6 | 2.6% |
| `read_buffer.py::new_prefetch` | 9.4 | 2.3% |
| `double_buffered_scratchpad_mem.py::print_ifmap_sram_trace` | 8.6 | 2.1% |
| `systolic_compute_ws.py::create_ifmap_prefetch_mat` | 1.4 | 0.3% |
| `read_buffer.py::_finalize_trace_matrix` | 1.2 | 0.3% |
| `single_layer_sim.py::run` | 1.2 | 0.3% |

## Changed functions

```
vanilla   : 1747.60s   755,493,348 calls
optimized :  413.76s   604,718,187 calls
speedup   : 4.22x

                                                                  |   ----- vanilla -----    |   ---- optimized ----   
fix  function                                                     |      calls    tot    cum |      calls    tot    cum
-----------------------------------------------------------------------------------------------------------------------
#2   systolic_compute_ws.py::create_ifmap_prefetch_mat            |         58   5.36   6.25 |         58   1.02   1.44
#3   read_buffer_estimate_bw.py::check_hit                        |          0   0.00   0.00 |          0   0.00   0.00
#3   read_buffer_estimate_bw.py::manage_prefetches                |          0   0.00   0.00 |          0   0.00   0.00
#3   read_buffer_estimate_bw.py::service_reads                    |          0   0.00   0.00 |          0   0.00   0.00
#5   read_buffer_estimate_bw.py::prefetch                         |          0   0.00   0.00 |          0   0.00   0.00
#5   read_buffer_estimate_bw.py::_finalize_trace_matrix           |          0   0.00   0.00 |          0   0.00   0.00
#4a  read_buffer.py::active_buffer_hit                            | 49,556,151  97.37  97.37 | 49,556,151  29.41  64.18
#4a  read_buffer.py::<genexpr>                                    |          0   0.00   0.00 | 215,362,858  16.56  16.56
#4a  read_buffer.py::prepare_hashed_buffer                        |        116   5.83   7.22 |        116  14.76  18.94
#4b  read_buffer.py::set_fetch_matrix                             |        116  14.08  25.45 |        116   0.02  18.98
#1   read_buffer.py::service_reads                                |  3,975,958  41.16 1362.21 |  3,975,958  29.27 107.40
#5   read_buffer.py::new_prefetch                                 |     23,373  18.62 1122.23 |     23,373   8.26   9.43
#5   read_buffer.py::_finalize_trace_matrix                       |          0   0.00   0.00 |        232   0.06   1.22
#1   write_buffer.py::service_writes                              |  1,987,979  23.09 148.99 |  1,987,979  16.90  78.83
#6   write_buffer.py::append_to_trace_mat                         |      5,367   0.18  15.00 |      5,367   0.01   0.19
#6   write_buffer.py::_append_chunk_to_trace_matrix               |          0   0.00   0.00 |      5,365   0.16   0.16
#1   std.py::__init__                                             |  5,964,054  14.05  94.63 |         59   0.00   0.01
--   double_buffered_scratchpad_mem.py::service_memory_requests   |         58  24.67 1542.35 |         58  20.66 212.86

full reports written to /data/grizos/Scale-Sim-SPM/benchmark/cprofile_compare/profiles/googlenet_a32_s32_user_<version>_{tottime,cumtime}.txt
```

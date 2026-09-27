# cProfile: smokeconv_a32_s64_calc

- topology: `/tmp/claude-1000/-home-george-Desktop-SCALE-Sim/64074a31-523b-4d83-b5cc-a9ffc24380d8/scratchpad/smoke/smokeconv.csv` (conv)
- config: array 32x32, SRAM 64 KB, InterfaceBandwidth CALC, dataflow ws (`/home/george/Desktop/SCALE-Sim/benchmark/cprofile_compare/configs/smokeconv_a32_s64_calc.cfg`)
- vanilla: `/home/george/Music/SCALE-Sim`; optimized: `/home/george/Desktop/SCALE-Sim`; parallel=1; host george-HP-Laptop-15-fc0xxx; 2026-09-26T21:29:33+03:00
- vanilla: 5.0s wall (under cProfile), peak RSS 89 MB
- optimized: 3.1s wall (under cProfile), peak RSS 111 MB

## Same simulated result?

- COMPUTE_REPORT.csv: identical
- BANDWIDTH_REPORT.csv: identical
- DETAILED_ACCESS_REPORT.csv: identical
- trace CSVs: identical (6 files, md5)

## Where the time goes

vanilla total 5.0s, optimized total 3.0s, speedup 1.64x
(categories cover 100.0% / 100.0% of each run)

| Category | Vanilla s | Vanilla % | Optimized s | Optimized % | Saved s | Share of saving |
|---|---|---|---|---|---|---|
| Read hit/miss + prefetch decisions (CALC) | 2.2 | 44.1% | 1.1 | 35.1% | 1.1 | 58.2% |
| DRAM trace building (in memory) | 0.7 | 14.3% | 0.6 | 20.9% | 0.1 | 4.2% |
| tqdm progress-bar objects | 0.6 | 11.4% | 0.0 | 0.3% | 0.6 | 28.5% |
| Trace CSV writing (savetxt) | 0.5 | 10.2% | 0.6 | 19.1% | -0.1 | -3.5% |
| Write-buffer servicing | 0.3 | 7.0% | 0.2 | 8.0% | 0.1 | 5.4% |
| Per-cycle memory loop + SRAM trace assembly | 0.3 | 6.6% | 0.3 | 10.5% | 0.0 | 0.7% |
| Prefetch-matrix diagonal flatten | 0.2 | 3.4% | 0.0 | 1.4% | 0.1 | 6.4% |
| Operand/demand matrix generation | 0.1 | 2.2% | 0.1 | 3.3% | 0.0 | 0.4% |
| Report statistics (start/stop cycles, counts) | 0.0 | 0.4% | 0.0 | 0.7% | -0.0 | -0.1% |
| Other (config/topology parsing, report files, driver) | 0.0 | 0.3% | 0.0 | 0.7% | -0.0 | -0.2% |
| **Total** | **5.0** | 100% | **3.0** | 100% | **1.9** | 100% |

Top 15 functions by exclusive time -- vanilla
| Function | Seconds | % of run |
|---|---|---|
| `read_buffer_estimate_bw.py::check_hit` | 1.6 | 31.4% |
| `write_buffer.py::store_to_trace_mat_cache` | 0.6 | 12.9% |
| `tqdm (all functions)` | 0.6 | 11.4% |
| `write_buffer.py::service_writes` | 0.3 | 7.0% |
| `double_buffered_scratchpad_mem.py::service_memory_requests` | 0.3 | 6.6% |
| `read_buffer_estimate_bw.py::service_reads` | 0.3 | 6.4% |
| `read_buffer_estimate_bw.py::manage_prefetches` | 0.3 | 6.2% |
| `systolic_compute_ws.py::create_ifmap_prefetch_mat` | 0.2 | 3.4% |
| `read_buffer_estimate_bw.py::print_trace` | 0.1 | 2.9% |
| `write_buffer.py::print_trace` | 0.1 | 2.6% |
| `double_buffered_scratchpad_mem.py::print_ofmap_sram_trace` | 0.1 | 1.9% |
| `double_buffered_scratchpad_mem.py::print_ifmap_sram_trace` | 0.1 | 1.5% |
| `double_buffered_scratchpad_mem.py::print_filter_sram_trace` | 0.1 | 1.2% |
| `read_buffer_estimate_bw.py::prefetch` | 0.1 | 1.2% |
| `operand_matrix.py::calc_ifmap_elem_addr` | 0.1 | 1.0% |

Top 15 functions by exclusive time -- optimized
| Function | Seconds | % of run |
|---|---|---|
| `write_buffer.py::store_to_trace_mat_cache` | 0.6 | 18.9% |
| `read_buffer_estimate_bw.py::manage_prefetches` | 0.4 | 13.3% |
| `read_buffer_estimate_bw.py::check_hit` | 0.3 | 11.5% |
| `double_buffered_scratchpad_mem.py::service_memory_requests` | 0.3 | 10.5% |
| `read_buffer_estimate_bw.py::service_reads` | 0.3 | 10.3% |
| `write_buffer.py::service_writes` | 0.2 | 8.0% |
| `read_buffer_estimate_bw.py::print_trace` | 0.2 | 5.6% |
| `write_buffer.py::print_trace` | 0.1 | 4.8% |
| `double_buffered_scratchpad_mem.py::print_ifmap_sram_trace` | 0.1 | 3.2% |
| `double_buffered_scratchpad_mem.py::print_ofmap_sram_trace` | 0.1 | 2.8% |
| `double_buffered_scratchpad_mem.py::print_filter_sram_trace` | 0.1 | 2.4% |
| `operand_matrix.py::calc_ifmap_elem_addr` | 0.1 | 2.0% |
| `read_buffer_estimate_bw.py::prefetch` | 0.1 | 1.7% |
| `systolic_compute_ws.py::create_ifmap_prefetch_mat` | 0.0 | 1.4% |
| `single_layer_sim.py::run` | 0.0 | 0.6% |

## Changed functions

```
vanilla   :    4.96s     3,456,985 calls
optimized :    3.02s     2,754,506 calls
speedup   : 1.64x

                                                                  |   ----- vanilla -----    |   ---- optimized ----   
fix  function                                                     |      calls    tot    cum |      calls    tot    cum
-----------------------------------------------------------------------------------------------------------------------
#2   systolic_compute_ws.py::create_ifmap_prefetch_mat            |          1   0.17   0.19 |          1   0.04   0.04
#3   read_buffer_estimate_bw.py::check_hit                        |    405,504   1.49   1.56 |    405,504   0.22   0.35
#3   read_buffer_estimate_bw.py::manage_prefetches                |    405,504   0.28   1.93 |    405,504   0.36   0.80
#3   read_buffer_estimate_bw.py::service_reads                    |     25,840   0.32   2.25 |     25,840   0.31   1.11
#5   read_buffer_estimate_bw.py::prefetch                         |         14   0.04   0.06 |         14   0.02   0.05
#5   read_buffer_estimate_bw.py::_finalize_trace_matrix           |          0   0.00   0.00 |          4   0.00   0.00
#4a  read_buffer.py::active_buffer_hit                            |          0   0.00   0.00 |          0   0.00   0.00
#4a  read_buffer.py::<genexpr>                                    |          0   0.00   0.00 |          0   0.00   0.00
#4a  read_buffer.py::prepare_hashed_buffer                        |          0   0.00   0.00 |          0   0.00   0.00
#4b  read_buffer.py::set_fetch_matrix                             |          0   0.00   0.00 |          0   0.00   0.00
#1   read_buffer.py::service_reads                                |          0   0.00   0.00 |          0   0.00   0.00
#5   read_buffer.py::new_prefetch                                 |          0   0.00   0.00 |          0   0.00   0.00
#5   read_buffer.py::_finalize_trace_matrix                       |          0   0.00   0.00 |          0   0.00   0.00
#1   write_buffer.py::service_writes                              |     12,920   0.29   1.53 |     12,920   0.22   0.82
#6   write_buffer.py::append_to_trace_mat                         |         25   0.01   0.01 |         25   0.00   0.00
#6   write_buffer.py::_append_chunk_to_trace_matrix               |          0   0.00   0.00 |         14   0.00   0.00
#1   std.py::__init__                                             |     12,923   0.06   0.37 |          2   0.00   0.01
--   double_buffered_scratchpad_mem.py::service_memory_requests   |          1   0.27   4.11 |          1   0.26   2.25

full reports written to /home/george/Desktop/SCALE-Sim/benchmark/cprofile_compare/profiles/smokeconv_a32_s64_calc_<version>_{tottime,cumtime}.txt
```

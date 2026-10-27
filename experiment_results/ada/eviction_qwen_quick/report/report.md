# GPU KV-cache eviction results

Model: `Qwen/Qwen2.5-3B-Instruct`. Revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Family: `qwen2`. Dtype: `torch.float16`.
GPU: NVIDIA GeForce GTX 1080 Ti. Mode: `chunked_prefill_matched`.

All methods use identical chunked prefill. This full-precision reference is separate from the original monolithic baseline.
H2O-style uses explicit attention scoring and prunes after each chunk. Recent/sink policies apply retention within each chunk.
H2O-style shares per-layer selections across heads and batch members. It is an adaptation, not an exact paper reproduction.
Kernel choice, scoring, and pruning overhead differ. Compare measured latency without attributing differences solely to token selection.
Cache bytes include retained-position metadata. Allocator peaks measure the whole process, including model weights.
No claim of acceleration follows from persistent storage reduction alone.

## Coverage

Recorded attention fallback calls: **0**.
Fallback kernels can change latency and peak memory. Inspect kernel support before interpreting those measurements.

Complete grid attempted: **True**. Budget reached: **False**.

| Stage | Planned cases | Recorded cases |
|---|---:|---:|
| needle | 21 | 21 |
| ppl | 7 | 7 |
| sweep | 7 | 7 |

Recorded failures: 0. See failures.csv for exact error messages.

Missing cells have no result. OOM and unexpected errors are excluded from successful averages.
Quality means can cover different cases after failures. Use matched_comparisons.csv for paired comparisons only.

Timing and memory sweep summaries use medians across successful trials.
needle_by_depth.csv preserves information-position effects. needle_failure_cases.csv pairs each method with the full-cache reference.
Failure categories describe associations with code-token retention. They do not establish causation.

## Measurements

| Stage | Context | Batch | Method | Success / recorded | Cache MiB | TTFT ms | TPOT ms | Strict accuracy | PPL |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| needle | 1024 | 1 | full_fp16 | 3 / 3 | 36.423 | 1772.726 | 100.494 | 1.000 | — |
| needle | 1024 | 1 | h2o_style_r0.05 | 3 / 3 | 1.814 | 941.248 | 115.666 | 0.000 | — |
| needle | 1024 | 1 | h2o_style_r0.5 | 3 / 3 | 18.211 | 980.048 | 116.813 | 0.333 | — |
| needle | 1024 | 1 | recent_r0.05 | 3 / 3 | 1.807 | 1311.493 | 55.861 | 0.333 | — |
| needle | 1024 | 1 | recent_r0.5 | 3 / 3 | 18.141 | 1676.973 | 67.165 | 0.000 | — |
| needle | 1024 | 1 | sink_window_r0.05 | 3 / 3 | 1.807 | 1360.636 | 62.263 | 0.333 | — |
| needle | 1024 | 1 | sink_window_r0.5 | 3 / 3 | 18.141 | 1700.971 | 80.827 | 0.333 | — |
| ppl | 2048 | 1 | full_fp16 | 1 / 1 | 72.527 | — | — | — | 6.674 |
| ppl | 2048 | 1 | h2o_style_r0.05 | 1 / 1 | 3.628 | — | — | — | 10.278 |
| ppl | 2048 | 1 | h2o_style_r0.5 | 1 / 1 | 36.422 | — | — | — | 6.682 |
| ppl | 2048 | 1 | recent_r0.05 | 1 / 1 | 3.614 | — | — | — | 109.178 |
| ppl | 2048 | 1 | recent_r0.5 | 1 / 1 | 36.281 | — | — | — | 14.796 |
| ppl | 2048 | 1 | sink_window_r0.05 | 1 / 1 | 3.614 | — | — | — | 13.989 |
| ppl | 2048 | 1 | sink_window_r0.5 | 1 / 1 | 36.281 | — | — | — | 6.702 |
| sweep | 1024 | 1 | full_fp16 | 1 / 1 | 36.529 | 1741.910 | 98.930 | — | — |
| sweep | 1024 | 1 | h2o_style_r0.05 | 1 / 1 | 1.814 | 934.779 | 115.741 | — | — |
| sweep | 1024 | 1 | h2o_style_r0.5 | 1 / 1 | 18.211 | 970.953 | 117.111 | — | — |
| sweep | 1024 | 1 | recent_r0.05 | 1 / 1 | 1.807 | 1292.405 | 55.869 | — | — |
| sweep | 1024 | 1 | recent_r0.5 | 1 / 1 | 18.141 | 1655.316 | 66.521 | — | — |
| sweep | 1024 | 1 | sink_window_r0.05 | 1 / 1 | 1.807 | 1359.562 | 61.760 | — | — |
| sweep | 1024 | 1 | sink_window_r0.5 | 1 / 1 | 18.141 | 1682.575 | 80.137 | — | — |

Needle accuracy uses whitespace-normalized exact codes. Substring accuracy is retained as a separate legacy metric.
Code-token retention uses actual token offsets and layer-zero positions after prefill when available.
Every perplexity method uses identical windows, explicit prefix policy, chunk size, and token weighting.
Timing trials use identical seeded random token prompts. Fixed batches do not measure continuous-serving concurrency.
Profiler rows and traces are instrumented runs. They are excluded from timing and quality tables.

![sweep_batch1.png](sweep_batch1.png)

![needle_batch1.png](needle_batch1.png)

![ppl_batch1.png](ppl_batch1.png)

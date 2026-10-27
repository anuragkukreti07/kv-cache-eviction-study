# Experiment results

## Ada Qwen experiments

- `ada/eviction_qwen_1080ti_b1_v1/`: Qwen2.5-3B-Instruct FP16 results on a GTX 1080 Ti.
- Contexts: 1K, 2K, and 4K. Batch size: 1.
- Policies: recent-window, sink-plus-window, and H2O-style, plus a matched full-cache reference.
- Retention budgets: 50%, 25%, 10%, and 5%.
- Recorded measurements: 4,173 successful cases, with no recorded OOMs or errors.
- `ada/eviction_qwen_quick/`: the preliminary quick run.
- `ada/gpu_family_check.json`: tiny random-model GPU checks for LLaMA, Qwen2, and Mistral.

Open `ada/eviction_qwen_1080ti_b1_v1/report/report.md` for tables and plots.
Raw measurements, configuration, source hashes, and completion records remain beside each report.

## Other included artifacts

- `results/`: earlier LLaMA laptop and L40S server baselines, with their comparison reports.
- `ada/laptop_rtx5060_1b/`: a copy of the earlier laptop baseline.
- `ada/test_oom/`: tiny-model memory-failure tests, not real-checkpoint quality measurements.

## Comparison limits

Real-checkpoint LLaMA and Mistral eviction experiments are still pending.
Tiny-model checks do not establish their real-checkpoint quality or performance.
Keep older LLaMA baselines separate from the new Qwen timing results because hardware and models differ.
H2O-style is an adaptation with different attention and pruning operations.
Cache-storage reduction does not imply the same reduction in total GPU memory.

Local log files and macOS system files are excluded from Git.

# Baseline runs side by side

| metric | laptop_rtx5060_1b | server_3b |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | NVIDIA L40S |
| model | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct |
| weights (GiB) | 2.3 | 5.98 |
| KV per token (KiB) | 32.0 | 112.0 |
| formula error (max %) | 0.0 | 0.0 |
| batch 1: largest ctx / first OOM | 16K / none | 64K / none |
| batch 2: largest ctx / first OOM | 16K / none | 64K / none |
| batch 4: largest ctx / first OOM | 8K / 16K | 32K / 64K |
| batch 8: largest ctx / first OOM | 4K / 8K | 16K / 32K |
| b1 4K TTFT (ms) | 390.8 | 166.1 |
| b1 4K TPOT (ms) | 17.96 | 19.02 |
| b1 16K TTFT (ms) | 2442.0 | 971.2 |
| b1 16K TPOT (ms) | 33.23 | 19.02 |
| sweep trials (min per point) | 3 | 3 |
| read bandwidth (GB/s) | 348 | 745 |
| needle accuracy | 100% of 250 | 98% of 350 |
| needle 1K | 100% | 98% |
| needle 2K | 100% | 96% |
| needle 4K | 100% | 98% |
| needle 8K | 100% | 94% |
| needle 16K | 100% | 100% |
| ppl 2K | 12.57 |  |
| ppl 4K | 12.72 |  |
| b1 32K TTFT (ms) |  | 2511.8 |
| b1 32K TPOT (ms) |  | 27.48 |
| needle 32K |  | 100% |
| needle 64K |  | 100% |
| ppl 2K (BOS) |  | 9.74 |
| ppl 4K (BOS) |  | 9.8 |
| ppl 8K (BOS) |  | 9.26 |
| ppl 16K (BOS) |  | 8.51 |
| ppl 32K (BOS) |  | 8.55 |

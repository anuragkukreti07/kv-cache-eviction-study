# Review of the laptop baseline, and what changed for the server

## Verdict on the laptop run

The laptop baseline (Llama-3.2-1B, RTX 5060 Laptop 8 GB, `results/laptop_rtx5060_1b/`) is methodologically sound and
fine to present as the baseline. Checked:

- **Cache accounting:** measured bytes = formula on all 51 points (0.000% error); steady-state memory after prefill
  equals the cache exactly.
- **Memory views:** peak is 3.0x the cache at every point. That matches the expected prefill temporaries for this
  model (MLP gate/up/activation ≈ 1.5x the cache per token, plus the repeated K/V in attention ≈ 0.5x), so the
  report's explanation is right.
- **Out-of-memory frontier** (batch 4 x 16K, batch 8 x 8K) is consistent with weights + 3x cache against the 7.16 GiB
  allocator cap.
- **Latency:** warm-up, synchronised timing, median of 3 trials, TTFT and TPOT separated.
- **Perplexity:** the chunked harness math is correct.
- **Needle:** prompts are exactly the requested length, and the answer is never in the filler.
- **Attention patch:** the GQA folding is numerically correct; I re-checked it, including the 3B head layout (24 query
  heads, 8 KV heads). No call fell back to the slow kernel.

Small things to fix or say in the presentation:

1. **TPOT at long context is limited by the attention kernel, not bandwidth.**
   - The data shows batch 1 x 16K (3.0 GB read per step) at 33 ms, but batch 8 x 4K (3.55 GB) at only 27 ms.
   - Decode time depends on the context length of a single sequence more than on bytes read.
   - Likely cause: the decode attention kernel parallelises over batch x heads only, so one long sequence uses only a
     few GPU blocks.
   - This supports the report's "kernel inefficiency" hypothesis with evidence; it is worth one sentence.
2. The out-of-memory takeaway compared against 7.96 GiB instead of the 7.16 GiB cap. Fixed in the report script.
3. The perplexity windows did not start with a BOS token. That matters later, because sink-based eviction keeps the
   first tokens. Added `ppl.bos` (on for the 3B server run, off for the 1B runs, so they stay comparable).
4. The Llama-3 chat template inserts *today's date*, so needle prompts changed from day to day. Now fixed to one date.
5. `model_revision.lock` held only one model, so switching to 3B would overwrite the 1B pin. It now holds one line per
   model.
6. `s0_check_env.py` ignored `--config`. Fixed.

## Changes for the server

| Change | Why |
|---|---|
| `configs/server_3b.yaml`: Llama-3.2-3B, contexts 1K-64K, batches 1-8 | The final model from the proposal; with 48 GB the cache (112 KiB/token) becomes the bottleneck, and the frontier (64K x 4, 32K x 8) is visible |
| `configs/server_1b.yaml` | Same model and settings as the laptop, so GPU differences can be shown |
| `setup_server.sh`, `run_server.sh` | Linux one-command setup and run; picks the CUDA build from the driver; picks the free GPU |
| Time budget (`--until`, per-stage `max_minutes`, `kvstudy/deadline.py`) | The run stops cleanly before the slot ends; the report is always built |
| Trial-first sweep order, sample-first needle order | If time runs out, every point/cell already has data instead of some having none |
| Per-prompt seeds for needle and sweep inputs | Identical prompts after resuming or reordering |
| `scripts/prefetch.py`, offline runs | Models and wikitext are downloaded and pinned once; runs do not wait on the network |
| `s0 --model-check` | Loads the real model and answers one 2K needle before any long run (catches download, FP16 and tokenizer problems in the first minute) |
| `scripts/compare_runs.py` | One table: laptop 1B vs server 1B vs server 3B |
| `transformers==5.19.0` pinned | Same version as the laptop run |
| `tests/` | CPU smoke tests with a tiny random model, including simulated out-of-memory and resume-after-time-limit |

## Tested here

- The whole pipeline ran end to end on CPU with a tiny random Llama-style model (GQA 4:2):
  `run_all.py`, `run_server.sh` (with a stand-in `nvidia-smi`), `compare_runs.py` and the report.
- The time-limit stop, resume without duplicates, and out-of-memory skipping were tested with simulated OOMs.
- Not tested here: a real GPU or real model downloads. That is what `./run_server.sh --quick` checks in the first
  minutes of the slot.

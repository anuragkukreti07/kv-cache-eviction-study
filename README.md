# KV-cache eviction study: baseline (unmodified FP16 full cache)

## Multi-family extension on `sbidisha_dev`

The extension adds Qwen2.5-3B and Mistral-7B full-cache baselines alongside the existing LLaMA baseline.
It adds physical recent-token and sink-plus-window eviction at 50%, 25%, 10%, and 5% retention.
FP16 is the default. BF16 is supported on compatible CUDA hardware.
See [MODEL_STUDY.md](MODEL_STUDY.md) for setup, GPU checks, commands, and comparison limits.
The original LLaMA configurations and launchers remain available below.
Their runtime estimates have not been verified for these new experiments.

Produces the numbers every eviction policy will later be compared against
(proposal RQ3 baseline, guide Step 3 "Baseline and bottleneck", plus the full-cache quality reference for T1-T3).

| Run | Model | Machine | Config | Results |
|---|---|---|---|---|
| Laptop baseline (done) | Llama-3.2-1B-Instruct | RTX 5060 Laptop 8 GB, Windows | `configs/baseline.yaml` | `results/laptop_rtx5060_1b/` |
| **Server baseline** | **Llama-3.2-3B-Instruct** | college GPU (RTX PRO 5000 48 GB), Linux | `configs/server_3b.yaml` | `results/server_3b/` |
| Server comparison | Llama-3.2-1B-Instruct | college GPU | `configs/server_1b.yaml` | `results/server_1b/` |

---

## On the college GPU server (Linux) — step by step

Everything below happens inside your slot (login is refused outside it). Keep the terminal open the whole time:
logging out, or the SSH connection dropping, stops the run. Results are saved after every single measurement, so
nothing finished is ever lost, and rerunning the same command resumes where it stopped.

**Before the slot (on your laptop):** keep `kv_study.zip` exactly as provided. If you edit a `.sh` file on Windows,
run `sed -i 's/\r$//' *.sh` on the server afterwards (Windows line endings break bash scripts).

**1. Log in with a keep-alive (stops idle disconnects) and copy the code over** (from your laptop):

```bash
scp kv_study.zip team20@10.4.25.63:/data/cs3406/team20/
ssh -o ServerAliveInterval=30 team20@10.4.25.63
```

**2. On the server:**

```bash
cd /data/cs3406/team20
unzip -q kv_study.zip && cd kv_study
chmod +x setup_server.sh run_server.sh
./setup_server.sh              # first slot only: ~10-20 min (PyTorch ~3 GB, models ~9 GB, checks the GPU)
./run_server.sh --quick        # 3-5 min smoke test on the real 3B model
./run_server.sh --until 00:00  # full baseline; 00:00 = the time your slot ENDS (new work stops 12 min earlier)
```

`--until` makes the run finish safely inside the slot: each stage also has its own cap (sweep 35 min, needle 25 min,
perplexity 10 min). With ~70+ minutes available the full 3B baseline finishes, and the 1B comparison runs after it
if at least 10 minutes are left. Expected 3B run time on a 48 GB Blackwell/Ada card: about 40-70 minutes.

**3. Copy the results to your laptop** (the run prints this command; run it on your laptop before the slot ends):

```bash
scp team20@10.4.25.63:/data/cs3406/team20/kv_study/results_<date>_<time>.tar.gz .
```

Open `results/server_3b/report/baseline_report.html`. `results/comparison.md` puts laptop 1B, server 1B and server
3B side by side.

**If something fails**

| Message | What to do |
|---|---|
| `[FAIL] ... no internet access to huggingface.co` in setup | The server cannot download models. Tell the team; fallback is downloading on a laptop and copying `.hf_cache` over. |
| `Python 3.10+ not found` | Send the output of `python3 --version` and `ls /usr/bin/python*`. |
| `inf/NaN in this dtype` | Edit `configs/server_3b.yaml`: `dtype: bfloat16`, then rerun. |
| `other processes are using the GPU` | Another job is on the card; numbers will be off. Check `nvidia-smi`. |
| Slot ended mid-run | Nothing to fix: next slot run the same `./run_server.sh --until ...` and it resumes. |
| Model loading fails offline | `KV_ONLINE=1 ./run_server.sh --until 00:00` (loads with network access). |

---

## On the laptop (Windows)

1. `setup.bat` (once). Creates `.venv`, installs PyTorch cu128 (needed for Blackwell / sm_120) and the requirements,
   then checks the GPU and model access.
2. `run_baseline.bat --quick` for a few-minute smoke test, then `run_baseline.bat` for the full baseline.

Options (both machines, via `python scripts/run_all.py`): `--config`, `--quick`, `--skip needle,ppl`, `--fresh`
(delete old results of that run), `--minutes N` or `--deadline <unix time>` (time budget).

## What you get (`results/<run>/report/`)

| File | What it shows |
|---|---|
| `baseline_report.html` | Everything in one page (open this for the presentation) |
| `fig1_cache_growth.png` | Measured vs. formula KV cache by context x batch, against weights and GPU memory; X = out of memory |
| `fig2_memory_views.png` | Peak (prefill transients included) vs. steady-state memory |
| `fig3_latency.png` | Time to first token and time per output token |
| `fig4_roofline.png` | Decode step vs. the measured memory-bandwidth limit |
| `fig5_needle_heatmap.png` | Full-cache needle accuracy, depth x context (the ceiling for T1) |
| `fig6_perplexity.png` | Full-cache perplexity through the chunked harness (T2/T3 baseline) |
| `baseline_summary.md`, `*.csv` | Key numbers and tables; raw per-run lines are in `results/<run>/*.jsonl` |

Raw results are append-only JSONL, one line per run, with GPU / torch / transformers versions.

## How it maps to the guide

- Step 3.1 formula check: every sweep point logs measured cache bytes next to `2 x L x H x d x S x B x bytes` (error %).
- Step 3.2 memory: three views, theoretical cache, measured peak/steady allocated, reserved + device-used.
- Step 3.3 latency: warm-up, CUDA sync around timing, TTFT and TPOT separately, greedy decoding, fixed seed, median of 3 trials.
- Step 3.4 growth curve: context x batch until the first out-of-memory per batch.
- Step 3.5 bytes per decode step: weights + whole cache per step, compared with measured bandwidth.
- Step 4 workloads: needle (seeded per prompt, token-exact depth, answer positions stored) and chunked perplexity.

## Attention kernels and memory (important)

With Llama-3.2's grouped-query attention, PyTorch can fall back to the `math` SDPA kernel, which builds the full
S x S score matrix per head. Memory and prefill time then grow with S^2 and swamp the KV cache. `kvstudy/attention.py`
keeps SDPA on flash / memory-efficient kernels (decode folds the query-head groups, prefill repeats KV), and
`s0_check_env.py` verifies both numerics and 4K-prefill memory. `vram_fraction` caps the allocator so a full GPU gives a
clean out-of-memory instead of Windows spilling into shared system memory.

## Things to say honestly in the presentation

- The sweep uses random token ids (fine for memory/latency, not for quality). Quality uses the needle and wikitext-2 runs.
- Filler text is wikitext-2, not PG-19 as in the guide (small download, no login).
- Needle uses up to 10 prompts per cell (guide: 20-30 for final numbers).
- An out-of-memory point can be caused by prefill activations, not only the cache; fig2 shows both views. Chunked prefill (T9) is the study that addresses this.
- Server 3B perplexity windows start with BOS (`ppl.bos: true`); the laptop 1B run did not, so those perplexities are
  not directly comparable (the server 1B run keeps the laptop setting so it is).

## Layout

`kvstudy/` library (`modelio.py` cache accounting + plain decode loop, `budget.py`, `resultlog.py`, `deadline.py`,
`workloads/`), `scripts/` stage scripts (`s0`..`s4`, `make_report.py`, `run_all.py`, `prefetch.py`, `compare_runs.py`),
`configs/`, `results/`, `tests/` (CPU smoke tests with a tiny random model: `python tests/make_tiny_model.py`, then
`python scripts/run_all.py --config tests/test_cpu.yaml` and `python tests/test_oom_and_budget.py`).
See `CHANGES.md` for what changed after the laptop run and why.

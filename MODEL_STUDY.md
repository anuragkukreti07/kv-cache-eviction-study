# Multi-family eviction study

Work belongs to `sbidisha_dev`. The original LLaMA configuration files remain unchanged.
This extension studies eviction strategies. It does not add quantization.

## What was already present

The baseline loads Hugging Face causal models through automatic model classes.
It measures full-cache memory, latency, bandwidth, needle retrieval, and perplexity.
It includes reports, revision pinning, saved results, time budgets, and resume support.
Its supplied server configurations targeted LLaMA models.

## What this branch adds

| Area | Added behavior |
|---|---|
| Model families | Explicit LLaMA, Qwen2, and Mistral checks |
| New checkpoints | Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3 |
| Precision | FP16 defaults; explicit BF16 hardware validation |
| Model safety | Context checks and rejection of active native sliding windows during eviction |
| Quality inputs | Chat-template handling and explicit perplexity prefix policy |
| Eviction | Physical recent-token, sink-plus-window, and H2O-style attention retention |
| Budgets | 50%, 25%, 10%, and 5% of prompt/window length |
| Matched reference | Full cache using the same chunked prefill as eviction |
| Results | Durable measurements, resume identity checks, paired reports, plots, and failure tables |
| Validation | Tiny CPU tests and a download-free CUDA check for all three families |

Full eviction runs use 13 methods: one full-cache reference and 12 policy/budget combinations.
Quick runs use seven methods: full cache, plus three policies at 50% and 5%.
Sink-plus-window keeps four initial tokens. Remaining slots retain recent tokens.
Budget tokens equal `floor(context_length * ratio)`. Budgets stay fixed during decoding.
Absolute token positions remain unchanged after eviction.
Persistent cache storage stays bounded. Temporary chunks can increase peak memory.
The implementation uses Python cache operations and explicit attention masks.
It does not claim kernel-level optimization or guaranteed acceleration.

## Attention-based policy

H2O-style retention adds the missing comparison between attention importance and recency.
Each layer accumulates causal attention probabilities for its retained tokens.
Scores average query heads and batch members. All heads within a layer share selected positions.
Half the budget protects recent tokens, rounded down with at least one recent slot.
The remaining slots retain older tokens with the largest accumulated scores.
Equal scores prefer earlier positions. Selected positions stay in chronological order.
The policy preserves FP16 or BF16 K/V. Its score metadata uses FP32 and counts toward storage.

This is a chunked H2O-style adaptation, not an exact reproduction of the published algorithm.
It prunes after each prefill chunk and each decoding token.
The recent and sink policies apply their retention masks within each prefill chunk.
That difference can affect quality. Keep the configured chunk size in reported results.
H2O-style uses explicit attention probabilities. The other methods use the configured attention backend.
Measured timing includes score collection, pruning, and backend switching.
Kernel and pruning differences prevent attributing latency differences solely to token selection.
Layer-zero retention diagnostics do not describe every layer's selected tokens.

SnapKV and PyramidKV are not implemented in this branch.
SnapKV uses prompt observation windows and per-head token selection.
PyramidKV also allocates different budgets across layers.
Those methods need separate lifecycle and allocation checks before meaningful comparisons.
H2O-style completes the initial three-policy comparison without adding quantization.
No policy is claimed to perform best before the GPU experiments.

The three eviction YAMLs select policies through `eviction.policies`.
Allowed values are `recent`, `sink_window`, and `h2o`.
Quick configurations contain their own policy list. Update both lists when changing a run.
Result labels use `h2o_style_r0.05`, for example, to identify the adaptation.
Use a new output directory after this code change. Older manifests reject the changed implementation.

## Configurations

| Family | Checkpoint | Full-cache baseline | Eviction grid |
|---|---|---|---|
| LLaMA | unsloth/Llama-3.2-3B-Instruct | `configs/server_3b.yaml` | `configs/eviction_llama.yaml` |
| Qwen2 | Qwen/Qwen2.5-3B-Instruct | `configs/server_qwen_3b.yaml` | `configs/eviction_qwen.yaml` |
| Mistral | mistralai/Mistral-7B-Instruct-v0.3 | `configs/server_mistral_7b.yaml` | `configs/eviction_mistral.yaml` |

The new grids cover 1K–16K tokens, within the selected checkpoints' context limits.
The original LLaMA baseline grid remains unchanged.
Use the eviction grids for common-context comparisons across all three models.
Each checkpoint resolves to an exact revision before measurements are recorded.
The local revision lock records the resolved revisions.

## Team 20: remote GPU session

Login works only during your confirmed booking. Keep SSH open throughout the run.
The server stops programs when the slot ends or you log out.
Keep project files under `/data/cs3406/team20`.
Do not rely on `nohup`, `screen`, or `tmux` to continue after logout.

From your laptop, copy this branch to the server:

```bash
git archive --format=tar.gz --output=/tmp/kv-cache-eviction-study.tar.gz sbidisha_dev
scp /tmp/kv-cache-eviction-study.tar.gz team20@10.4.25.63:/data/cs3406/team20/
ssh -o ServerAliveInterval=30 team20@10.4.25.63
```

On the server:

```bash
mkdir -p /data/cs3406/team20/kv-cache-eviction-study
cd /data/cs3406/team20/kv-cache-eviction-study
tar xzf ../kv-cache-eviction-study.tar.gz
bash setup_server.sh
export HF_HOME="$PWD/.hf_cache"
```

The existing setup prepares LLaMA and installs dependencies.
LLaMA access may require your existing Hugging Face credentials and model approval.
Never commit access tokens. Reuse an existing compatible environment when available.
Model downloads can consume much of a two-hour slot.
Download only the models planned for that slot.

For Qwen:

```bash
.venv/bin/python scripts/prefetch.py --config configs/server_qwen_3b.yaml
```

For Mistral:

```bash
.venv/bin/python scripts/prefetch.py --config configs/server_mistral_7b.yaml
```

Run the download-free CUDA check first:

```bash
.venv/bin/python scripts/check_models_gpu.py --dtype float16
```

This uses tiny random models. It checks all three model classes, caches, generation, and loss.
It saves `results/gpu_family_check.json` only after all checks pass.
It does not validate downloaded checkpoints or establish quality scores.
Inspect attention fallback counts before interpreting latency or peak memory.

Then run one real checkpoint quickly:

```bash
.venv/bin/python scripts/run_model_study.py --model qwen --mode baseline --quick --minutes 12
.venv/bin/python scripts/run_model_study.py --model qwen --mode eviction --quick --minutes 12
```

Use `--model llama` or `--model mistral` to select another family.
Full experiments omit `--quick`:

```bash
.venv/bin/python scripts/run_model_study.py --model qwen --mode eviction --minutes 70
```

Add `--slot-end` with your confirmed end date, time, and UTC offset.
For example, a slot ending at noon IST uses `--slot-end 2026-10-25T12:00:00+05:30`.
The launcher reserves 15 minutes before that end time.
It rejects past deadlines. Confirm your booking before using the example.
Limits are checked between cases. One case can exceed the remaining time.
Reserve time for reports and copying results.
Do not assume all three full grids fit within one slot.

Optional BF16 requires changing `dtype` to `bfloat16` in a copied configuration.
Choose a distinct `run_name` and output directory for that configuration.
Run it directly with the appropriate runner:

```bash
.venv/bin/python scripts/run_eviction.py --config configs/my_bf16.yaml --quick --minutes 12
.venv/bin/python scripts/check_models_gpu.py --dtype bfloat16 --output results/gpu_family_check_bf16.json
```

## Results and presentation

Baseline results use `results/server_qwen/` or `results/server_mistral/`.
Quick baseline results use the configured name with `_quick` appended.
Eviction results use `results/eviction_<family>/`; quick results append `_quick`.
Each eviction directory contains `measurements.jsonl`, `manifest.json`, and `completion.json`.
Reports contain measurement tables, matched comparisons, needle depth tables, failure cases, and plots.
An attempted grid can contain OOM failures. Check successful coverage before claiming completion.
Rerunning the same configuration resumes recorded cases.
Changed code, hardware, data, or configuration requires a new output directory.
Unexpected error rows remain recorded. Fix the cause and use a new output directory to retry them.

Archive results on the server before the booking ends:

```bash
tar czf /data/cs3406/team20/kv-eviction-results.tar.gz results model_revision.lock
```

Copy them from a separate laptop terminal while the booking remains active:

```bash
scp team20@10.4.25.63:/data/cs3406/team20/kv-eviction-results.tar.gz .
```

Compare each eviction method against its own model's matched full-cache reference.
The original baseline uses monolithic prefill. Eviction runs use matched chunked prefill.
Keep those timing references separate.
Model sizes differ: Mistral is 7B; the selected LLaMA and Qwen checkpoints are 3B.
Tokenizers, training, and model dimensions also differ.
These comparisons do not isolate architecture effects.
Raw perplexities across different tokenizers are not directly comparable.
Use within-model perplexity ratios and quality changes against the paired reference.
Report context, batch, precision, checkpoint revision, GPU, coverage, and attention fallback counts.
Persistent cache reduction alone does not prove lower latency or lower peak memory.

## Validation still required on the GPU

Run the CUDA check and real-checkpoint quick runs first.
Then collect full-grid memory, latency, retrieval, and perplexity measurements.
Review failed cells and kernel fallbacks before selecting results for slides.
Prepare the PPT after those measurements exist. No GPU performance results are claimed by this branch.

Local correctness tests require the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python tests/make_tiny_model.py
PYTHONPATH=. python -m pytest -q
```

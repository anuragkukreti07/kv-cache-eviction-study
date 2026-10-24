"""Stage 1: KV growth curve + latency + memory views (guide Step 3).

For every (batch, context): prefill random tokens, then greedy-decode `decode_tokens`.
Logs theoretical vs measured cache bytes, peak/steady memory, TTFT, TPOT. Runs until out of memory.

Order: trial 0 for every (batch, context) first, then trial 1, then trial 2. If the time budget runs out, every
point already has at least one measurement (and the out-of-memory frontier is known) instead of some points
having three trials and others none.
"""
import argparse
import json

import _boot  # noqa: F401
import torch

from kvstudy import budget
from kvstudy.config import load_config
from kvstudy.deadline import Budget
from kvstudy.env import env_info, result_env
from kvstudy.modelio import (MB, clean, greedy_generate, is_oom, load_model,
                             weight_bytes)
from kvstudy.resultlog import ResultLog

STUDY = "B2_sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    sw = cfg["sweep"]
    bud = Budget(sw.get("max_minutes"), "sweep")
    print(f"[budget] sweep: {bud.describe()}")
    info = env_info()
    model, tok, name, dev = load_model(cfg)
    wbytes = weight_bytes(model)
    elem = next(model.parameters()).element_size()          # 2 bytes for fp16/bf16
    bpt = budget.bytes_per_token(model.config, elem)
    L, H, d = budget.model_dims(model.config)
    attn_desc = ("SDPA, flash/mem-efficient kernels only (GQA-aware)" if cfg.get("attn_patch", True)
                 else "SDPA, PyTorch default kernel choice")
    meta = dict(model=name, model_revision=cfg.get("_revision"), attn_desc=attn_desc, dtype=cfg.get("dtype"),
                model_family=model.config.model_type,
                vram_fraction=cfg.get("vram_fraction"), weights_bytes=wbytes, kv_bytes_per_token=bpt, layers=L,
                kv_heads=H, head_dim=d, q_heads=model.config.num_attention_heads,
                vram_total_bytes=info.get("gpu_vram_bytes", 0), **info)
    with open(f"{cfg['_results_dir']}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] weights {wbytes / 1024**3:.2f} GiB | KV {bpt / 1024:.1f} KiB/token ({L} layers x {H} KV heads x {d})")

    log = ResultLog(f"{cfg['_results_dir']}/baseline_sweep.jsonl", result_env(info))
    seed = cfg.get("seed", 0)
    vocab = model.config.vocab_size
    lo, hi = min(1000, vocab // 10), vocab - min(1000, vocab // 10)   # avoid special tokens at both ends

    print("[warmup] 3 short generations")
    g = torch.Generator().manual_seed(seed)
    for _ in range(3):
        ids = torch.randint(lo, hi, (1, 256), generator=g).to(dev)
        greedy_generate(model, ids, 8, dev)
        del ids
    clean(dev)

    # smallest context that ran out of memory, per batch (also from earlier, resumed runs)
    oom_at = {}
    for r in log.find(study=STUDY, status="oom"):
        oom_at[r["batch"]] = min(oom_at.get(r["batch"], 10 ** 12), r["context"])

    for trial in range(sw["trials"]):
        for batch in sw["batches"]:
            for ctx in sw["contexts"]:
                if ctx >= oom_at.get(batch, 10 ** 12):
                    continue
                if log.has(study=STUDY, batch=batch, context=ctx, trial=trial, status="ok"):
                    continue
                if bud.expired():
                    break
                theory = budget.cache_bytes(model.config, ctx, batch, elem)
                rec = dict(study=STUDY, policy="full", budget=1.0, context=ctx, batch=batch,
                           seed=seed, trial=trial, cache_bytes_theory=theory, weights_bytes=wbytes)
                # input ids depend only on (seed, batch, context, trial): identical across resumes and reorderings
                g = torch.Generator().manual_seed(hash((seed, batch, ctx, trial)) % (2 ** 31))
                failed, ids = False, None
                try:
                    clean(dev)
                    ids = torch.randint(lo, hi, (batch, ctx), generator=g).to(dev)
                    r = greedy_generate(model, ids, sw["decode_tokens"], dev)
                    del ids
                    r.pop("tokens")
                    mid = ctx + sw["decode_tokens"] // 2
                    rec.update(r, status="ok",
                               cache_bytes=r["cache_bytes_measured"],
                               cache_err_pct=100 * (r["cache_bytes_measured"] - theory) / theory,
                               peak_mem_mb=max(r["peak_prefill_bytes"], r["peak_decode_bytes"]) / MB,
                               bytes_per_step=wbytes + budget.cache_bytes(model.config, mid, batch, elem),
                               prefill_tok_s=batch * ctx / (r["ttft_ms"] / 1000),
                               decode_tok_s=(batch * 1000 / r["tpot_ms"]) if r["tpot_ms"] else None)
                    print(f"  trial={trial} batch={batch} ctx={ctx:6d}  cache {rec['cache_bytes'] / MB:9.1f} MiB "
                          f"(err {rec['cache_err_pct']:+.2f}%)  peak {rec['peak_mem_mb']:9.1f} MiB  "
                          f"TTFT {rec['ttft_ms']:9.1f} ms  TPOT {rec['tpot_ms']:.2f} ms", flush=True)
                    log.append(rec)
                except Exception as e:
                    if not is_oom(e):
                        raise
                    failed = True
                if failed:  # handled outside the except block so the traceback no longer pins tensors
                    ids = None
                    clean(dev)
                    print(f"  batch={batch} ctx={ctx:6d}  OUT OF MEMORY (theory cache {theory / MB:.0f} MiB)",
                          flush=True)
                    log.append({**rec, "status": "oom", "trial": 0})
                    oom_at[batch] = ctx
            if bud.stopped:
                break
        if bud.stopped:
            break
    from kvstudy import attention
    meta["attention_stats"] = dict(attention.STATS)
    with open(f"{cfg['_results_dir']}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[attention] {attention.STATS}")
    print("[done] sweep" + (" (stopped early: time budget)" if bud.stopped else ""))


if __name__ == "__main__":
    main()

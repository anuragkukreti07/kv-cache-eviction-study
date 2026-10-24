"""Stage 3: full-cache needle-in-a-haystack accuracy by depth x context (quality reference for T1).

Order: sample 0 for every (context, depth) cell, then sample 1, and so on. If the time budget runs out, every cell
has the same number of prompts (just fewer of them) instead of the long contexts having none.
"""
import argparse

import _boot  # noqa: F401
import torch

from kvstudy.config import load_config
from kvstudy.deadline import Budget
from kvstudy.env import env_info, result_env
from kvstudy.modelio import MB, clean, greedy_generate, is_oom, load_model
from kvstudy.resultlog import ResultLog
from kvstudy.workloads.needle import NeedleBuilder
from kvstudy.workloads.text import filler_text

STUDY = "B4_needle"


def eos_set(tok, model):
    ids = set()
    g = model.generation_config.eos_token_id
    for x in (g if isinstance(g, (list, tuple)) else [g]):
        if x is not None:
            ids.add(int(x))
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    nd = cfg["needle"]
    bud = Budget(nd.get("max_minutes"), "needle")
    print(f"[budget] needle: {bud.describe()}")
    info = env_info()
    model, tok, name, dev = load_model(cfg)
    text, source = filler_text(cfg)
    nb = NeedleBuilder(tok, text, seed=cfg.get("seed", 0))
    print(f"[needle] filler: {source}, {len(nb.filler)} tokens")
    log = ResultLog(f"{cfg['_results_dir']}/baseline_needle.jsonl", result_env(info))
    eos = eos_set(tok, model)
    oom_ctx = min([r["context"] for r in log.find(study=STUDY, status="oom")] or [10 ** 12])

    for s in range(nd["samples"]):
        n_ok = n_run = 0
        for ctx in nd["contexts"]:
            if ctx >= oom_ctx:
                continue
            for depth in nd["depths"]:
                if log.has(study=STUDY, context=ctx, depth=depth, trial=s, status="ok"):
                    continue
                if bud.expired():
                    break
                sample = nb.build(ctx, depth, s)
                rec = dict(study=STUDY, model=name, policy="full", budget=1.0, context=ctx, depth=depth,
                           batch=1, seed=cfg.get("seed", 0), trial=s, kept_answer_tokens=1.0,
                           needle_start=sample["needle_start"], needle_len=sample["needle_len"],
                           filler=source)
                failed, ids = False, None
                try:
                    clean(dev)
                    ids = torch.tensor([sample["ids"]], device=dev)
                    r = greedy_generate(model, ids, nd["max_new_tokens"], dev, eos_ids=eos)
                    ans = tok.decode(r["tokens"][0], skip_special_tokens=True)
                    ok = int(str(sample["code"]) in ans)
                    rec.update(status="ok", correct=ok, answer=ans.strip()[:60], code=sample["code"],
                               ttft_ms=r["ttft_ms"], tpot_ms=r["tpot_ms"],
                               peak_mem_mb=max(r["peak_prefill_bytes"], r["peak_decode_bytes"]) / MB,
                               cache_bytes=r["cache_bytes_measured"])
                    log.append(rec)
                    n_ok += ok
                    n_run += 1
                except Exception as e:
                    if not is_oom(e):
                        raise
                    failed = True
                if failed:
                    ids = None
                    clean(dev)
                    print(f"  ctx={ctx} depth={depth}: out of memory (larger contexts skipped)", flush=True)
                    log.append({**rec, "status": "oom"})
                    oom_ctx = ctx
                    break
            if bud.stopped:
                break
        if n_run:
            print(f"  sample round {s}: {n_ok}/{n_run} correct", flush=True)
        if bud.stopped:
            break
    done = log.find(study=STUDY, status="ok")
    print(f"[done] needle: {len(done)} prompts logged, {sum(r['correct'] for r in done)} correct"
          + (" (stopped early: time budget)" if bud.stopped else ""))


if __name__ == "__main__":
    main()

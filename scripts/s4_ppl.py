"""Stage 4: full-cache perplexity through the chunked harness (guide Step 4: feed text in chunks).

Same loop the evicting caches will use later; with a full cache it equals ordinary perplexity.

`ppl.bos: true` starts every evaluation window with the BOS token (as every real prompt does). This matters for
the eviction study: sink-based policies keep the first tokens, and the BOS token is the model's usual attention
sink. The original laptop run used bos: false, so its numbers are not directly comparable to bos: true runs.
"""
import argparse
import math

import _boot  # noqa: F401
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from kvstudy.config import load_config
from kvstudy.deadline import Budget
from kvstudy.env import env_info, result_env
from kvstudy.modelio import clean, load_model
from kvstudy.families import prefix_token_id
from kvstudy.resultlog import ResultLog
from kvstudy.workloads.text import perplexity_text

STUDY = "B5_ppl"


@torch.no_grad()
def chunked_nll(model, ids, chunk):
    cache, prev, nll, n = DynamicCache(), None, 0.0, 0
    for s in range(0, ids.shape[1], chunk):
        ch = ids[:, s:s + chunk]
        lg = model(input_ids=ch, past_key_values=cache, use_cache=True).logits.float()
        if prev is None:
            preds, tgt = lg[:, :-1], ch[:, 1:]
        else:
            preds, tgt = torch.cat([prev, lg[:, :-1]], 1), ch
        if tgt.numel():
            nll += F.cross_entropy(preds.reshape(-1, preds.shape[-1]), tgt.reshape(-1), reduction="sum").item()
            n += tgt.numel()
        prev = lg[:, -1:]
    return nll, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    pp = cfg["ppl"]
    bud = Budget(pp.get("max_minutes"), "ppl")
    print(f"[budget] ppl: {bud.describe()}")
    info = env_info()
    text, source = perplexity_text(cfg)
    if text is None:
        print("[skip] no wikitext available (needs internet for 'datasets', or data/wikitext2_test.json); "
              "perplexity skipped")
        return
    model, tok, name, dev = load_model(cfg)
    ids_all = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prefix_id = prefix_token_id(tok, model.config) if pp.get("bos", False) else None
    bos = prefix_id is not None
    print(f"[ppl] {source}: {len(ids_all)} tokens; BOS at window start: {bos}")
    log = ResultLog(f"{cfg['_results_dir']}/baseline_ppl.jsonl", result_env(info))
    for L in pp["lengths"]:
        body = L - 1 if bos else L
        nseg = min(pp["segments"], len(ids_all) // body)
        for s in range(nseg):
            if log.has(study=STUDY, context=L, trial=s):
                continue
            if bud.expired():
                break
            seg = ids_all[s * body:(s + 1) * body]
            if bos:
                seg = torch.cat([torch.tensor([prefix_id]), seg])
            nll, n = chunked_nll(model, seg.unsqueeze(0).to(dev), pp["chunk"])
            log.append(dict(study=STUDY, model=name, policy="full", budget=1.0, context=L, trial=s,
                            chunk=pp["chunk"], bos=bos, prefix_token_id=prefix_id,
                            nll=nll, n_tokens=n, ppl=math.exp(nll / n)))
            clean(dev)
        rows = log.find(study=STUDY, context=L)
        if rows:
            tot_nll, tot_n = sum(r["nll"] for r in rows), sum(r["n_tokens"] for r in rows)
            print(f"  length {L:6d}: perplexity {math.exp(tot_nll / tot_n):.3f} over {len(rows)} segments", flush=True)
        if bud.stopped:
            break
    print("[done] ppl" + (" (stopped early: time budget)" if bud.stopped else ""))


if __name__ == "__main__":
    main()

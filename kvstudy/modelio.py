"""Model loading, KV-cache accounting and a plain greedy decode loop (unmodified model)."""
import gc
import inspect
import os
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvstudy import attention
from kvstudy.config import ROOT

MB = 1024 ** 2


def _new_enough_for_dtype_kw():
    parts = transformers.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (4, 56)
    except ValueError:
        return True


def _read_lock(lock):
    pins = {}
    if lock.exists():
        for line in lock.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                pins[parts[0]] = parts[1]
    return pins


def resolve_revision(cfg, name):
    """Pin the exact repo commit: config `revision`, else model_revision.lock, else resolve once and add it to the lock.

    The lock holds one line per model ("<repo> <sha>"), so switching between the 1B and 3B models never
    overwrites the other model's pin. Local model folders are not pinned.
    """
    if cfg.get("revision"):
        return cfg["revision"]
    if os.path.isdir(name):
        return None
    lock = ROOT / "model_revision.lock"
    pins = _read_lock(lock)
    if name in pins:
        return pins[name]
    try:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(name).sha
        pins[name] = rev
        lock.write_text("".join(f"{k} {v}\n" for k, v in pins.items()))
        return rev
    except Exception as e:
        print(f"[model] could not resolve the repo revision ({type(e).__name__}); loading latest, not pinned")
        return None


def load_model(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    dtype = getattr(torch, cfg.get("dtype", "float16"))
    names = [cfg["model"]] + list(cfg.get("fallback_models", []))
    if cfg.get("attn_patch", True):
        attention.install()
    if device.type == "cuda" and cfg.get("vram_fraction"):
        torch.cuda.set_per_process_memory_fraction(float(cfg["vram_fraction"]))
    last = None
    for name in names:
        try:
            rev = resolve_revision(cfg, name)
            cfg["_revision"] = rev
            tok = AutoTokenizer.from_pretrained(name, revision=rev)
            kw = {"attn_implementation": cfg.get("attn", "sdpa")}
            kw["dtype" if _new_enough_for_dtype_kw() else "torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(name, revision=rev, **kw)
            model.to(device).eval()
            print(f"[model] loaded {name} @ {rev} ({next(model.parameters()).dtype}) on {device}")
            return model, tok, name, device
        except Exception as e:  # gated repo, no token, offline ...
            last = e
            print(f"[model] could not load {name}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"No model could be loaded. Last error: {last}")


def weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())  # tied weights counted once


def iter_kv(cache):
    """Yield every cached K and V tensor, across transformers cache API versions."""
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None and t.numel():
                    yield t
    elif hasattr(cache, "key_cache"):
        for t in list(cache.key_cache) + list(cache.value_cache):
            if t is not None and t.numel():
                yield t
    else:
        for k, v in cache:
            yield k
            yield v


def cache_bytes_measured(cache):
    return sum(t.numel() * t.element_size() for t in iter_kv(cache))


def _logits_kw(model):
    params = inspect.signature(model.forward).parameters
    for k in ("logits_to_keep", "num_logits_to_keep"):
        if k in params:
            return {k: 1}
    return {}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def mem_snapshot(dev):
    if dev.type != "cuda":
        return dict(alloc=0, peak=0, reserved=0, used=0)
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=torch.cuda.memory_allocated(), peak=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.memory_reserved(), used=total - free)


def clean(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@torch.no_grad()
def greedy_generate(model, input_ids, max_new, dev, eos_ids=None):
    """Prefill, then one token at a time with a DynamicCache. Greedy, no padding.

    Returns timing (TTFT, per-step decode times) and memory views for the full-cache baseline.
    Memory numbers are relative to what was allocated before the call (weights + inputs).
    """
    kw = _logits_kw(model)
    bs = input_ids.shape[0]
    cache = DynamicCache()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base = mem_snapshot(dev)["alloc"]

    sync(dev)
    t0 = time.perf_counter()
    o = model(input_ids=input_ids, past_key_values=cache, use_cache=True, **kw)
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    sync(dev)
    res = {"ttft_ms": (time.perf_counter() - t0) * 1000}
    del o
    m = mem_snapshot(dev)
    res["peak_prefill_bytes"] = m["peak"] - base
    res["after_prefill_bytes"] = m["alloc"] - base
    res["cache_bytes_measured"] = cache_bytes_measured(cache)
    res["cache_len"] = int(cache.get_seq_length())
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    toks, steps = [nxt], []
    for _ in range(max_new - 1):
        if eos_ids is not None and bs == 1 and int(nxt.item()) in eos_ids:
            break
        sync(dev)
        t = time.perf_counter()
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True, **kw)
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        sync(dev)
        steps.append((time.perf_counter() - t) * 1000)
        toks.append(nxt)
        del o
    m = mem_snapshot(dev)
    res["peak_decode_bytes"] = m["peak"] - base
    res["reserved_bytes"] = m["reserved"]
    res["device_used_bytes"] = m["used"]
    steps_sorted = sorted(steps)
    res["tpot_ms"] = steps_sorted[len(steps_sorted) // 2] if steps else None
    res["tokens"] = torch.cat(toks, 1)
    del cache
    return res

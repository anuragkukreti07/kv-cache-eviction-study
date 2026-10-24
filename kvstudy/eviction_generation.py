"""Matched chunked greedy generation for dense Llama, Qwen2 and Mistral models."""
import time
from statistics import median

import torch
import torch.nn.functional as F

from kvstudy.eviction import make_cache
from kvstudy.families import validate_model_config, validate_sequence_length
from kvstudy.modelio import _logits_kw, mem_snapshot, sync


def _validate(model, ids, dev, chunk):
    if dev.type != "cuda" or ids.device.type != "cuda":
        raise ValueError("experiment entry points require CUDA inputs and a CUDA device")
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("expected nonempty, unpadded [batch, tokens] input_ids")
    if chunk < 1:
        raise ValueError("prefill chunk must be positive")
    validate_model_config(model.config, {}, eviction=True)
    if getattr(model.config, "_attn_implementation", "sdpa") not in {"sdpa", "eager"}:
        raise ValueError("eviction requires sdpa or eager attention")


def _forward(model, ids, cache, **kwargs):
    count = ids.shape[1]
    position_ids = torch.arange(cache.seen_tokens, cache.seen_tokens + count,
                                device=ids.device)[None, :]
    dtype = next(model.parameters()).dtype
    mask = cache.attention_mask(count, ids.device, dtype)
    if cache.spec.policy == "h2o":
        from kvstudy.h2o import scoped_h2o_attention
        with scoped_h2o_attention(model, cache):
            return model(input_ids=ids, position_ids=position_ids, attention_mask=mask,
                         past_key_values=cache, use_cache=True, **kwargs)
    return model(input_ids=ids, position_ids=position_ids, attention_mask=mask,
                 past_key_values=cache, use_cache=True, **kwargs)


def _positions(cache):
    p = cache.retained_positions
    return [] if p is None else p.detach().cpu().tolist()


@torch.no_grad()
def evicting_generate(model, input_ids, max_new, dev, spec, prefill_chunk=256, eos_ids=None):
    """Return CUDA timings plus physical cache accounting; no local GPU required to import."""
    dev = torch.device(dev)
    _validate(model, input_ids, dev, prefill_chunk)
    if max_new < 1:
        raise ValueError("max_new must be positive")
    validate_sequence_length(model.config, input_ids.shape[1], max_new)
    cache = make_cache(spec)
    kw = _logits_kw(model)
    sync(dev)
    base = mem_snapshot(dev)["alloc"]
    torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    for offset in range(0, input_ids.shape[1], prefill_chunk):
        out = _forward(model, input_ids[:, offset:offset + prefill_chunk], cache, **kw)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        del out
    sync(dev)
    ttft_ms = (time.perf_counter() - start) * 1000
    prefill_mem = mem_snapshot(dev)
    prefill_peak_reserved = torch.cuda.max_memory_reserved(dev)
    prefill_cache = cache.summary()
    prefill_positions = _positions(cache)
    torch.cuda.reset_peak_memory_stats(dev)
    tokens, steps = [nxt], []
    for _ in range(max_new - 1):
        if eos_ids is not None and input_ids.shape[0] == 1 and int(nxt.item()) in eos_ids:
            break
        sync(dev)
        step_start = time.perf_counter()
        out = _forward(model, nxt, cache, **kw)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        del out
        sync(dev)
        steps.append((time.perf_counter() - step_start) * 1000)
        tokens.append(nxt)
    # Sample before copying diagnostics or concatenating generated outputs.
    mem = mem_snapshot(dev)
    return dict(tokens=torch.cat(tokens, dim=1), ttft_ms=ttft_ms,
                tpot_ms=median(steps) if steps else None,
                decode_step_ms=steps,
                decode_tok_s=(input_ids.shape[0] * len(steps) * 1000 / sum(steps)) if steps else None,
                generated_tokens=len(tokens) * input_ids.shape[0],
                cache=cache.summary(), prefill_cache=prefill_cache,
                retained_positions=_positions(cache), prefill_retained_positions=prefill_positions,
                peak_allocated_bytes=max(prefill_mem["peak"], mem["peak"]),
                steady_allocated_bytes=mem["alloc"],
                peak_reserved_bytes=max(prefill_peak_reserved, torch.cuda.max_memory_reserved(dev)),
                peak_prefill_bytes=prefill_mem["peak"] - base,
                after_prefill_bytes=prefill_mem["alloc"] - base,
                peak_decode_bytes=mem["peak"] - base,
                cache_bytes_measured=cache.summary()["storage_bytes"],
                cache_len=int(cache.get_seq_length()), reserved_bytes=mem["reserved"],
                device_used_bytes=mem["used"], overhead_ms=None,
                memory_note="Persistent cache is bounded; current chunks and temporary candidates can increase GPU peaks.")


@torch.no_grad()
def eviction_nll(model, ids, chunk, spec):
    """Teacher-forced NLL, scoring each next-token target exactly once.

    Chunk boundaries do not discard targets: the final logit in each chunk
    predicts the first token in the following chunk. The final input token has
    no next-token target and contributes no loss.
    """
    _validate(model, ids, ids.device, chunk)
    if ids.shape[1] < 2:
        raise ValueError("NLL requires at least two input tokens")
    validate_sequence_length(model.config, ids.shape[1])
    cache = make_cache(spec)
    nll = torch.zeros((), device=ids.device, dtype=torch.float64)
    n = 0
    # Feed all but the final token. Every output position has a known target.
    for offset in range(0, ids.shape[1] - 1, chunk):
        end = min(offset + chunk, ids.shape[1] - 1)
        out = _forward(model, ids[:, offset:end], cache)
        targets = ids[:, offset + 1:end + 1]
        loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                               targets.reshape(-1), reduction="sum")
        nll += loss.double()
        n += targets.numel()
        del out, loss
    return dict(nll=float(nll.item()), n=n, cache=cache.summary(),
                retained_positions=_positions(cache))

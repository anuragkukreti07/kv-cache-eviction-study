"""Keep PyTorch SDPA on memory-efficient kernels for grouped-query attention.

Why: with Llama-3.2 (32 query heads, 8 KV heads) Hugging Face calls
scaled_dot_product_attention(..., enable_gqa=True). On builds without a flash kernel for that case
(e.g. Windows) PyTorch silently uses the 'math' kernel, which materialises the full S x S score matrix
for every head. Memory and prefill time then grow with S^2 and dominate the KV cache, which is not the
baseline we want to measure.

Fix: for GQA we (a) fold the query-head groups into the query axis during decode (no KV copy) and
(b) repeat KV heads during prefill, then call SDPA restricted to the flash / memory-efficient kernels.
Numerics are unchanged (checked by selfcheck()). Assumes no padding, which holds for every workload here.
"""
import math
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_ORIG = F.scaled_dot_product_attention
STATS = {"restricted_calls": 0, "fallback_calls": 0}
_warned = False


def _call(q, k, v, attn_mask, dropout_p, is_causal, scale, **kw):
    global _warned
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = _ORIG(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        STATS["restricted_calls"] += 1
        return out
    except RuntimeError as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            raise
        STATS["fallback_calls"] += 1
        if not _warned:
            _warned = True
            warnings.warn("No flash/memory-efficient SDPA kernel for this call; falling back to the default "
                          f"kernel (memory will grow with S^2). Reason: {str(e)[:160]}")
        return _ORIG(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    hq, hk = q.shape[1], k.shape[1]
    if hq == hk:
        return _call(q, k, v, attn_mask, dropout_p, is_causal, scale)
    rep = hq // hk
    if q.shape[2] == 1 and attn_mask is None:
        # decode: query head h uses KV head h // rep, so the `rep` heads sharing a KV head become `rep` queries
        b, _, _, d = q.shape
        out = _call(q.reshape(b, hk, rep, d), k, v, None, dropout_p, False, scale)
        return out.reshape(b, hq, 1, v.shape[-1])
    k = k.repeat_interleave(rep, dim=1)   # same head order as transformers' repeat_kv
    v = v.repeat_interleave(rep, dim=1)
    return _call(q, k, v, attn_mask, dropout_p, is_causal, scale)


def install():
    if getattr(F.scaled_dot_product_attention, "_kv_patched", False):
        return
    patched_sdpa._kv_patched = True
    F.scaled_dot_product_attention = patched_sdpa


@torch.no_grad()
def selfcheck(device="cuda", dtype=torch.float16):
    """Numerical check against a plain fp32 reference + peak-memory check for a 4K prefill."""
    dt = dtype
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g).to(device=device, dtype=dt)

    def ref(q, k, v, causal):
        rep = q.shape[1] // k.shape[1]
        kf, vf = k.repeat_interleave(rep, 1).float(), v.repeat_interleave(rep, 1).float()
        s = (q.float() @ kf.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if causal:
            m = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device).tril(k.shape[2] - q.shape[2])
            s = s.masked_fill(~m, float("-inf"))
        return s.softmax(-1) @ vf

    res = {}
    q, k, v = rnd(2, 32, 1, 64), rnd(2, 8, 300, 64), rnd(2, 8, 300, 64)
    res["decode_max_err"] = float((patched_sdpa(q, k, v, enable_gqa=True).float() - ref(q, k, v, False)).abs().max())
    q, k, v = rnd(1, 32, 512, 64), rnd(1, 8, 512, 64), rnd(1, 8, 512, 64)
    res["prefill_max_err"] = float((patched_sdpa(q, k, v, is_causal=True, enable_gqa=True).float()
                                    - ref(q, k, v, True)).abs().max())
    if device == "cuda":
        q, k, v = rnd(1, 32, 4096, 64), rnd(1, 8, 4096, 64), rnd(1, 8, 4096, 64)
        torch.cuda.synchronize()
        for name, fn in (("patched", lambda: patched_sdpa(q, k, v, is_causal=True, enable_gqa=True)),
                         ("default", lambda: _ORIG(q, k, v, is_causal=True, enable_gqa=True))):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            try:
                out = fn()
                torch.cuda.synchronize()
                res[f"{name}_peak_mib"] = (torch.cuda.max_memory_allocated() - base) / 1024 ** 2
                del out
            except Exception as e:
                res[f"{name}_peak_mib"] = None
                res[f"{name}_error"] = str(e)[:100]
    res["stats"] = dict(STATS)
    return res

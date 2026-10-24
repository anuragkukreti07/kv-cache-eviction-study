"""Stage 0: is the GPU usable with this PyTorch build, is attention efficient, and does the model really work?

  --model-check   also load the configured model and answer one 2K needle prompt (catches download problems,
                  FP16 overflow and wrong outputs before any long run starts; ~30-60 s)
"""
import argparse
import shutil
import subprocess
import sys

import _boot  # noqa: F401
import torch

from kvstudy.config import ROOT, load_config
from kvstudy.env import env_info


def gpu_processes():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True, timeout=10).strip()
        return [l for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def model_check(cfg):
    from kvstudy.modelio import greedy_generate, load_model
    from kvstudy.workloads.needle import NeedleBuilder
    from kvstudy.workloads.text import filler_text
    model, tok, name, dev = load_model(cfg)
    text, source = filler_text(cfg)
    nb = NeedleBuilder(tok, text, seed=12345)
    s = nb.build(2048, 0.5, 0)
    ids = torch.tensor([s["ids"]], device=dev)
    with torch.no_grad():
        lg = model(input_ids=ids[:, :512], logits_to_keep=1).logits
    finite = bool(torch.isfinite(lg).all())
    r = greedy_generate(model, ids, 20, dev)
    ans = tok.decode(r["tokens"][0], skip_special_tokens=True).strip()
    ok = str(s["code"]) in ans
    print(f"  model check      {name}  filler={source}")
    print(f"                   2K needle: expected {s['code']}, model said {ans[:40]!r} -> {'correct' if ok else 'WRONG'}")
    print(f"                   finite logits: {finite}; TTFT {r['ttft_ms']:.0f} ms, TPOT {r['tpot_ms']:.1f} ms")
    if not finite:
        print("[FAIL] the model produces inf/NaN in this dtype. Set `dtype: bfloat16` in the config and rerun.")
        return False
    if not ok:
        print("[warn] the model missed a 2K needle; results will still run, but check the model/tokenizer.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model-check", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)

    info = env_info()
    for k, v in info.items():
        print(f"  {k:16s} {v}")
    ok = True
    if cfg.get("device", "cuda") != "cuda":
        print(f"[warn] device is {cfg.get('device')}: GPU checks skipped (test mode)")
        if args.model_check:
            ok = model_check(cfg)
        sys.exit(0 if ok else 1)
    if not torch.cuda.is_available():
        print("\n[FAIL] CUDA is not available to PyTorch. Re-run the setup script (it installs a CUDA 12.8 build).")
        sys.exit(1)
    arch = torch.cuda.get_arch_list()
    cap = f"sm_{info['gpu_capability'].replace('.', '')}"
    print(f"  compiled archs   {arch}")
    if cap not in arch and not any(a.startswith("compute_") for a in arch):
        print(f"[FAIL] This PyTorch build has no kernels for {cap}. Install a CUDA 12.8+ build.")
        ok = False
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU memory       {free / 1024**3:.1f} GiB free of {total / 1024**3:.1f} GiB")
    procs = gpu_processes()
    if procs:
        print(f"[warn] other processes are using the GPU (memory/latency numbers will be affected): {procs}")
    if free < 0.8 * total:
        print("[warn] less than 80% of GPU memory is free; out-of-memory points will come earlier than they should.")
    du = shutil.disk_usage(ROOT)
    print(f"  disk free        {du.free / 1024**3:.1f} GiB in {ROOT}")
    try:  # real kernel launch (this is where 'no kernel image' errors appear)
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        float((a @ a).float().sum())
        print("[ok] fp16 matmul on GPU works")
    except Exception as e:
        print(f"[FAIL] GPU kernel launch failed: {e}")
        ok = False
    try:
        from kvstudy import attention
        r = attention.selfcheck("cuda")
        print(f"  attention check  {r}")
        if r["decode_max_err"] > 2e-2 or r["prefill_max_err"] > 2e-2:
            print("[FAIL] patched attention disagrees with the reference; do not trust results.")
            ok = False
        elif r.get("patched_peak_mib") is None or r["patched_peak_mib"] > 300 or r["stats"]["fallback_calls"]:
            print("[FAIL] flash/memory-efficient SDPA kernels are not usable on this setup; memory would grow with S^2.")
            ok = False
        else:
            d = r.get("default_peak_mib")
            print(f"[ok] attention: 4K prefill peak {r['patched_peak_mib']:.0f} MiB with the patch"
                  + (f" vs {d:.0f} MiB with PyTorch's default choice" if d else ""))
    except Exception as e:
        print(f"[FAIL] attention check crashed: {type(e).__name__}: {e}")
        ok = False
    if args.model_check and ok:
        try:
            ok = model_check(cfg) and ok
        except Exception as e:
            print(f"[FAIL] model check crashed: {type(e).__name__}: {str(e)[:300]}")
            print("       If this is a download/access error: check internet access and the model name in the config.")
            ok = False
    elif not args.model_check:
        try:
            from transformers import AutoConfig
            from kvstudy.modelio import resolve_revision
            rev = resolve_revision(cfg, cfg["model"])   # pinned commit (works offline once prefetched)
            AutoConfig.from_pretrained(cfg["model"], revision=rev)
            print(f"[ok] model config reachable: {cfg['model']} @ {str(rev)[:10]}")
        except Exception as e:
            print(f"[FAIL] cannot access {cfg['model']}: {type(e).__name__}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

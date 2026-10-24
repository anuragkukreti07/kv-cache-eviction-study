"""Stage 2: achievable GPU memory bandwidth (for the decode-step roofline, guide Step 3.5)."""
import argparse

import _boot  # noqa: F401
import torch

from kvstudy.config import load_config
from kvstudy.env import env_info, result_env
from kvstudy.resultlog import ResultLog


def timed(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best / 1000  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    bw = cfg["bandwidth"]
    if not torch.cuda.is_available() or cfg.get("device", "cuda") != "cuda":
        print("[skip] bandwidth test needs a CUDA GPU")
        return
    info = env_info()
    log = ResultLog(f"{cfg['_results_dir']}/baseline_bandwidth.jsonl", result_env(info))
    n = bw["size_mb"] * 1024 * 1024 // 2
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    nbytes = x.numel() * 2
    t_read = timed(lambda: torch.sum(x, dtype=torch.float32), bw["iters"])
    t_copy = timed(lambda: y.copy_(x), bw["iters"])
    read_gbs, copy_gbs = nbytes / t_read / 1e9, 2 * nbytes / t_copy / 1e9
    print(f"  read-only {read_gbs:7.1f} GB/s   copy (r+w) {copy_gbs:7.1f} GB/s")
    log.append(dict(study="B3_bandwidth", read_gbs=read_gbs, copy_gbs=copy_gbs,
                    size_mb=bw["size_mb"], trial=len(log.find(study="B3_bandwidth"))))


if __name__ == "__main__":
    main()

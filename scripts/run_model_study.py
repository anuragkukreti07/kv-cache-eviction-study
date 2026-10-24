"""Select one family and run within a remote slot, reserving time for copying results."""
import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["llama", "qwen", "mistral"], required=True)
    ap.add_argument("--mode", choices=["baseline", "eviction"], default="eviction")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--minutes", type=float, default=85, help="Work budget; must be positive")
    ap.add_argument("--slot-end", help="Slot end with date and UTC offset, e.g. 2026-10-25T12:00:00+05:30")
    ap.add_argument("--reserve", type=float, default=15, help="Minutes reserved before slot end")
    args = ap.parse_args()
    if args.minutes <= 0 or args.reserve < 0:
        ap.error("--minutes must be positive and --reserve nonnegative")
    deadline = time.time() + args.minutes * 60
    if args.slot_end:
        end = dt.datetime.fromisoformat(args.slot_end)
        if end.tzinfo is None:
            ap.error("--slot-end requires a UTC offset")
        deadline = min(deadline, end.timestamp() - args.reserve * 60)
    remaining = (deadline - time.time()) / 60
    if remaining <= 0:
        ap.error("The work deadline has passed")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the remote CUDA GPU server during your booking.")
    baseline = {"llama": "server_3b.yaml", "qwen": "server_qwen_3b.yaml", "mistral": "server_mistral_7b.yaml"}
    config = baseline[args.model] if args.mode == "baseline" else f"eviction_{args.model}.yaml"
    script = "run_all.py" if args.mode == "baseline" else "run_eviction.py"
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / "configs" / config),
           "--minutes", str(remaining)]
    if args.quick:
        cmd.append("--quick")
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"Running {args.model} {args.mode}; work budget {remaining:.1f} minutes.", flush=True)
    print("Keep SSH open. Save and copy results before your slot ends.", flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()

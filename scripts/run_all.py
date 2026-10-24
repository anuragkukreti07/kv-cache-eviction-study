"""One command for the whole baseline: env check -> sweep -> bandwidth -> needle -> perplexity -> report.

  python scripts/run_all.py --config configs/server_3b.yaml --minutes 80
  python scripts/run_all.py --quick                 smoke test (a few minutes)
  python scripts/run_all.py --skip needle,ppl
  python scripts/run_all.py --fresh                 delete previous results of this run first
  python scripts/run_all.py --deadline 1760467200   stop starting new work at this unix time

Interrupted runs resume where they stopped (finished configurations are skipped). With a time limit, every stage
stops cleanly when the limit is reached and the report is always built from whatever finished.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _boot  # noqa: F401
from kvstudy.config import ROOT, load_config

STAGES = [("env", "s0_check_env.py"), ("sweep", "s1_sweep.py"), ("bandwidth", "s2_bandwidth.py"),
          ("needle", "s3_needle.py"), ("ppl", "s4_ppl.py")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="", help="comma list: env,sweep,bandwidth,needle,ppl")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--minutes", type=float, default=None, help="time budget for this run (minutes)")
    ap.add_argument("--deadline", type=float, default=None, help="absolute stop time (unix seconds)")
    ap.add_argument("--open", action="store_true", help="open the HTML report when done (Windows)")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    cfg = load_config(args.config, args.quick)
    rd = Path(cfg["_results_dir"])
    if args.fresh and rd.exists():
        shutil.rmtree(rd)
        rd.mkdir(parents=True)
    env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "TOKENIZERS_PARALLELISM": "false"}
    deadline = args.deadline
    if args.minutes:
        d = time.time() + 60 * args.minutes
        deadline = min(deadline, d) if deadline else d
    if deadline:
        env["KV_DEADLINE"] = str(deadline)
        print(f"[budget] run must stop starting new work by {time.strftime('%H:%M:%S', time.localtime(deadline))} "
              f"({(deadline - time.time()) / 60:.0f} min from now)")
    extra = ["--config", args.config] + (["--quick"] if args.quick else [])
    print(f"[run] config {args.config} -> results in {rd}")
    status, t_all = [], time.time()
    for name, script in STAGES:
        if name in skip:
            status.append((name, "skipped", 0))
            continue
        if deadline and time.time() >= deadline and name != "env":
            status.append((name, "skipped (time)", 0))
            continue
        print(f"\n===== {name} =====", flush=True)
        t = time.time()
        cmd = [sys.executable, str(ROOT / "scripts" / script)] + extra
        if name == "env" and cfg.get("model_check", True):
            cmd.append("--model-check")
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        status.append((name, "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t))
        if name == "env" and rc != 0:
            print("\nEnvironment check failed - fix the message above and run again.")
            break
    if not any(n == "env" and s.startswith("FAILED") for n, s, _ in status):
        print("\n===== report =====", flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_report.py")] + extra, cwd=ROOT,
                            env=env).returncode
        status.append(("report", "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t))

    print("\n===== summary =====")
    lines = [f"  {n:10s} {st:18s} {dt / 60:5.1f} min" for n, st, dt in status]
    lines.append(f"  total {(time.time() - t_all) / 60:.1f} min")
    print("\n".join(lines))
    (rd / "run_summary.txt").write_text(f"config {args.config}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    rep = rd / "report" / "baseline_report.html"
    if rep.exists():
        print(f"\nReport:  {rep}\nFigures: {rep.parent}")
        if args.open and hasattr(os, "startfile"):
            os.startfile(str(rep))
    sys.exit(0 if all(s[1] in ("ok", "skipped", "skipped (time)") for s in status) else 1)


if __name__ == "__main__":
    main()

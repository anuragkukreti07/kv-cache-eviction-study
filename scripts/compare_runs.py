"""Side-by-side key numbers for every finished baseline run in results/ (e.g. laptop 1B, server 1B, server 3B).

Writes results/comparison.md and results/comparison.csv. Quick (smoke-test) runs are ignored.
"""
import json
import math

import _boot  # noqa: F401
import pandas as pd

from kvstudy.config import ROOT

GIB = 1024 ** 3


def kfmt(n):
    n = int(n)
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def read(p):
    return pd.read_json(p, lines=True) if p.exists() and p.stat().st_size else pd.DataFrame()


def summarize(rd):
    meta = json.loads((rd / "meta.json").read_text())
    row = {"run": rd.name, "GPU": meta.get("gpu"), "model": str(meta.get("model")).split("/")[-1],
           "weights (GiB)": round(meta["weights_bytes"] / GIB, 2),
           "KV per token (KiB)": round(meta["kv_bytes_per_token"] / 1024, 1)}
    sw = read(rd / "baseline_sweep.jsonl")
    if not sw.empty:
        ok, oom = sw[sw.status == "ok"], sw[sw.status == "oom"]
        row["formula error (max %)"] = round(float(ok.cache_err_pct.abs().max()), 4) if len(ok) else None
        agg = ok.groupby(["batch", "context"]).median(numeric_only=True).reset_index()
        for b in sorted(sw.batch.unique()):
            a, o = agg[agg.batch == b], oom[oom.batch == b]
            fit = kfmt(a.context.max()) if len(a) else "-"
            row[f"batch {b}: largest ctx / first OOM"] = f"{fit} / {kfmt(o.context.min()) if len(o) else 'none'}"
        for ctx in (4096, 16384, 32768):
            r = agg[(agg.batch == 1) & (agg.context == ctx)]
            if len(r):
                row[f"b1 {kfmt(ctx)} TTFT (ms)"] = round(float(r.ttft_ms.iloc[0]), 1)
                row[f"b1 {kfmt(ctx)} TPOT (ms)"] = round(float(r.tpot_ms.iloc[0]), 2)
        row["sweep trials (min per point)"] = int(ok.groupby(["batch", "context"]).size().min()) if len(ok) else 0
    bw = read(rd / "baseline_bandwidth.jsonl")
    if not bw.empty:
        row["read bandwidth (GB/s)"] = round(float(bw.read_gbs.max()))
    nd = read(rd / "baseline_needle.jsonl")
    if not nd.empty and "correct" in nd:
        nd = nd[nd.status == "ok"]
        row["needle accuracy"] = f"{nd.correct.mean() * 100:.0f}% of {len(nd)}"
        for c, v in nd.groupby("context").correct.mean().items():
            row[f"needle {kfmt(c)}"] = f"{v * 100:.0f}%"
    pp = read(rd / "baseline_ppl.jsonl")
    if not pp.empty:
        bos = bool(pp["bos"].iloc[0]) if "bos" in pp else False
        for c, d in pp.groupby("context"):
            row[f"ppl {kfmt(c)}{' (BOS)' if bos else ''}"] = round(math.exp(d.nll.sum() / d.n_tokens.sum()), 2)
    return row


def main():
    res = ROOT / "results"
    rows = []
    for rd in sorted(res.iterdir()) if res.exists() else []:
        if rd.is_dir() and (rd / "meta.json").exists() and not rd.name.endswith("_quick"):
            try:
                rows.append(summarize(rd))
            except Exception as e:
                print(f"[compare] skipped {rd.name}: {type(e).__name__}: {e}")
    if not rows:
        print("[compare] no finished runs found")
        return
    df = pd.DataFrame(rows).set_index("run").T
    df.to_csv(res / "comparison.csv")
    md = "| metric | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns) + "\n"
    for k, r in df.iterrows():
        md += f"| {k} | " + " | ".join("" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
                                       for v in r) + " |\n"
    (res / "comparison.md").write_text("# Baseline runs side by side\n\n" + md, encoding="utf-8")
    print(md)
    print(f"[compare] wrote {res / 'comparison.md'}")


if __name__ == "__main__":
    main()

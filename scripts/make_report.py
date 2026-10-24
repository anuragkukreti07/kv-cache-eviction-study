"""Stage 5: turn the JSONL results into PNG figures, CSV tables and a single HTML/Markdown report."""
import argparse
import base64
import html
import json
import math
from pathlib import Path

import _boot  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kvstudy.config import load_config

GIB, MIB = 1024 ** 3, 1024 ** 2
BATCH_COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}   # fixed per batch size
_FALLBACK = ["#e87ba4", "#6250d6", "#008300", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def bcolor(b):
    return BATCH_COLORS.get(int(b), _FALLBACK[int(b) % len(_FALLBACK)])


def kfmt(n):
    n = int(n)
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def read(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_json(p, lines=True)


def ctx_axis(ax, ctxs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctxs)
    ax.set_xticklabels([kfmt(c) for c in ctxs])
    ax.minorticks_off()
    ax.set_xlabel("Context length (tokens)")


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    return name


def theil_sen(x, y):
    """Robust line fit (median of pairwise slopes). Returns (slope ms/GB, intercept ms) or None if not physical."""
    pairs = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if abs(x[j] - x[i]) > 1e-9]
    if len(x) < 3 or not pairs:
        return None
    sl = float(np.median(pairs))
    ic = float(np.median(np.asarray(y) - sl * np.asarray(x)))
    return (sl, ic) if sl > 0 and ic >= 0 else None


def md_table(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def html_table(df, index=True):
    d = df.reset_index() if index else df
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns) + "</tr></thead><tbody>"
    for _, r in d.iterrows():
        h += "<tr>" + "".join(
            f"<td>{'' if (isinstance(v, float) and math.isnan(v)) else html.escape(str(v))}</td>" for v in r) + "</tr>"
    return h + "</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    rd = Path(cfg["_results_dir"])
    out = rd / "report"
    out.mkdir(exist_ok=True)

    meta = json.load(open(rd / "meta.json")) if (rd / "meta.json").exists() else {}
    sweep, bwdf = read(rd / "baseline_sweep.jsonl"), read(rd / "baseline_bandwidth.jsonl")
    needle, ppl = read(rd / "baseline_needle.jsonl"), read(rd / "baseline_ppl.jsonl")

    sections, takeaways, figs = [], [], []   # sections: (title, [("img"|"html"|"md", payload)])
    wb = meta.get("weights_bytes")
    bpt = meta.get("kv_bytes_per_token")
    vram = meta.get("vram_total_bytes") or 0

    # ---------------- setup table ----------------
    if meta:
        setup = pd.DataFrame({"Value": [
            f"{meta.get('gpu')} ({vram / GIB:.2f} GiB VRAM, sm_{str(meta.get('gpu_capability', '')).replace('.', '')})",
            f"{meta.get('model')} (revision {str(meta.get('model_revision'))[:10]})", f"FP16, {meta.get('attn_desc', 'SDPA attention')}, full KV cache (no eviction)",
            f"{wb / GIB:.2f} GiB", f"{bpt / 1024:.1f} KiB  ({meta['layers']} layers x {meta['kv_heads']} KV heads x "
            f"{meta['head_dim']} dim x 2 (K,V) x 2 B)",
            f"torch {meta.get('torch')}, transformers {meta.get('transformers')}, driver {meta.get('driver', '?')}"]},
            index=pd.Index(["GPU", "Model", "Setup", "Weights", "KV cache per token", "Software"], name="Item"))
        sections.append(("Setup", [("html", html_table(setup)), ("md", md_table(setup))]))

    # ---------------- sweep ----------------
    if not sweep.empty and "status" in sweep:
        ok = sweep[sweep.status == "ok"]
        oomr = sweep[sweep.status == "oom"]
        agg = ok.groupby(["batch", "context"]).median(numeric_only=True).reset_index()
        batches = sorted(agg.batch.unique())
        ctxs = sorted(sweep.context.unique())

        # Fig 1: cache growth
        fig, ax = plt.subplots(figsize=(8.4, 5))
        for b in batches:
            a = agg[agg.batch == b]
            th = np.array(ctxs) * bpt * b / GIB
            ax.plot(ctxs, th, color=bcolor(b), lw=1, ls=":", alpha=0.8)
            ax.plot(a.context, a.cache_bytes / GIB, color=bcolor(b), lw=2, marker="o", ms=6,
                    label=f"batch {int(b)}")
            ax.annotate(f"batch {int(b)}", (ctxs[-1], bpt * ctxs[-1] * b / GIB), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
            o = oomr[oomr.batch == b]
            if len(o):
                ax.plot(o.context, o.cache_bytes_theory / GIB, "X", color=bcolor(b), ms=10,
                        markeredgecolor="white", markeredgewidth=1.2, ls="none")
        if wb:
            ax.axhline(wb / GIB, color=INK2, ls="--", lw=1.2)
            ax.text(ctxs[0], wb / GIB, f" weights {wb / GIB:.2f} GiB", va="bottom", fontsize=9, color=INK2)
        if vram:
            ax.axhline(vram / GIB, color=INK, lw=1.4)
            ax.text(ctxs[0], vram / GIB, f" GPU memory {vram / GIB:.1f} GiB", va="bottom", fontsize=9, color=INK)
        ctx_axis(ax, ctxs)
        ax.set_ylabel("KV cache size (GiB)")
        ax.set_title("KV cache growth vs. model weights (dots = measured, dotted = formula, X = out of memory)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
        ax.set_xlim(ctxs[0] * 0.9, ctxs[-1] * 1.45)
        figs.append(("Where the cache becomes the bottleneck", save(fig, out, "fig1_cache_growth.png")))

        # Fig 2: memory views
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for b in batches:
            a = agg[agg.batch == b]
            peak = (wb + a[["peak_prefill_bytes", "peak_decode_bytes"]].max(axis=1)) / GIB
            steady = (wb + a.after_prefill_bytes) / GIB
            axs[0].plot(a.context, peak, color=bcolor(b), lw=2, marker="o", ms=5, label=f"batch {int(b)}")
            axs[1].plot(a.context, steady, color=bcolor(b), lw=2, marker="o", ms=5, label=f"batch {int(b)}")
            o = oomr[oomr.batch == b]
            if len(o) and vram:
                axs[0].plot(o.context, [vram / GIB] * len(o), "X", color=bcolor(b), ms=10,
                            markeredgecolor="white", markeredgewidth=1.2, ls="none")
        for ax, t in zip(axs, ["Peak during the run (prefill transients included)",
                               "Steady state after prefill (weights + cache)"]):
            if vram:
                ax.axhline(vram / GIB, color=INK, lw=1.4)
            ctx_axis(ax, ctxs)
            ax.set_title(t)
        axs[0].set_ylabel("GPU memory allocated (GiB)")
        axs[0].legend(frameon=False, loc="upper left")
        figs.append(("Peak vs. steady-state memory (X = out of memory; black line = GPU memory)",
                     save(fig, out, "fig2_memory_views.png")))

        # Fig 3: latency
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
        for b in batches:
            a = agg[agg.batch == b]
            axs[0].plot(a.context, a.ttft_ms, color=bcolor(b), lw=2, marker="o", ms=5, label=f"batch {int(b)}")
            axs[1].plot(a.context, a.tpot_ms, color=bcolor(b), lw=2, marker="o", ms=5, label=f"batch {int(b)}")
        axs[0].set_yscale("log")
        axs[0].set_ylabel("Time to first token (ms, log)")
        axs[0].set_title("Prefill: time to first token")
        axs[1].set_ylabel("Time per output token (ms)")
        axs[1].set_title("Decode: time per output token")
        axs[1].set_ylim(bottom=0)
        for ax in axs:
            ctx_axis(ax, ctxs)
        axs[0].legend(frameon=False, loc="upper left")
        figs.append(("Latency (median of trials, greedy decoding, warm GPU)", save(fig, out, "fig3_latency.png")))

        # Fig 4: roofline
        bw_gbs = None
        if not bwdf.empty:
            bw_gbs = float(max(bwdf.read_gbs.max(), bwdf.copy_gbs.max()))
        fit = None
        x = agg.bytes_per_step / 1e9
        y = agg.tpot_ms
        fit = theil_sen(x.values, y.values)
        fig, ax = plt.subplots(figsize=(8.4, 5))
        for b in batches:
            a = agg[agg.batch == b]
            ax.scatter(a.bytes_per_step / 1e9, a.tpot_ms, color=bcolor(b), s=46, label=f"batch {int(b)}",
                       edgecolor="white", linewidth=1, zorder=3)
        xs = np.linspace(0, float(x.max()) * 1.05, 50)
        if bw_gbs:
            ax.plot(xs, xs / bw_gbs * 1000, color=INK, lw=1.6, label=f"memory-bound limit ({bw_gbs:.0f} GB/s measured)")
        if fit:
            ax.plot(xs, fit[0] * xs + fit[1], color=INK2, lw=1.4, ls="--",
                    label=f"linear fit: {fit[1]:.1f} ms fixed + {1000 / fit[0]:.0f} GB/s" if fit[0] > 0 else "linear fit")
        ax.set_xlabel("Bytes read per decode step (GB): all weights + whole KV cache")
        ax.set_ylabel("Measured time per output token (ms)")
        ax.set_title("Decode step vs. the memory-bandwidth limit")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.legend(frameon=False, loc="upper left")
        figs.append(("Roofline: how close decoding is to the bandwidth limit", save(fig, out, "fig4_roofline.png")))

        # tables
        st = meta.get("attention_stats", {})
        if st.get("fallback_calls"):
            takeaways.append(f"WARNING: {st['fallback_calls']} attention calls fell back to PyTorch's default kernel; "
                             "memory/latency may be dominated by S^2 score matrices.")
        err = ok.cache_err_pct.abs().max()
        takeaways.append(f"KV cache formula check: measured cache bytes match 2 x L x H x d x S x B x 2 B within "
                         f"{err:.3f}% over {len(ok)} runs (max |error|).")
        rows = []
        for b in batches:
            a = agg[agg.batch == b]
            o = oomr[oomr.batch == b]
            rows.append({"batch": int(b),
                         "cache = weights at (tokens)": f"{wb / (bpt * b):,.0f}",
                         "largest context that ran": kfmt(a.context.max()) if len(a) else "-",
                         "first out-of-memory": kfmt(o.context.min()) if len(o) else "none up to tested max",
                         "cache at largest run (GiB)": f"{a.cache_bytes.iloc[-1] / GIB:.2f}" if len(a) else "-"})
        crossing = pd.DataFrame(rows).set_index("batch")
        if wb:
            b1, bl = batches[0], batches[-1]
            takeaways.append(f"Weights are {wb / GIB:.2f} GiB. The cache is as large as the weights at "
                             f"{wb / (bpt * b1):,.0f} tokens for batch {int(b1)} and {wb / (bpt * bl):,.0f} tokens "
                             f"for batch {int(bl)}; beyond that the cache dominates memory.")
        if len(oomr):
            first = oomr.sort_values(["batch", "context"]).iloc[0]
            tot = (first.cache_bytes_theory + wb) / GIB
            why = ("; weights + cache alone would fit, so the failure comes from prefill transients"
                   if vram and tot < 0.95 * (meta.get("vram_fraction") or 1.0) * vram / GIB else "")
            frac = meta.get("vram_fraction") or 1.0
            cap = f"; usable cap {frac * vram / GIB:.2f} GiB = {frac:.0%} of {vram / GIB:.2f} GiB" if frac < 1 else \
                f" of {vram / GIB:.2f} GiB"
            takeaways.append(f"Out of memory first appears at batch {int(first.batch)} x {kfmt(first.context)} tokens "
                             f"(weights + formula cache = {tot:.2f} GiB{cap}{why}).")
        ref = agg[(agg.batch == batches[0]) & (agg.context == min(4096, agg[agg.batch == batches[0]].context.max()))]
        if len(ref):
            r = ref.iloc[0]
            takeaways.append(f"Batch {int(r.batch)}, {kfmt(r.context)} context: TTFT {r.ttft_ms:.0f} ms, "
                             f"TPOT {r.tpot_ms:.1f} ms ({1000 / r.tpot_ms:.0f} tokens/s).")
        if not fit:
            takeaways.append("Decode-time vs. bytes-read fit is not reported: it was not physically meaningful on this data.")
        if fit and fit[0] > 0:
            takeaways.append(f"Decode time is about {fit[1]:.1f} ms of fixed per-step overhead plus "
                             f"{fit[0]:.2f} ms per GB read (effective {1000 / fit[0]:.0f} GB/s). This is the speed-up "
                             f"ceiling eviction can reach by shrinking the cache (T10).")
        lat = agg.pivot(index="context", columns="batch", values=["ttft_ms", "tpot_ms"]).round(1)
        lat.columns = [f"{'TTFT ms' if a == 'ttft_ms' else 'TPOT ms'} (batch {int(b)})" for a, b in lat.columns]
        lat.index = [kfmt(i) for i in lat.index]
        lat.index.name = "context"
        mem = agg.copy()
        mem["context"] = mem.context.map(kfmt)
        mem = mem.assign(**{"cache GiB": (mem.cache_bytes / GIB).round(3),
                            "formula GiB": (mem.cache_bytes_theory / GIB).round(3),
                            "error %": mem.cache_err_pct.round(3),
                            "peak MiB": mem.peak_mem_mb.round(0),
                            "steady MiB": (mem.after_prefill_bytes / MIB).round(0)})
        mem = mem[["batch", "context", "cache GiB", "formula GiB", "error %", "peak MiB", "steady MiB"]].set_index("batch")
        sections.append(("Cache bottleneck", [("fig", figs[0]), ("fig", figs[1]), ("html", html_table(crossing)),
                                              ("md", md_table(crossing))]))
        sections.append(("Latency and the bandwidth limit", [("fig", figs[2]), ("fig", figs[3]),
                                                             ("html", html_table(lat)), ("md", md_table(lat))]))
        sections.append(("Measured vs. formula (per batch x context)", [("html", html_table(mem)), ("md", md_table(mem))]))
        agg.to_csv(out / "sweep_median.csv", index=False)
        ok.to_csv(out / "sweep_all_trials.csv", index=False)

    # ---------------- bandwidth ----------------
    if not bwdf.empty:
        takeaways.append(f"Achievable GPU memory bandwidth: {bwdf.read_gbs.max():.0f} GB/s (read), "
                         f"{bwdf.copy_gbs.max():.0f} GB/s (copy, read+write).")

    # ---------------- needle ----------------
    if not needle.empty and "status" in needle:
        nd = needle[needle.status == "ok"]
        if len(nd):
            piv = nd.pivot_table(index="depth", columns="context", values="correct", aggfunc="mean")
            cnt = nd.pivot_table(index="depth", columns="context", values="correct", aggfunc="count")
            fig, ax = plt.subplots(figsize=(1.5 + 1.3 * piv.shape[1], 1.3 + 0.75 * piv.shape[0]))
            im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels([kfmt(c) for c in piv.columns])
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([f"{int(d * 100)}%" for d in piv.index])
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    ax.text(j, i, f"{v * 100:.0f}%\n(n={int(cnt.values[i, j])})", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.55 else INK)
            ax.grid(False)
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Needle depth")
            ax.set_title("Full-cache needle-in-a-haystack accuracy")
            fig.colorbar(im, ax=ax, label="accuracy", fraction=0.04)
            figs.append(("Needle accuracy", save(fig, out, "fig5_needle_heatmap.png")))
            by_ctx = nd.groupby("context").correct.mean()
            takeaways.append("Full-cache needle accuracy (mean over depths): " +
                             ", ".join(f"{kfmt(c)} {v * 100:.0f}%" for c, v in by_ctx.items()) +
                             f"; overall {nd.correct.mean() * 100:.0f}% over {len(nd)} prompts. "
                             "This is the ceiling every eviction policy is compared to.")
            piv.to_csv(out / "needle_accuracy.csv")
            sections.append(("Quality reference: needle-in-a-haystack (T1 baseline)",
                             [("fig", figs[-1])]))
    # ---------------- perplexity ----------------
    if not ppl.empty:
        g = pd.Series({c: math.exp(d.nll.sum() / d.n_tokens.sum()) for c, d in ppl.groupby("context")})
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar([kfmt(c) for c in g.index], g.values, color="#2a78d6", width=0.5)
        for b_, v in zip(bars, g.values):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Evaluation window (tokens)")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Full-cache perplexity, wikitext-2 test")
        ax.grid(axis="x", visible=False)
        figs.append(("Perplexity", save(fig, out, "fig6_perplexity.png")))
        bos_note = " (windows start with BOS)" if ("bos" in ppl and bool(ppl["bos"].iloc[0])) else ""
        takeaways.append(f"Full-cache perplexity (wikitext-2, chunked {int(ppl['chunk'].iloc[0])}){bos_note}: " +
                         ", ".join(f"{kfmt(c)} window {v:.2f}" for c, v in g.items()) + ".")
        sections.append(("Quality reference: perplexity (T2/T3 baseline)", [("fig", figs[-1])]))

    # ---------------- write report ----------------
    def img_html(item):
        cap, name = item
        b64 = base64.b64encode((out / name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'

    who = f"{meta.get('model', '?')} on {meta.get('gpu', '?')}" if meta else ""
    body, md = [], ["# Baseline: unmodified FP16 full KV cache\n", f"*{who}*\n" if who else ""]
    if takeaways:
        body.append("<section><h2>Key numbers</h2><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in takeaways) + "</ul></section>")
        md.append("## Key numbers\n" + "\n".join(f"- {t}" for t in takeaways) + "\n")
    for title, items in sections:
        body.append(f"<section><h2>{html.escape(title)}</h2>")
        md.append(f"## {title}\n")
        for kind, payload in items:
            if kind == "fig":
                body.append(img_html(payload))
                md.append(f"![{payload[0]}]({payload[1]})\n")
            elif kind == "html":
                body.append(f'<div class="tbl">{payload}</div>')
            elif kind == "md":
                md.append(payload + "\n")
        body.append("</section>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KV-cache baseline report</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--card:#222221}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,Segoe UI,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:36px 0 10px}}
.sub{{color:var(--ink2);margin:0 0 8px}}
figure{{margin:14px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px}}
figure img{{width:100%;height:auto;display:block}}figcaption{{color:#52514e;font-size:13px;padding:4px 6px}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;font-size:14px;margin:10px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 12px;text-align:left;white-space:nowrap}}th{{color:var(--ink2);font-weight:600}}
li{{margin:4px 0}}
</style></head><body><main>
<h1>Baseline: unmodified FP16 full KV cache</h1>
<p class="sub"><strong>{html.escape(who)}</strong></p>
<p class="sub">When Does KV-Cache Eviction Fail? &middot; Step 3 baseline and bottleneck, plus full-cache quality reference.
Everything here is what eviction policies will be compared against.</p>
{''.join(body)}
</main></body></html>"""
    (out / "baseline_report.html").write_text(page, encoding="utf-8")
    (out / "baseline_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[report] {out / 'baseline_report.html'}")
    print(f"[report] figures + CSVs in {out}")


if __name__ == "__main__":
    main()

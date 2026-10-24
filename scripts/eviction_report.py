"""Report one new eviction run; never pool old runs, models or devices."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def value(row, key):
    if key == "cache_bytes":
        cache = row.get("cache", {})
        return cache.get("total_storage_bytes", cache.get("storage_bytes"))
    if key == "prefill_cache_bytes":
        cache = row.get("prefill_cache", {})
        return cache.get("total_storage_bytes", cache.get("storage_bytes"))
    return row.get(key)


def avg(rows, key):
    values = [value(r, key) for r in rows]
    values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
    return mean(values) if values else None


def fmt(number, digits=3):
    return "—" if number is None else f"{number:.{digits}f}"


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["stage"] != "profile":
            grouped[(row["stage"], row["context"], row.get("batch", 1), row["method"])].append(row)
    result = []
    for (stage, context, batch, method), entries in sorted(grouped.items()):
        good = [r for r in entries if r["status"] == "ok"]
        row = dict(stage=stage, context=context, batch=batch, method=method, recorded=len(entries),
                   successful=len(good), oom=sum(r["status"] == "oom" for r in entries),
                   errors=sum(r["status"] == "error" for r in entries))
        for key in ("cache_bytes", "prefill_cache_bytes", "peak_allocated_bytes", "steady_allocated_bytes",
                    "peak_reserved_bytes", "ttft_ms", "tpot_ms", "decode_tok_s", "answer_retained_fraction"):
            values = [value(r, key) for r in good]
            values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
            row[key] = (median(values) if values else None) if stage == "sweep" else avg(good, key)
        row["accuracy"] = avg(good, "correct")
        row["substring_accuracy"] = avg(good, "contains_code")
        n = sum(r.get("n", 0) for r in good)
        nll = sum(r.get("nll", 0) for r in good)
        row["ppl"] = math.exp(nll / n) if n and nll / n < 709 else None
        row["evaluated_tokens"] = n
        result.append(row)
    return result


def case_key(row):
    return (row["stage"], row["context"], row.get("batch", 1),
            row.get("trial"), row.get("depth"), row.get("sample"), row.get("segment"))


def matched_comparisons(rows):
    """Only successful identical cases against this run's chunked full-precision reference."""
    refs = {case_key(r): r for r in rows if r["method"] in {"full_fp16", "full_bfloat16"} and r["status"] == "ok"}
    grouped = defaultdict(list)
    for row in rows:
        ref = refs.get(case_key(row))
        if row["method"] in {"full_fp16", "full_bfloat16"} or row["status"] != "ok" or ref is None or row["stage"] == "profile":
            continue
        # Quality rows must identify exactly matching token sequences.
        for sha in ("prompt_sha", "window_sha"):
            if sha in row and row[sha] != ref.get(sha):
                raise ValueError(f"Unmatched {sha}")
        grouped[(row["stage"], row["context"], row.get("batch", 1), row["method"])].append((row, ref))
    output = []
    for (stage, context, batch, method), pairs in sorted(grouped.items()):
        row = dict(stage=stage, context=context, batch=batch, method=method, matched_cases=len(pairs))
        for metric in ("cache_bytes", "peak_allocated_bytes", "ttft_ms", "tpot_ms", "decode_tok_s"):
            ratios = [value(a, metric) / value(b, metric) for a, b in pairs
                      if value(a, metric) is not None and value(b, metric) is not None and value(b, metric) > 0]
            row[metric + "_ratio_to_full"] = mean(ratios) if ratios else None
        row["accuracy_delta"] = (mean(int(a["correct"]) - int(b["correct"]) for a, b in pairs)
                                 if stage == "needle" else None)
        if stage == "ppl":
            n_a, n_b = sum(a["n"] for a, b in pairs), sum(b["n"] for a, b in pairs)
            delta = sum(a["nll"] for a, b in pairs) / n_a - sum(b["nll"] for a, b in pairs) / n_b
            row["ppl_ratio_to_full"] = math.exp(delta) if delta < 709 else None
        else:
            row["ppl_ratio_to_full"] = None
        output.append(row)
    return output


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def failure_analysis(rows):
    """Descriptive associations, not proof that token removal caused a failure."""
    refs = {case_key(r): r for r in rows
            if r["stage"] == "needle" and r["method"] in {"full_fp16", "full_bfloat16"} and r["status"] == "ok"}
    output = []
    for row in rows:
        if row["stage"] != "needle" or row["status"] != "ok" or row["method"] in {"full_fp16", "full_bfloat16"}:
            continue
        ref = refs.get(case_key(row))
        if ref is None:
            continue
        if row.get("prompt_sha") != ref.get("prompt_sha"):
            raise ValueError("Failure analysis requires identical prompts")
        retention = row.get("answer_retained_fraction")
        if not ref["correct"]:
            category = "baseline_failed" if not row["correct"] else "recovered"
        elif row["correct"]:
            category = "both_correct"
        elif retention is None:
            category = "new_failure_retention_unknown"
        elif retention < 1:
            category = "new_failure_code_partly_or_fully_evicted"
        else:
            category = "new_failure_code_retained"
        output.append(dict(method=row["method"], context=row["context"], depth=row["depth"],
                           sample=row["sample"], baseline_correct=ref["correct"], correct=row["correct"],
                           code_retained_fraction=retention, category=category,
                           expected=row.get("expected"), answer=row.get("answer")))
    return output


def depth_summary(rows):
    cells = defaultdict(list)
    for row in rows:
        if row["stage"] == "needle":
            cells[(row["context"], row["depth"], row["method"])].append(row)
    output = []
    for (context, depth, method), entries in sorted(cells.items()):
        good = [r for r in entries if r["status"] == "ok"]
        output.append(dict(context=context, depth=depth, method=method, recorded=len(entries),
                           successful=len(good), accuracy=avg(good, "correct"),
                           code_retention=avg(good, "answer_retained_fraction")))
    return output


def charts(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = []
    metrics = {"sweep": [("cache_bytes", "Persistent cache (MiB)", 1024 ** 2),
                          ("ttft_ms", "Time to first token (ms)", 1),
                          ("tpot_ms", "Decode time per token (ms)", 1)],
               "needle": [("accuracy", "Strict answer accuracy", 1),
                           ("answer_retained_fraction", "Code-token retention at prefill, layer 0", 1)],
               "ppl": [("ppl", "Token-weighted perplexity", 1)]}
    for stage, specs in metrics.items():
        batches = sorted({r["batch"] for r in summary if r["stage"] == stage})
        for batch in batches:
            entries = [r for r in summary if r["stage"] == stage and r["batch"] == batch and r["successful"]]
            if not entries:
                continue
            fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 4.5), squeeze=False)
            for ax, (metric, label, divisor) in zip(axes[0], specs):
                for method in sorted({r["method"] for r in entries}):
                    data = sorted([r for r in entries if r["method"] == method and r.get(metric) is not None], key=lambda r: r["context"])
                    if data:
                        ax.plot([r["context"] for r in data], [r[metric] / divisor for r in data], "o-", label=method)
                ax.set_xlabel("Prompt/window tokens")
                ax.set_ylabel(label)
                ax.set_xscale("log", base=2)
                ax.grid(alpha=0.3)
            axes[0, -1].legend(fontsize=7, loc="best")
            fig.suptitle(f"{stage}; batch {batch}; successful measured cases only")
            fig.tight_layout()
            name = f"{stage}_batch{batch}.png"
            fig.savefig(output / name, dpi=160)
            plt.close(fig)
            images.append(name)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing manifest.json and measurements.jsonl")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run / "measurements.jsonl").read_text().splitlines() if line]
    if any(r.get("fingerprint") != manifest["fingerprint"] for r in rows):
        raise ValueError("Rows do not match the run manifest")
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate result keys")
    summary = aggregate(rows)
    comparisons = matched_comparisons(rows)
    output = run / "report"
    output.mkdir(exist_ok=True)
    write_csv(output / "measurements.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "matched_comparisons.csv", comparisons)
    write_csv(output / "needle_by_depth.csv", depth_summary(rows))
    write_csv(output / "needle_failure_cases.csv", failure_analysis(rows))
    failures = [r for r in rows if r["status"] != "ok"]
    write_csv(output / "failures.csv", failures)
    images = [] if args.no_charts else charts(output, summary)
    ident = manifest["identity"]
    completion = json.loads((run / "completion.json").read_text()) if (run / "completion.json").exists() else None
    lines = ["# GPU KV-cache eviction results", "",
             f"Model: `{ident['model']}`. Revision: `{ident['revision']}`.",
             f"Family: `{ident.get('model_family', 'unrecorded')}`. Dtype: `{ident.get('model_dtype', 'unrecorded')}`.",
             f"GPU: {ident['environment']['gpu']}. Mode: `{ident['timing_mode']}`.", "",
             "All methods use identical chunked prefill. This full-precision reference is separate from the original monolithic baseline.",
             "H2O-style uses explicit attention scoring and prunes after each chunk. Recent/sink policies apply retention within each chunk.",
             "H2O-style shares per-layer selections across heads and batch members. It is an adaptation, not an exact paper reproduction.",
             "Kernel choice, scoring, and pruning overhead differ. Compare measured latency without attributing differences solely to token selection.",
             "Cache bytes include retained-position metadata. Allocator peaks measure the whole process, including model weights.",
             "No claim of acceleration follows from persistent storage reduction alone.", "",
             "## Coverage", ""]
    fallback_calls = sum(r.get("attention_stats", {}).get("fallback_calls", 0) for r in rows)
    lines += [f"Recorded attention fallback calls: **{fallback_calls}**.",
              "Fallback kernels can change latency and peak memory. Inspect kernel support before interpreting those measurements.", ""]
    if completion:
        lines += [f"Complete grid attempted: **{completion['complete']}**. Budget reached: **{completion['time_budget_reached']}**.", "",
                  "| Stage | Planned cases | Recorded cases |", "|---|---:|---:|"]
        for stage, total in completion["expected"].items():
            lines.append(f"| {stage} | {total} | {completion['recorded'].get(stage, 0)} |")
    else:
        lines += ["No completion file exists. Treat this run as interrupted and incomplete."]
    lines += ["", f"Recorded failures: {len(failures)}. See failures.csv for exact error messages.", "",
              "Missing cells have no result. OOM and unexpected errors are excluded from successful averages.",
              "Quality means can cover different cases after failures. Use matched_comparisons.csv for paired comparisons only.", "",
              "Timing and memory sweep summaries use medians across successful trials.",
              "needle_by_depth.csv preserves information-position effects. needle_failure_cases.csv pairs each method with the full-cache reference.",
              "Failure categories describe associations with code-token retention. They do not establish causation.", "",
              "## Measurements", "",
              "| Stage | Context | Batch | Method | Success / recorded | Cache MiB | TTFT ms | TPOT ms | Strict accuracy | PPL |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        cache = r["cache_bytes"] / 1024 ** 2 if r["cache_bytes"] is not None else None
        lines.append(f"| {r['stage']} | {r['context']} | {r['batch']} | {r['method']} | {r['successful']} / {r['recorded']} | {fmt(cache)} | {fmt(r['ttft_ms'])} | {fmt(r['tpot_ms'])} | {fmt(r['accuracy'])} | {fmt(r['ppl'])} |")
    lines += ["", "Needle accuracy uses whitespace-normalized exact codes. Substring accuracy is retained as a separate legacy metric.",
              "Code-token retention uses actual token offsets and layer-zero positions after prefill when available.",
              "Every perplexity method uses identical windows, explicit prefix policy, chunk size, and token weighting.",
              "Timing trials use identical seeded random token prompts. Fixed batches do not measure continuous-serving concurrency.",
              "Profiler rows and traces are instrumented runs. They are excluded from timing and quality tables.", ""]
    for name in images:
        lines.extend([f"![{name}]({name})", ""])
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")


if __name__ == "__main__":
    main()

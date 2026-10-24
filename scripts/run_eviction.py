"""Matched CUDA experiments for physical KV eviction.

This harness uses the same chunked prefill for every method. Its full-cache rows
are a new reference, not a relabeling of the original monolithic-prefill data.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_experiment_config(path, quick=False):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if quick:
        for key, value in cfg.get("quick", {}).items():
            cfg[key] = copy.deepcopy(value)
    cfg.pop("quick", None)
    cfg["quick_mode"] = quick
    if cfg.get("device") != "cuda" or cfg.get("dtype") not in {"float16", "bfloat16"}:
        raise ValueError("Eviction experiments require device: cuda and dtype: float16 or bfloat16")
    revision = cfg.get("revision", "")
    if revision and (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("revision must be a pinned 40-character model SHA")
    if cfg.get("fallback_models"):
        raise ValueError("Fallback models would invalidate matched comparisons")
    ratios = cfg["eviction"]["ratios"]
    policies = cfg["eviction"].get("policies", ["recent", "sink_window"])
    if (not policies or len(set(policies)) != len(policies)
            or any(p not in {"recent", "sink_window", "h2o"} for p in policies)):
        raise ValueError("eviction.policies must contain unique recent, sink_window, or h2o entries")
    if not ratios or any(not 0 < r < 1 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("eviction.ratios must contain unique fractions strictly between zero and one")
    sink = cfg["eviction"]["sink_tokens"]
    if not isinstance(sink, int) or sink < 0:
        raise ValueError("sink_tokens must be a nonnegative integer")
    for section, key in (("sweep", "contexts"), ("needle", "contexts"), ("ppl", "lengths")):
        lengths = cfg[section][key]
        if not lengths or any(not isinstance(n, int) or n < 2 for n in lengths):
            raise ValueError(f"{section}.{key} must contain integer lengths >= 2")
        if len(set(lengths)) != len(lengths):
            raise ValueError(f"Duplicate {section}.{key}")
        if any(max(1, int(n * r)) <= sink for n in lengths for r in ratios):
            raise ValueError("Each retention budget must exceed sink_tokens")
    if cfg["eviction"]["prefill_chunk"] < 1 or cfg["ppl"]["chunk"] < 1:
        raise ValueError("Chunk sizes must be positive")
    for section, keys in (("sweep", ("trials", "decode_tokens")),
                          ("needle", ("samples", "max_new_tokens")), ("ppl", ("segments",))):
        if any(not isinstance(cfg[section][key], int) or cfg[section][key] < 1 for key in keys):
            raise ValueError(f"{section} counts must be positive integers")
    batches = cfg["sweep"]["batches"]
    if not batches or len(set(batches)) != len(batches) or any(not isinstance(b, int) or b < 1 for b in batches):
        raise ValueError("sweep.batches must contain unique positive integers")
    depths = cfg["needle"]["depths"]
    if not depths or len(set(depths)) != len(depths) or any(not 0 <= d <= 1 for d in depths):
        raise ValueError("needle.depths must contain unique values in [0, 1]")
    return cfg


def methods(cfg, context):
    label = "full_fp16" if cfg["dtype"] == "float16" else "full_bfloat16"
    base = [dict(method=label, policy="full", ratio=1.0, budget_tokens=None)]
    for ratio in cfg["eviction"]["ratios"]:
        for policy in cfg["eviction"].get("policies", ["recent", "sink_window"]):
            label = "h2o_style" if policy == "h2o" else policy
            base.append(dict(method=f"{label}_r{ratio:g}", policy=policy,
                             ratio=ratio, budget_tokens=max(1, int(context * ratio))))
    return base


def timing_ids(cfg, model_vocab_size, batch, context, trial=0):
    """Identical deterministic CPU token tensors across methods, trials and engines."""
    import torch
    rng = torch.Generator(device="cpu").manual_seed(cfg["seed"] + context * 101 + batch)
    return torch.randint(100, model_vocab_size, (batch, context), generator=rng)


def normalize_answer(text):
    """Strict exact code after whitespace normalization; explanation text fails."""
    return " ".join(text.split())


class DurableLog:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                # A killed writer can leave one incomplete final record. Preserve it separately.
                end = raw.rfind(b"\n") + 1
                tail = raw[end:]
                try:
                    json.loads(tail)
                except (ValueError, UnicodeDecodeError):
                    self.path.with_suffix(".interrupted-tail").write_bytes(tail)
                    with self.path.open("r+b") as f:
                        f.truncate(end)
                else:
                    with self.path.open("ab") as f:
                        f.write(b"\n")
            self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        self.keys = {r["key"] for r in self.rows}
        if len(self.keys) != len(self.rows):
            raise ValueError(f"Duplicate result keys in {path}")
        # An exhausted slot can leave zero measurements; reporting still needs a valid empty log.
        self.path.touch(exist_ok=True)

    def append(self, row):
        if row["key"] in self.keys:
            raise ValueError("Duplicate result key")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.keys.add(row["key"])
        self.rows.append(row)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temp, path)


def bounded_needle(builder, context, depth, sample):
    """Reuse baseline scaffold with bounded code selection and exact code-token offsets."""
    rng = random.Random(f"needle-{builder.seed}-{context}-{depth}-{sample}")
    for _ in range(256):
        code = rng.randint(1000, 9999)
        text = f" The access code for the blue vault is {code}.\n"
        encoded = builder.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        needle, offsets = encoded.input_ids, encoded.offset_mapping
        count = context - len(builder.head) - len(builder.tail) - len(needle)
        if count <= 0 or count > len(builder.filler):
            raise ValueError("Needle context exceeds available scaffold/filler")
        start = rng.randint(0, len(builder.filler) - count)
        filler = builder.filler[start:start + count]
        if str(code) in builder.tok.decode(filler):
            continue
        at = round(depth * count)
        prefix = len(builder.head) + at
        code_start, code_end = text.index(str(code)), text.index(str(code)) + len(str(code))
        span = [prefix + i for i, (a, b) in enumerate(offsets) if a < code_end and b > code_start]
        if not span:
            raise ValueError("Tokenizer did not expose code offsets")
        return dict(ids=builder.head + filler[:at] + needle + filler[at:] + builder.tail,
                    code=str(code), answer_positions=span, needle_start=prefix, needle_len=len(needle))
    raise ValueError("Could not select a unique needle code in 256 attempts")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/eviction_llama.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="+", default=["sweep", "needle", "ppl"])
    ap.add_argument("--minutes", type=float, default=110, help="Total time budget; checked between cases")
    ap.add_argument("--profile", action="store_true", help="Separate instrumented runs, excluded from timing tables")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    stages = list(dict.fromkeys(s for group in args.stages for s in group.split(",")))
    if not stages or any(s not in {"sweep", "needle", "ppl"} for s in stages):
        ap.error("--stages accepts sweep, needle, ppl")
    if args.minutes <= 0:
        ap.error("--minutes must be positive")
    cfg = load_experiment_config(args.config, args.quick)
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Run this script on the remote GPU host.")
    from kvstudy.eviction import EvictionSpec
    from kvstudy.eviction_generation import evicting_generate, eviction_nll
    from kvstudy.modelio import clean, is_oom, load_model
    from kvstudy.families import prefix_token_id
    from kvstudy import attention
    from kvstudy.workloads.needle import NeedleBuilder
    from kvstudy.workloads.text import filler_text, perplexity_text

    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    out = (args.output or ROOT / "results" / (cfg["run_name"] + ("_quick" if args.quick else ""))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model, tok, model_name, dev = load_model(cfg)
    revision = cfg.get("_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("Model must resolve to an exact commit SHA before results are saved")
    if cfg.get("revision") and revision != cfg["revision"]:
        raise RuntimeError("Loaded model revision differs from the configured SHA")
    prefix = prefix_token_id(tok, model.config) if cfg["ppl"].get("bos", True) else None
    prefix_source = ("tokenizer" if tok.bos_token_id is not None else "model_config") if prefix is not None else "none"
    text, source = filler_text(cfg) if "needle" in stages else (None, None)
    ppl_text, ppl_source = perplexity_text(cfg) if "ppl" in stages else (None, None)
    if "ppl" in stages and ppl_text is None:
        raise RuntimeError("Perplexity text unavailable; provide text_file or cache WikiText-2")
    builder = NeedleBuilder(tok, text, cfg["seed"]) if text else None
    all_ids = tok(ppl_text, add_special_tokens=False, return_tensors="pt").input_ids[0] if ppl_text else None
    if all_ids is not None and any(len(all_ids) < n - int(prefix is not None) for n in cfg["ppl"]["lengths"]):
        raise ValueError("Perplexity text is shorter than a requested evaluation window")
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(list((ROOT / "kvstudy").rglob("*.py")) + [Path(__file__)])}
    props = torch.cuda.get_device_properties(dev)
    identity = dict(config=cfg, model=model_name, revision=cfg["_revision"], stages=stages,
                    profile=args.profile, source_hashes=sources,
                    prefix_policy=dict(token_id=prefix, source=prefix_source),
                    model_dtype=str(next(model.parameters()).dtype), model_family=model.config.model_type,
                    environment=dict(torch=torch.__version__, transformers=transformers.__version__,
                                     gpu=props.name, total_memory=props.total_memory,
                                     capability=list(torch.cuda.get_device_capability(dev))),
                    dataset=dict(needle_source=source, needle_sha=hashlib.sha256(text.encode()).hexdigest() if text else None,
                                 ppl_source=ppl_source, ppl_sha=hashlib.sha256(ppl_text.encode()).hexdigest() if ppl_text else None),
                    timing_mode="chunked_prefill_matched", memory_mode="persistent_cache_plus_metadata_and_allocator_peaks")
    fingerprint = digest(identity)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Output contains incompatible configuration/code/model/data/hardware. Choose a new --output directory.")
    else:
        if (out / "measurements.jsonl").exists():
            raise ValueError("Refusing to adopt results without their manifest")
        atomic_json(manifest_path, dict(fingerprint=fingerprint, identity=identity))
    log = DurableLog(out / "measurements.jsonl")
    if any(row.get("fingerprint") != fingerprint for row in log.rows):
        raise ValueError("Saved result rows do not match the manifest fingerprint")
    interrupted = False
    stage_limits = {}
    weights = {"sweep": 0.50, "needle": 0.30, "ppl": 0.20}
    remaining = max(0, deadline - time.monotonic())
    total_weight = sum(weights[s] for s in stages)
    allotments = {s: remaining * (0.95 if args.profile else 1.0) * weights[s] / total_weight for s in stages}
    warm = set()
    expected = {s: 0 for s in stages}
    attempted = {s: 0 for s in stages}
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None

    def spec_for(m):
        return EvictionSpec(policy=m["policy"], budget_tokens=m["budget_tokens"],
                               sink_tokens=cfg["eviction"]["sink_tokens"])

    def perform(stage, coordinates, method, run):
        nonlocal interrupted
        expected[stage] += 1
        key = digest(dict(stage=stage, **coordinates, method=method["method"]))
        if key in log.keys:
            attempted[stage] += 1
            return
        if time.monotonic() >= min(deadline, stage_limits.get(stage, deadline)):
            interrupted = True
            return
        row = dict(key=key, stage=stage, **coordinates, **method,
                   fingerprint=fingerprint, timing_mode="chunked_prefill_matched",
                   attention_backend="explicit_h2o_style" if method["policy"] == "h2o" else cfg.get("attn", "sdpa"))
        attention_before = dict(attention.STATS)
        try:
            clean(dev)
            result = run(spec_for(method))
            row.update(result)
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="oom" if is_oom(exc) else "error", error_type=type(exc).__name__, error=str(exc))
            print(f"[{stage}] {method['method']} {coordinates}: {row['status']}: {exc}", flush=True)
        finally:
            clean(dev)
        row["attention_stats"] = {key: value - attention_before.get(key, 0)
                                  for key, value in attention.STATS.items()}
        log.append(row)
        attempted[stage] += 1
        print(f"[{stage}] {method['method']} {coordinates}: {row['status']}", flush=True)

    def generation(ids, spec, count, stop=None):
        result = evicting_generate(model, ids, count, dev, spec,
                                      prefill_chunk=cfg["eviction"]["prefill_chunk"], eos_ids=stop)
        return result

    if "sweep" in stages:
        stage_limits["sweep"] = min(deadline, time.monotonic() + allotments["sweep"])
        sw = cfg["sweep"]
        # Trial is outermost: partial runs cover the grid before repeating points.
        for trial in range(sw["trials"]):
            for context in sw["contexts"]:
                for batch in sw["batches"]:
                    for method in methods(cfg, context):
                        def sweep(spec, context=context, batch=batch, method=method):
                            ids = timing_ids(cfg, model.config.vocab_size, batch, context).to(dev)
                            warm_key = method["method"]
                            if warm_key not in warm:
                                warm_spec = spec if method["policy"] == "full" else EvictionSpec(
                                    policy=method["policy"], budget_tokens=min(method["budget_tokens"], max(128, cfg["eviction"]["sink_tokens"] + 1)),
                                    sink_tokens=cfg["eviction"]["sink_tokens"])
                                generation(ids[:1, :min(context, 256)], warm_spec, 2)
                                warm.add(warm_key)
                                clean(dev)
                            result = generation(ids, spec, sw["decode_tokens"])
                            result.pop("tokens", None)
                            result.pop("retained_positions", None)
                            result.pop("prefill_retained_positions", None)
                            return result
                        perform("sweep", dict(context=context, batch=batch, trial=trial), method, sweep)
    if "needle" in stages:
        stage_limits["needle"] = min(deadline, time.monotonic() + allotments["needle"])
        ne = cfg["needle"]
        for sample in range(ne["samples"]):
            for context in ne["contexts"]:
                for depth in ne["depths"]:
                    # Build once outside method loop, guaranteeing the exact same prompt.
                    case = bounded_needle(builder, context, depth, sample)
                    prompt_hash = digest(case["ids"])
                    for method in methods(cfg, context):
                        def needle(spec, case=case):
                            ids = torch.tensor([case["ids"]], device=dev)
                            result = generation(ids, spec, ne["max_new_tokens"], eos)
                            answer = tok.decode(result.pop("tokens")[0], skip_special_tokens=True)
                            retained = result.pop("prefill_retained_positions", None)
                            phase = "prefill" if retained is not None else "end_of_generation"
                            if retained is None:
                                retained = result.get("retained_positions", [])
                            result.pop("retained_positions", None)
                            retained = set(retained)
                            result.update(answer=answer, expected=case["code"],
                                          correct=normalize_answer(answer) == case["code"],
                                          contains_code=case["code"] in answer,
                                          answer_positions=case["answer_positions"],
                                          answer_retained_fraction=sum(p in retained for p in case["answer_positions"]) / len(case["answer_positions"]),
                                          retention_phase=phase, retention_layer=0,
                                          prompt_sha=prompt_hash)
                            return result
                        perform("needle", dict(context=context, batch=1, depth=depth, sample=sample), method, needle)
    if "ppl" in stages:
        stage_limits["ppl"] = min(deadline, time.monotonic() + allotments["ppl"])
        pp = cfg["ppl"]
        for context in pp["lengths"]:
            body = context - int(prefix is not None)
            segments = min(pp["segments"], len(all_ids) // body)
            for segment in range(segments):
                window = all_ids[segment * body:(segment + 1) * body]
                if prefix is not None:
                    window = torch.cat([torch.tensor([prefix]), window])
                window_sha = digest(window.tolist())
                for method in methods(cfg, context):
                    def ppl(spec, window=window):
                        result = eviction_nll(model, window.unsqueeze(0).to(dev), pp["chunk"], spec)
                        result.pop("retained_positions", None)
                        if result["n"] != context - 1 or not math.isfinite(result["nll"]):
                            raise ValueError("Invalid loss or unexpected evaluated-token count")
                        ratio = result["nll"] / result["n"]
                        result.update(ppl=math.exp(ratio) if ratio < 709 else None, bos=prefix is not None, prefix_source=prefix_source, prefix_token_id=prefix,
                                      chunk=pp["chunk"], window_sha=window_sha)
                        if result["ppl"] is None:
                            raise ValueError("Perplexity overflow")
                        return result
                    perform("ppl", dict(context=context, batch=1, segment=segment), method, ppl)
    if args.profile:
        trace_dir = out / "traces"
        trace_dir.mkdir(exist_ok=True)
        context = min(cfg["sweep"]["contexts"])
        for method in methods(cfg, context):
            def profile(spec, method=method):
                ids = timing_ids(cfg, model.config.vocab_size, 1, context).to(dev)
                generation(ids, spec, 2)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as prof:
                    generation(ids, spec, min(8, cfg["sweep"]["decode_tokens"]))
                path = trace_dir / f"{method['method']}.json"
                prof.export_chrome_trace(str(path))
                return dict(trace=str(path.relative_to(out)), instrumented=True)
            expected.setdefault("profile", 0)
            attempted.setdefault("profile", 0)
            perform("profile", dict(context=context, batch=1), method, profile)
    errors = sum(r["status"] == "error" for r in log.rows)
    atomic_json(out / "completion.json", dict(expected=expected, recorded=attempted,
                complete=not interrupted and all(expected[s] == attempted[s] for s in expected),
                time_budget_reached=interrupted, unexpected_errors=errors,
                stage_budget_seconds=allotments,
                elapsed_seconds=time.monotonic() - started,
                note="Time limit is checked between cases; one case can exceed the remaining budget. OOM rows are attempted measurements, not successes."))
    print(f"Results: {out}; unexpected errors: {errors}; time budget reached: {interrupted}")
    # Preserve measurements even when reporting fails; unexpected failures remain nonzero exits.
    import subprocess
    report_status = subprocess.run([sys.executable, str(ROOT / "scripts/eviction_report.py"), str(out)], check=False).returncode
    if errors or report_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

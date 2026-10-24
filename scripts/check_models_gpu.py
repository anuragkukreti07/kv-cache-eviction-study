"""Download-free CUDA correctness check for all supported model families."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--output", type=Path, default=ROOT / "results/gpu_family_check.json")
    args = ap.parse_args()
    import torch
    from transformers import (DynamicCache, LlamaConfig, LlamaForCausalLM,
                              Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM)
    from kvstudy import attention
    from kvstudy.eviction import EvictionSpec, make_cache
    from kvstudy.eviction_generation import _forward, evicting_generate, eviction_nll
    from kvstudy.families import resolve_dtype

    dev = torch.device("cuda")
    dtype = resolve_dtype({"dtype": args.dtype}, dev)
    attention.install()
    report = dict(dtype=args.dtype, gpu=torch.cuda.get_device_name(dev), torch=torch.__version__, families={})
    classes = {"llama": (LlamaConfig, LlamaForCausalLM),
               "qwen2": (Qwen2Config, Qwen2ForCausalLM),
               "mistral": (MistralConfig, MistralForCausalLM)}
    tolerance = 0.05 if dtype == torch.bfloat16 else 0.01
    for family, (config_cls, model_cls) in classes.items():
        torch.manual_seed(17)
        config = config_cls(vocab_size=127, hidden_size=128, intermediate_size=192,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=256, sliding_window=None, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = model_cls(config).to(device=dev, dtype=dtype).eval()
        ids = torch.randint(3, 127, (1, 64), device=dev)
        with torch.no_grad():
            expected = model(ids, past_key_values=DynamicCache(), use_cache=True).logits
            cache = make_cache(EvictionSpec())
            actual = torch.cat([_forward(model, ids[:, s:s + 16], cache).logits
                                for s in range(0, 64, 16)], dim=1)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        records = {}
        for policy in ("full", "recent", "sink_window"):
            spec = EvictionSpec(policy=policy, budget_tokens=None if policy == "full" else 8, sink_tokens=2)
            result = evicting_generate(model, ids, 4, dev, spec, prefill_chunk=16)
            loss = eviction_nll(model, ids, 16, spec)
            assert loss["n"] == 63 and math.isfinite(loss["nll"])
            if policy != "full":
                assert result["cache_len"] == 8
                if policy == "sink_window":
                    assert result["retained_positions"][:2] == [0, 1]
            assert result["tokens"].shape == (1, 4)
            records[policy] = dict(cache=result["cache"], nll=loss["nll"], scored_tokens=loss["n"])
        report["families"][family] = dict(status="passed", methods=records)
        print(f"{family}: passed ({args.dtype})", flush=True)
        del model, expected, actual, cache, result, loss
        torch.cuda.empty_cache()
    report["attention_stats"] = dict(attention.STATS)
    report["note"] = "Tiny random models check correctness. Real checkpoint quality and performance require separate experiments."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if attention.STATS["fallback_calls"]:
        print("Attention used fallback kernels. Check kernel support before interpreting GPU timing or peak memory.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

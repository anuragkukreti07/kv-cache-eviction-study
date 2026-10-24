"""Model-family, precision and sequence-limit checks shared by the runners."""
import torch

SUPPORTED_EVICTION_FAMILIES = ("llama", "qwen2", "mistral")


def resolve_dtype(cfg, device):
    """Use the requested precision without silently substituting another dtype."""
    name = cfg.get("dtype", "float16")
    choices = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in choices:
        raise ValueError(f"Unsupported dtype {name!r}; choose float16, bfloat16 or float32")
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if name == "bfloat16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("This CUDA device lacks native BF16 support; explicitly choose dtype: float16")
    return choices[name]


def native_sliding_window_enabled(config):
    """A numeric Qwen window alone does not enable its sliding-window attention."""
    window = getattr(config, "sliding_window", None)
    if not window:
        return False
    flag = getattr(config, "use_sliding_window", None)
    if flag is not None:
        return bool(flag)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return any(kind == "sliding_attention" for kind in layer_types)
    return True


def validate_sequence_length(config, prompt_length, max_new_tokens=0):
    """Bound prompt plus requested generation by the configured positional limit."""
    if prompt_length <= 0 or max_new_tokens < 0:
        raise ValueError("Prompt length must be positive and generation length cannot be negative")
    limit = getattr(config, "max_position_embeddings", None)
    if limit and prompt_length + max_new_tokens > int(limit):
        raise ValueError(f"Prompt {prompt_length} + generation {max_new_tokens} exceeds "
                         f"max_position_embeddings={limit}; reduce context or decode length")


def validate_model_config(config, cfg, eviction=False):
    """Fail before workloads start when their model settings are unsupported."""
    family = getattr(config, "model_type", "unknown")
    expected_family = cfg.get("model_family")
    if expected_family and family != expected_family:
        raise ValueError(f"Expected model family {expected_family!r}; checkpoint declares {family!r}")
    if eviction and family not in SUPPORTED_EVICTION_FAMILIES:
        raise ValueError(f"Eviction supports {SUPPORTED_EVICTION_FAMILIES}; received {family!r}")
    if eviction and native_sliding_window_enabled(config):
        raise ValueError("Eviction requires full native attention. This checkpoint enables sliding-window "
                         "attention; choose a full-attention checkpoint such as Mistral-7B-Instruct-v0.3")
    for section, lengths, decode_key in (("sweep", "contexts", "decode_tokens"),
                                         ("needle", "contexts", "max_new_tokens"),
                                         ("ppl", "lengths", None)):
        settings = cfg.get(section, {})
        for length in settings.get(lengths, []):
            validate_sequence_length(config, int(length), int(settings.get(decode_key, 0)) if decode_key else 0)
    return family


def prefix_token_id(tokenizer, config):
    """Explicit PPL prefix: tokenizer BOS, then checkpoint BOS, otherwise no prefix.

    Qwen chat prompts remain controlled by its chat template. This helper only
    selects the separately configured PPL prefix; EOS is never substituted.
    """
    token = getattr(tokenizer, "bos_token_id", None)
    if token is None:
        token = getattr(config, "bos_token_id", None)
    return int(token) if token is not None else None


def validate_run_identity(cfg, model_name, revision):
    """Protect new family baseline runs from mixing checkpoint or precision rows."""
    if not cfg.get("model_family") or cfg.get("eviction") or not cfg.get("_results_dir"):
        return
    import json
    from pathlib import Path
    root = Path(cfg["_results_dir"])
    expected = {"model": model_name, "dtype": cfg.get("dtype", "float16"), "model_revision": revision}
    identity = root / "baseline_identity.json"
    for candidate in (identity, root / "meta.json"):
        if not candidate.exists():
            continue
        saved = json.loads(candidate.read_text())
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"Baseline result identity mismatch for {key}: {saved.get(key)!r} != {value!r}; "
                                 "choose a fresh run_name before changing model, revision or dtype")
    if not revision:
        raise ValueError("New model-family baseline runs require a resolved checkpoint revision")
    root.mkdir(parents=True, exist_ok=True)
    temporary = identity.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, indent=2))
    temporary.replace(identity)

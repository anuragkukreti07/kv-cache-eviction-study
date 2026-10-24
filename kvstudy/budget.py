"""Cache-bytes calculator: 2 x layers x KV heads x head_dim x tokens x batch x bytes."""


def model_dims(cfg):
    L = cfg.num_hidden_layers
    H = cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return L, H, d


def bytes_per_token(cfg, p=2):
    L, H, d = model_dims(cfg)
    return 2 * L * H * d * p


def cache_bytes(cfg, tokens, batch=1, p=2):
    return bytes_per_token(cfg, p) * tokens * batch


def crossover_tokens(weights_bytes, bpt, batch=1):
    """Context length at which the cache is as large as the weights."""
    return weights_bytes / (bpt * batch)

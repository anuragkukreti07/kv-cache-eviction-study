"""Attention-based H2O-style retention, adapted for chunked multi-family runs.

Each layer accumulates normalized causal attention mass. Scores average batch
members and query heads, then sum over query positions. All heads in a layer
share retained positions. Half the budget protects recent tokens; old slots use
accumulated scores. This is an empirical adaptation, not a paper reproduction.
Pruning happens after each prefill chunk and after each decoding token.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from transformers import AttentionInterface
from transformers.cache_utils import DynamicLayer

BACKEND = "kvstudy_h2o_style"
_ACTIVE_CACHE = ContextVar("kvstudy_h2o_cache", default=None)


def select_heavy_recent(positions, scores, budget):
    """Return chronologically ordered indices; equal old scores favor earlier positions."""
    size = positions.numel()
    if budget is None or size <= budget:
        return torch.arange(size, device=positions.device)
    recent_count = max(1, budget // 2)
    old_end = size - recent_count
    heavy_count = budget - recent_count
    # Positions are always stored chronologically; stable sorting makes ties explicit.
    heavy = torch.argsort(scores[:old_end], descending=True, stable=True)[:heavy_count]
    recent = torch.arange(old_end, size, device=positions.device)
    return torch.cat((heavy, recent)).sort().values


class H2OLayer(DynamicLayer):
    """Keep pending candidates until their causal attention probabilities are scored."""
    is_croppable = False

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.positions = None
        self.scores = None
        self.seen_tokens = 0
        self.score_updates = 0
        self.scored_queries = 0
        self.pending_queries = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("K/V must have equal [batch, heads, tokens, dim] shapes")
        count = key_states.shape[-2]
        if count < 1:
            raise ValueError("empty KV updates are unsupported")
        if self.pending_queries:
            raise RuntimeError("H2O candidates must be scored before the next cache update; use _forward")
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.is_initialized = True
        new_positions = torch.arange(self.seen_tokens, self.seen_tokens + count, device=key_states.device)
        new_scores = torch.zeros(count, dtype=torch.float32, device=key_states.device)
        if self.keys is None:
            self.keys, self.values = key_states, value_states
            self.positions, self.scores = new_positions, new_scores
        else:
            self.keys = torch.cat((self.keys, key_states), dim=-2)
            self.values = torch.cat((self.values, value_states), dim=-2)
            self.positions = torch.cat((self.positions, new_positions))
            self.scores = torch.cat((self.scores, new_scores))
        self.seen_tokens += count
        self.pending_queries = count
        return self.keys, self.values

    def score_and_prune(self, mass):
        """Accumulate a K-vector of head/batch-mean probability mass, then prune."""
        if not self.pending_queries:
            raise RuntimeError("H2O has no pending attention candidates")
        if mass.ndim != 1 or mass.numel() != self.positions.numel():
            raise ValueError("H2O attention mass must match the candidate token count")
        self.scores = self.scores + mass.detach().to(device=self.scores.device, dtype=torch.float32)
        keep = select_heavy_recent(self.positions, self.scores, self.spec.budget_tokens)
        self.keys = self.keys.index_select(-2, keep)
        self.values = self.values.index_select(-2, keep)
        self.positions = self.positions.index_select(0, keep)
        self.scores = self.scores.index_select(0, keep)
        self.score_updates += 1
        self.scored_queries += self.pending_queries
        self.pending_queries = 0

    def reset(self):
        self.keys = self.values = self.positions = self.scores = None
        self.seen_tokens = self.score_updates = self.scored_queries = self.pending_queries = 0
        self.is_initialized = False


def h2o_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Compute causal probabilities using this layer's absolute cache positions.

    Model-created masks may describe layer zero, whose retained positions differ.
    These experiments have no padding; use each layer's positions instead.
    Head blocks bound temporary score tensors to [B, 4, chunk, budget+chunk].
    """
    cache = _ACTIVE_CACHE.get()
    if cache is None:
        raise RuntimeError("H2O attention requires the scoped _forward cache")
    if dropout or module.training:
        raise ValueError("H2O evaluation requires model.eval() and zero attention dropout")
    layer = cache.layers[module.layer_idx]
    b, heads, queries, dim = query.shape
    if queries != layer.pending_queries or key.shape[-2] != layer.positions.numel():
        raise ValueError("H2O query/candidate shapes do not match the layer cache")
    if heads % key.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    group = heads // key.shape[1]
    query_positions = layer.positions[-queries:]
    allowed = layer.positions[None, :] <= query_positions[:, None]
    scale = dim ** -0.5 if scaling is None else scaling
    mass = torch.zeros(key.shape[-2], device=query.device, dtype=torch.float32)
    blocks = []
    for start in range(0, heads, 4):
        end = min(heads, start + 4)
        indices = torch.arange(start, end, device=query.device) // group
        kb = key.index_select(1, indices)
        vb = value.index_select(1, indices)
        logits = torch.matmul(query[:, start:end].float(), kb.float().transpose(-1, -2)) * scale
        logits.masked_fill_(~allowed[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        mass += probabilities.detach().sum(dim=(0, 1, 2)) / (b * heads)
        # Attention output follows the model precision; scores remain float32.
        blocks.append(torch.matmul(probabilities.to(query.dtype), vb))
    output = torch.cat(blocks, dim=1).transpose(1, 2).contiguous()
    layer.score_and_prune(mass)
    return output, None


@contextmanager
def scoped_h2o_attention(model, cache):
    """Switch only this forward to the registered backend and restore settings on errors."""
    AttentionInterface.register(BACKEND, h2o_attention)
    settings = {}
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and id(config) not in settings:
            settings[id(config)] = (config, getattr(config, "_attn_implementation", None))
    original = model.config._attn_implementation
    token = _ACTIVE_CACHE.set(cache)
    try:
        model.set_attn_implementation(BACKEND)
        if model.config._attn_implementation != BACKEND:
            raise RuntimeError("This model cannot switch to the H2O attention backend")
        yield
    finally:
        try:
            model.set_attn_implementation(original)
        finally:
            # Restore exact nested settings even if the setter raises or partially changes them.
            for config, implementation in settings.values():
                config._attn_implementation_internal = implementation
            _ACTIVE_CACHE.reset(token)

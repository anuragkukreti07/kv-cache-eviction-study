"""Full-precision KV caches with physical eviction policies."""
from dataclasses import dataclass
from functools import partial

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, DynamicLayer


@dataclass(frozen=True)
class EvictionSpec:
    policy: str = "full"
    budget_tokens: int | None = None
    sink_tokens: int = 4

    def __post_init__(self):
        if self.policy not in {"full", "recent", "sink_window", "h2o"}:
            raise ValueError("policy must be full, recent, sink_window, or h2o")
        if self.budget_tokens is not None and self.budget_tokens < 1:
            raise ValueError("budget_tokens must be positive or None")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be nonnegative")
        if (self.policy == "sink_window" and self.budget_tokens is not None
                and self.sink_tokens >= self.budget_tokens):
            raise ValueError("sink_tokens must leave at least one recent-token slot")
        if self.policy == "full" and self.budget_tokens is not None:
            raise ValueError("full policy requires budget_tokens=None")


def _keep_indices(positions, spec):
    budget = spec.budget_tokens
    if spec.policy == "full" or budget is None or positions.numel() <= budget:
        return torch.arange(positions.numel(), device=positions.device)
    if spec.policy == "recent":
        return torch.arange(positions.numel() - budget, positions.numel(), device=positions.device)
    sink = positions < spec.sink_tokens
    recent_count = budget - int(sink.sum())
    recent = torch.arange(positions.numel(), device=positions.device) >= positions.numel() - recent_count
    return torch.where(sink | recent)[0]


class EvictionLayer(DynamicLayer):
    is_croppable = False

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.positions = None
        self.seen_tokens = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("K/V must have equal [batch, heads, tokens, dim] shapes")
        count = key_states.shape[-2]
        if count == 0:
            raise ValueError("empty KV updates are unsupported")
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.is_initialized = True
        start = self.seen_tokens
        new_positions = torch.arange(start, start + count, device=key_states.device)
        if self.keys is None:
            keys, values, positions = key_states, value_states, new_positions
        else:
            old_k, old_v = self.keys, self.values
            keys = torch.cat((old_k, key_states), dim=-2)
            values = torch.cat((old_v, value_states), dim=-2)
            positions = torch.cat((self.positions, new_positions))
        keep = _keep_indices(positions, self.spec)
        self.positions = positions.index_select(0, keep)
        # index_select owns its allocation; a view could retain evicted backing storage.
        self.keys, self.values = keys.index_select(-2, keep), values.index_select(-2, keep)
        self.seen_tokens += count
        # Current queries see the complete candidate list, constrained by the
        # caller's absolute-position causal/retention mask. Stored state is bounded.
        return keys, values

    def reset(self):
        self.keys = self.values = self.positions = None
        self.seen_tokens = 0
        self.is_initialized = False


class EvictionDynamicCache(DynamicCache):
    def __init__(self, spec):
        # Skip DynamicCache's config-derived layer selection deliberately.
        if spec.policy == "h2o":
            from kvstudy.h2o import H2OLayer
            layer_class = H2OLayer
        else:
            layer_class = EvictionLayer
        Cache.__init__(self, layer_class_to_replicate=partial(layer_class, spec))
        self.spec = spec

    @property
    def retained_positions(self):
        return self.layers[0].positions if self.layers else None

    @property
    def seen_tokens(self):
        return self.layers[0].seen_tokens if self.layers else 0

    def attention_mask(self, count, device, dtype):
        """Explicit 4D additive mask; columns match update()'s returned K/V."""
        if self.spec.policy == "h2o":
            # The custom backend creates causal masks from each layer's own positions.
            return None
        query = torch.arange(self.seen_tokens, self.seen_tokens + count, device=device)
        old = self.retained_positions
        candidates = query if old is None else torch.cat((old, query))
        allowed = candidates[None, :] <= query[:, None]
        if self.spec.policy != "full" and self.spec.budget_tokens is not None:
            sinks = self.spec.sink_tokens if self.spec.policy == "sink_window" else 0
            # Before all sink positions exist, their unused slots remain available.
            sink_count = torch.minimum(query + 1, torch.full_like(query, sinks))
            window = self.spec.budget_tokens - sink_count
            recent = candidates[None, :] > query[:, None] - window[:, None]
            allowed &= (candidates[None, :] < sinks) | recent
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask[None, None, :, :]

    def summary(self):
        kv_bytes = position_bytes = score_bytes = 0
        for layer in self.layers:
            for tensor in (layer.keys, layer.values):
                if tensor is not None:
                    kv_bytes += tensor.untyped_storage().nbytes()
            if layer.positions is not None:
                position_bytes += layer.positions.untyped_storage().nbytes()
            scores = getattr(layer, "scores", None)
            if scores is not None:
                score_bytes += scores.untyped_storage().nbytes()
        result = dict(storage_bytes=kv_bytes, kv_bytes=kv_bytes, metadata_bytes=position_bytes + score_bytes,
                    total_storage_bytes=kv_bytes + position_bytes + score_bytes,
                    retained_tokens=int(self.get_seq_length()), seen_tokens=self.seen_tokens,
                    policy=self.spec.policy, budget_tokens=self.spec.budget_tokens,
                    sink_tokens=self.spec.sink_tokens)
        if self.spec.policy == "h2o":
            result.update(algorithm_label="h2o_style", score_bytes=score_bytes,
                          recent_tokens=None if self.spec.budget_tokens is None else max(1, self.spec.budget_tokens // 2),
                          score_updates=sum(layer.score_updates for layer in self.layers),
                          scored_queries=sum(layer.scored_queries for layer in self.layers),
                          selection_granularity="after_each_prefill_chunk_and_decode_token",
                          aggregation="per_layer_mean_batch_and_query_heads",
                          position_diagnostics="layer_zero_only")
        return result



def make_cache(spec):
    return EvictionDynamicCache(spec)

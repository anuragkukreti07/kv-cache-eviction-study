"""Download-free H2O-style attention and physical-cache correctness checks."""
from types import SimpleNamespace

import pytest
import torch
from transformers import (DynamicCache, LlamaConfig, LlamaForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM)

from kvstudy.eviction import EvictionSpec, make_cache
from kvstudy.eviction_generation import _forward
from kvstudy.h2o import BACKEND, h2o_attention, scoped_h2o_attention, select_heavy_recent


def tiny_model(family="llama", attention="eager"):
    classes = {"llama": (LlamaConfig, LlamaForCausalLM),
               "qwen2": (Qwen2Config, Qwen2ForCausalLM),
               "mistral": (MistralConfig, MistralForCausalLM)}
    config_cls, model_cls = classes[family]
    torch.manual_seed(17)
    config = config_cls(vocab_size=41, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, attention_dropout=0.0,
                        sliding_window=None, use_sliding_window=False)
    config._attn_implementation = attention
    return model_cls(config).eval()


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_unrestricted_h2o_matches_native_and_restores(family, attention):
    model = tiny_model(family, attention)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])
    with torch.no_grad():
        expected = model(ids, past_key_values=DynamicCache(), use_cache=True).logits
        cache = make_cache(EvictionSpec(policy="h2o"))
        actual = torch.cat([_forward(model, ids[:, s:s + 3], cache).logits
                            for s in range(0, ids.shape[1], 3)], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert model.config._attn_implementation == attention
    assert cache.retained_positions.tolist() == list(range(7))
    assert all(layer.score_updates == 3 and layer.scored_queries == 7 for layer in cache.layers)


def test_causal_probabilities_exclude_future_and_accumulate_normalized_mass():
    model = tiny_model()
    cache = make_cache(EvictionSpec(policy="h2o"))
    key = torch.ones(1, 1, 4, 1)
    values = torch.tensor([[[[2.], [4.], [8.], [16.]]]])
    k, v = cache.update(key, values, 0)
    query = torch.ones(1, 1, 4, 1)
    module = SimpleNamespace(layer_idx=0, training=False)
    with scoped_h2o_attention(model, cache):
        output, weights = h2o_attention(module, query, k, v, None, scaling=1.)
    expected = torch.tensor([2., 3., 14/3, 7.5])
    torch.testing.assert_close(output.flatten(), expected)
    torch.testing.assert_close(cache.layers[0].scores,
                               torch.tensor([1+1/2+1/3+1/4, 1/2+1/3+1/4, 1/3+1/4, 1/4]))
    assert weights is None
    assert cache.layers[0].scores.sum().item() == pytest.approx(4.)
    # A later query adds mass to existing scores rather than replacing history.
    k, v = cache.update(key[..., :1, :], values[..., :1, :], 0)
    before = cache.layers[0].scores.clone()
    with scoped_h2o_attention(model, cache):
        h2o_attention(module, query[..., :1, :], k, v, None, scaling=1.)
    torch.testing.assert_close(cache.layers[0].scores, before + .2)


def test_old_heavy_hitters_plus_recent_and_stable_ties():
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=4))
    x = torch.ones(1, 1, 6, 2)
    cache.update(x, x, 0)
    cache.layers[0].score_and_prune(torch.tensor([9., 1., 8., 0., 0., 0.]))
    assert cache.retained_positions.tolist() == [0, 2, 4, 5]
    positions = torch.arange(8)
    assert select_heavy_recent(positions, torch.zeros(8), 4).tolist() == [0, 1, 6, 7]
    assert select_heavy_recent(positions, torch.zeros(8), 1).tolist() == [7]


def test_head_blocks_and_batch_aggregation_match_dense_attention():
    model = tiny_model()
    cache = make_cache(EvictionSpec(policy="h2o"))
    torch.manual_seed(31)
    query = torch.randn(2, 8, 3, 4)
    key = torch.randn(2, 2, 3, 4)
    value = torch.randn_like(key)
    cache.update(key, value, 0)
    expanded_key = key.repeat_interleave(4, dim=1)
    expanded_value = value.repeat_interleave(4, dim=1)
    logits = query @ expanded_key.transpose(-1, -2) * .5
    logits.masked_fill_(~torch.ones(3, 3, dtype=torch.bool).tril(), float("-inf"))
    probabilities = logits.softmax(-1)
    expected = (probabilities @ expanded_value).transpose(1, 2)
    with scoped_h2o_attention(model, cache):
        output, _ = h2o_attention(SimpleNamespace(layer_idx=0, training=False), query, key, value, None, scaling=.5)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(cache.layers[0].scores, probabilities.sum(dim=(0, 1, 2)) / 16)


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_h2o_forward(family, dtype):
    model = tiny_model(family).to(dtype=dtype)
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=4))
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    with torch.no_grad():
        output = _forward(model, ids, cache)
        decoded = _forward(model, ids[:, :1], cache)
    assert torch.isfinite(output.logits).all() and torch.isfinite(decoded.logits).all()
    assert cache.get_seq_length() == 4
    assert all(layer.keys.dtype == dtype and layer.scores.dtype == torch.float32 for layer in cache.layers)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_h2o_physical_bytes_scores_and_reset(dtype):
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=4))
    x = torch.ones(2, 3, 10, 8, dtype=dtype)
    cache.update(x, x, 0)
    cache.layers[0].score_and_prune(torch.arange(10).float())
    layer = cache.layers[0]
    assert layer.keys.dtype == layer.values.dtype == dtype
    assert layer.scores.dtype == torch.float32
    assert layer.keys.shape[-2] == 4
    assert layer.keys.untyped_storage().nbytes() == layer.keys.numel() * x.element_size()
    summary = cache.summary()
    assert summary["kv_bytes"] == 2 * 2 * 3 * 4 * 8 * x.element_size()
    assert summary["score_bytes"] == 4 * 4
    assert summary["metadata_bytes"] == 4 * (8 + 4)
    assert summary["total_storage_bytes"] == summary["kv_bytes"] + summary["metadata_bytes"]
    assert summary["algorithm_label"] == "h2o_style"
    cache.reset()
    assert cache.summary()["total_storage_bytes"] == 0
    assert layer.score_updates == layer.scored_queries == layer.pending_queries == 0
    assert cache.seen_tokens == 0


def test_selections_are_per_layer():
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=3))
    x = torch.ones(1, 1, 6, 2)
    for index, mass in enumerate(([9., 8., 0., 0., 0., 0.], [0., 0., 9., 8., 0., 0.])):
        cache.update(x, x, index)
        cache.layers[index].score_and_prune(torch.tensor(mass))
    assert cache.layers[0].positions.tolist() == [0, 1, 5]
    assert cache.layers[1].positions.tolist() == [2, 3, 5]


def test_scope_restores_on_error_and_no_context_leaks():
    model = tiny_model(attention="sdpa")
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=3))
    with pytest.raises(RuntimeError, match="deliberate"):
        with scoped_h2o_attention(model, cache):
            assert model.config._attn_implementation == BACKEND
            raise RuntimeError("deliberate")
    assert model.config._attn_implementation == "sdpa"
    with pytest.raises(RuntimeError, match="scoped"):
        h2o_attention(SimpleNamespace(layer_idx=0, training=False), None, None, None, None)


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
def test_h2o_budget_after_every_chunk_and_decode(family):
    model = tiny_model(family)
    cache = make_cache(EvictionSpec(policy="h2o", budget_tokens=4))
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17]])
    with torch.no_grad():
        for offset in range(0, ids.shape[1], 3):
            output = _forward(model, ids[:, offset:offset + 3], cache)
            assert torch.isfinite(output.logits).all()
            for layer in cache.layers:
                assert layer.get_seq_length() <= 4
                assert layer.pending_queries == 0
                assert layer.positions[-2:].tolist() == list(range(layer.seen_tokens-2, layer.seen_tokens))
        _forward(model, ids[:, :1], cache)
    assert cache.get_seq_length() == 4
    assert cache.seen_tokens == 10
    assert all(layer.scored_queries == 10 for layer in cache.layers)

"""Small CPU correctness checks; remote experiment entry points require CUDA."""
import pytest
import torch
from transformers import (DynamicCache, LlamaConfig, LlamaForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM)

from kvstudy.eviction import EvictionSpec, make_cache
from kvstudy.eviction_generation import _forward, evicting_generate, eviction_nll


def tiny_model(family="llama", attention="eager"):
    configs = {"llama": (LlamaConfig, LlamaForCausalLM),
               "qwen2": (Qwen2Config, Qwen2ForCausalLM),
               "mistral": (MistralConfig, MistralForCausalLM)}
    config_class, model_class = configs[family]
    torch.manual_seed(17)
    config = config_class(vocab_size=41, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         attention_dropout=0.0, sliding_window=None)
    config._attn_implementation = attention
    return model_class(config).eval()


def chunk_logits(model, ids, spec, chunk):
    cache = make_cache(spec)
    with torch.no_grad():
        logits = [_forward(model, ids[:, start:start + chunk], cache).logits
                  for start in range(0, ids.shape[1], chunk)]
    return torch.cat(logits, dim=1), cache


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("policy", ["full", "recent", "sink_window"])
def test_unrestricted_matches_native_full_cache(policy, family, attention):
    model = tiny_model(family, attention)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])
    with torch.no_grad():
        expected = model(ids, past_key_values=DynamicCache(), use_cache=True).logits
    actual, cache = chunk_logits(model, ids, EvictionSpec(policy=policy), 3)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert cache.retained_positions.tolist() == list(range(ids.shape[1]))


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("policy,sinks", [("recent", 0), ("sink_window", 2)])
def test_eviction_chunk_boundaries_preserve_logits(policy, sinks, family, attention):
    model = tiny_model(family, attention)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17]])
    spec = EvictionSpec(policy=policy, budget_tokens=4, sink_tokens=sinks)
    sequential, _ = chunk_logits(model, ids, spec, 1)
    chunked, cache = chunk_logits(model, ids, spec, 5)
    torch.testing.assert_close(chunked, sequential, atol=2e-6, rtol=2e-5)
    assert cache.get_seq_length() == 4
    assert cache.seen_tokens == 9
    assert cache.retained_positions.tolist() == ([5, 6, 7, 8] if policy == "recent" else [0, 1, 7, 8])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_physical_budget_and_precision(dtype):
    cache = make_cache(EvictionSpec(policy="sink_window", budget_tokens=5, sink_tokens=2))
    x = torch.randn(2, 3, 12, 16).to(dtype)
    k, v = cache.update(x, x + 1, 0)
    assert k.shape[-2] == v.shape[-2] == 12
    layer = cache.layers[0]
    assert layer.keys.shape[-2] == 5
    assert cache.retained_positions.tolist() == [0, 1, 9, 10, 11]
    summary = cache.summary()
    assert summary["storage_bytes"] == 2 * 2 * 3 * 5 * 16 * x.element_size()
    assert summary["metadata_bytes"] == 5 * 8
    assert layer.keys.dtype == layer.values.dtype == dtype
    assert layer.keys.untyped_storage().nbytes() == layer.keys.numel() * layer.keys.element_size()


def test_budget_boundary_and_reset():
    cache = make_cache(EvictionSpec(policy="recent", budget_tokens=3))
    x = torch.ones(1, 1, 3, 4)
    cache.update(x, x, 0)
    assert cache.retained_positions.tolist() == [0, 1, 2]
    cache.update(x[..., :1, :], x[..., :1, :], 0)
    assert cache.retained_positions.tolist() == [1, 2, 3]
    assert cache.seen_tokens == 4
    cache.reset()
    assert cache.summary()["storage_bytes"] == 0
    assert cache.seen_tokens == 0


def test_mask_uses_absolute_positions_and_excludes_future():
    cache = make_cache(EvictionSpec(policy="sink_window", budget_tokens=3, sink_tokens=1))
    x = torch.ones(1, 1, 8, 4)
    cache.update(x, x, 0)
    mask = cache.attention_mask(3, torch.device("cpu"), torch.float32)[0, 0]
    # Candidate positions are [0,6,7,8,9,10]. Query positions are [8,9,10].
    allowed = mask == 0
    assert allowed.tolist() == [[True, False, True, True, False, False],
                                [True, False, False, True, True, False],
                                [True, False, False, False, True, True]]


def test_reject_empty_and_invalid_specs():
    x = torch.empty(1, 1, 0, 4)
    with pytest.raises(ValueError, match="empty"):
        make_cache(EvictionSpec()).update(x, x, 0)
    for kwargs in [dict(policy="bad"), dict(budget_tokens=0),
                   dict(policy="sink_window", budget_tokens=4, sink_tokens=4)]:
        with pytest.raises(ValueError):
            EvictionSpec(**kwargs)


def test_gpu_entry_points_reject_cpu():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="CUDA"):
        evicting_generate(model, ids, 2, torch.device("cpu"), EvictionSpec())
    with pytest.raises(ValueError, match="CUDA"):
        eviction_nll(model, ids, 2, EvictionSpec())


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
def test_chunked_nll_covers_each_target(family):
    model = tiny_model(family)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    spec = EvictionSpec(policy="sink_window", budget_tokens=3, sink_tokens=1)
    logits, _ = chunk_logits(model, ids[:, :-1], spec, 1)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 ids[:, 1:].reshape(-1), reduction="sum")
    chunked, _ = chunk_logits(model, ids[:, :-1], spec, 3)
    actual = torch.nn.functional.cross_entropy(chunked.reshape(-1, chunked.shape[-1]),
                                               ids[:, 1:].reshape(-1), reduction="sum")
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

"""Download-free family and workload contracts; real checkpoint/GPU runs are separate."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                          Qwen2ForCausalLM, MistralConfig, MistralForCausalLM)

from kvstudy.families import (native_sliding_window_enabled, prefix_token_id,
                              resolve_dtype, validate_model_config, validate_sequence_length)
from kvstudy.modelio import _logits_kw, greedy_generate
from kvstudy.workloads.needle import NeedleBuilder


@pytest.mark.parametrize("config_cls,model_cls", [(LlamaConfig,LlamaForCausalLM),
                                                 (Qwen2Config,Qwen2ForCausalLM),
                                                 (MistralConfig,MistralForCausalLM)])
def test_tiny_full_cache_families(config_cls, model_cls):
    config = config_cls(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, sliding_window=None,
                        use_sliding_window=False, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    torch.manual_seed(7)
    model = model_cls(config).float().eval()
    validate_model_config(config, {"sweep":{"contexts":[16],"decode_tokens":4}}, eviction=True)
    ids = torch.tensor([[3,4,5,6]])
    result = greedy_generate(model, ids, 4, torch.device("cpu"))
    with torch.no_grad():
        reference = ids.clone()
        for _ in range(4):
            nxt = model(input_ids=reference, **_logits_kw(model)).logits[:,-1].argmax(-1,keepdim=True)
            reference = torch.cat([reference,nxt],dim=1)
    assert torch.equal(result["tokens"], reference[:,ids.shape[1]:])
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    expected = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * ids.shape[1] * 4
    assert result["cache_bytes_measured"] == expected


def test_qwen_disabled_native_window():
    cfg = SimpleNamespace(model_type="qwen2", sliding_window=32768, use_sliding_window=False,
                          max_position_embeddings=32768)
    assert not native_sliding_window_enabled(cfg)
    validate_model_config(cfg, {}, eviction=True)
    cfg.use_sliding_window=True
    with pytest.raises(ValueError, match="sliding-window"):
        validate_model_config(cfg, {}, eviction=True)


def test_mistral_enabled_window_rejected_only_for_eviction():
    cfg = SimpleNamespace(model_type="mistral", sliding_window=4096, max_position_embeddings=32768)
    validate_model_config(cfg, {}, eviction=False)
    with pytest.raises(ValueError, match="sliding-window"):
        validate_model_config(cfg, {}, eviction=True)


def test_sequence_limit_counts_generation():
    cfg = SimpleNamespace(max_position_embeddings=32768)
    validate_sequence_length(cfg,32736,32)
    with pytest.raises(ValueError,match="exceeds"):
        validate_sequence_length(cfg,32768,32)
    with pytest.raises(ValueError,match="exceeds"):
        validate_model_config(cfg,{"needle":{"contexts":[32768],"max_new_tokens":20}})


def test_dtype_is_explicit():
    assert resolve_dtype({"dtype":"float32"},"cpu") == torch.float32
    assert resolve_dtype({"dtype":"bfloat16"},"cpu") == torch.bfloat16
    with pytest.raises(ValueError,match="Unsupported dtype"):
        resolve_dtype({"dtype":"int8"},"cpu")
    with patch("torch.cuda.is_available",return_value=False):
        with pytest.raises(RuntimeError,match="unavailable"):
            resolve_dtype({"dtype":"bfloat16"},"cuda")


def test_prefix_never_substitutes_eos():
    tok=SimpleNamespace(bos_token_id=None,eos_token_id=9)
    assert prefix_token_id(tok,SimpleNamespace(bos_token_id=7)) == 7
    assert prefix_token_id(tok,SimpleNamespace()) is None
    tok.bos_token_id=3
    assert prefix_token_id(tok,SimpleNamespace(bos_token_id=7)) == 3


class TemplateTokenizer:
    """Character-level stand-in to exercise three chat scaffolds without downloads."""
    def __init__(self,family):
        self.family=family
    def __call__(self,text,add_special_tokens=False):
        return SimpleNamespace(input_ids=list(text.encode("utf-8")))
    def decode(self,ids):
        return bytes(ids).decode("utf-8")
    def apply_chat_template(self,messages,add_generation_prompt,tokenize,date_string):
        assert messages[0]["role"]=="user"
        content=messages[0]["content"]
        if self.family=="llama":
            return f"<bos><user>{date_string}{content}<assistant>"
        if self.family=="qwen":
            return f"<im_start>user\n{content}<im_end>\n<im_start>assistant\n"
        return f"<s>[INST] {content} [/INST]"


@pytest.mark.parametrize("family",["llama","qwen","mistral"])
def test_needle_templates_exact_length_and_determinism(family):
    tok=TemplateTokenizer(family)
    builder=NeedleBuilder(tok,"harmless filler text "*1000,seed=12)
    first=builder.build(1024,0.5,3)
    builder.build(2048,1.0,0)
    assert first==builder.build(1024,0.5,3)
    assert len(first["ids"])==1024
    text=tok.decode(first["ids"])
    assert text.count(str(first["code"]))==1
    assert str(first["code"]) in tok.decode(first["ids"][first["needle_start"]:
                                                                first["needle_start"]+first["needle_len"]])


def test_baseline_identity_prevents_dtype_mix(tmp_path):
    from kvstudy.families import validate_run_identity
    cfg={"model_family":"qwen2", "_results_dir":str(tmp_path), "dtype":"float16"}
    validate_run_identity(cfg,"test/model","revision1")
    validate_run_identity(cfg,"test/model","revision1")
    with pytest.raises(ValueError,match="identity mismatch"):
        validate_run_identity({**cfg,"dtype":"bfloat16"},"test/model","revision1")
    with pytest.raises(ValueError,match="identity mismatch"):
        validate_run_identity(cfg,"test/model","revision2")


def test_baseline_identity_checks_existing_meta(tmp_path):
    import json
    from kvstudy.families import validate_run_identity
    (tmp_path/"meta.json").write_text(json.dumps({"model":"test/model","dtype":"bfloat16", "model_revision":"r"}))
    cfg={"model_family":"qwen2", "_results_dir":str(tmp_path), "dtype":"float16"}
    with pytest.raises(ValueError,match="identity mismatch"):
        validate_run_identity(cfg,"test/model","r")

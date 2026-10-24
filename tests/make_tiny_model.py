"""Build a tiny random Llama-style model + tokenizer with a Llama-3-like chat template, for CPU smoke tests.

The model is untrained (its answers are random); the test only checks that every stage of the pipeline runs,
logs, resumes and reports correctly. No download needed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from kvstudy.workloads.text import synthetic_text  # noqa: E402

OUT = HERE / "tiny_llama"
TEMPLATE = (
    "{{- bos_token }}"
    "{%- if not date_string is defined %}{%- set date_string = '26 Jul 2024' %}{%- endif %}"
    "<|start_header_id|>system<|end_header_id|>\n\nToday Date: {{ date_string }}<|eot_id|>"
    "{%- for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{%- endif %}"
)


def main():
    text = synthetic_text(60_000, seed=1) + " The access code for the blue vault is 1234567890. Based only on the text"
    specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator([text], trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
    tok.chat_template = TEMPLATE
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=131072,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    OUT.mkdir(exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    rows = [l + "\n" for l in synthetic_text(80_000, seed=2).split(". ")]
    (HERE / "tiny_text.json").write_text(json.dumps(rows))
    print(f"tiny model ({sum(p.numel() for p in model.parameters())} params, vocab {len(tok)}) -> {OUT}")


if __name__ == "__main__":
    main()

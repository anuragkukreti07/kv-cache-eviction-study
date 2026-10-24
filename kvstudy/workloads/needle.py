"""Needle-in-a-haystack prompts with exact token-level control (guide Step 4)."""
import random

QUESTION = ("Based only on the text above, what is the access code for the blue vault? "
            "Answer with the number only.")
_MARK = "<<CTX>>"
# Llama-3.x chat templates insert today's date into the system header unless told otherwise.
# A fixed date keeps prompts byte-identical across days, machines and reruns.
FIXED_DATE = "26 Jul 2024"


class NeedleBuilder:
    def __init__(self, tok, filler_text, seed=0):
        self.tok = tok
        self.seed = seed
        self.filler = tok(filler_text, add_special_tokens=False).input_ids
        msgs = [{"role": "user", "content": f"{_MARK}\n\n{QUESTION}"}]
        templ = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, date_string=FIXED_DATE)
        head, tail = templ.split(_MARK)
        self.head = tok(head, add_special_tokens=False).input_ids   # template already contains BOS text
        self.tail = tok(tail, add_special_tokens=False).input_ids

    def build(self, context, depth, sample):
        """Return dict(ids, code, needle_start, needle_len, n_filler). len(ids) == context exactly.

        Each (context, depth, sample) has its own seeded RNG, so a prompt never depends on the order in which
        prompts were built (runs can be interrupted, resumed or reordered and still see identical prompts).
        """
        rng = random.Random(f"needle-{self.seed}-{context}-{depth}-{sample}")
        while True:
            code = rng.randint(1000, 9999)
            needle = self.tok(f" The access code for the blue vault is {code}.\n",
                              add_special_tokens=False).input_ids
            n_filler = context - len(self.head) - len(self.tail) - len(needle)
            if n_filler <= 0:
                raise ValueError(f"context {context} too short for the prompt scaffold")
            if n_filler > len(self.filler):
                raise ValueError(f"filler text has {len(self.filler)} tokens, context {context} needs {n_filler}")
            off = rng.randint(0, len(self.filler) - n_filler)
            seg = self.filler[off:off + n_filler]
            if str(code) in self.tok.decode(seg):   # code must not occur naturally in the filler
                continue
            break
        k = int(round(depth * n_filler))
        ids = self.head + seg[:k] + needle + seg[k:] + self.tail
        assert len(ids) == context
        return dict(ids=ids, code=code, needle_start=len(self.head) + k, needle_len=len(needle),
                    n_filler=n_filler)

"""Filler / evaluation text: wikitext-2 (test split) from the Hugging Face hub, a local text file, or a seeded
synthetic fallback (filler only).

Set `text_file: path/to/file.txt` in the config to use a local file instead of downloading (useful on a
machine without internet). setup_server.sh saves wikitext-2 to data/wikitext2_test.json (the exact dataset rows)
so later runs need no network.
"""
import json
import os
import random

from kvstudy.config import ROOT

_WORDS = ("river stone market winter garden letter window engine paper silver morning harbor "
          "lantern valley bridge forest copper village season journey evening mountain ocean "
          "kitchen library street travel measure quiet bright narrow story simple carry gather "
          "ancient modern council planet signal summer pattern thread meadow canvas orchard "
          "compass anchor whisper marble feather timber cobble ribbon saddle thistle").split()

LOCAL_WIKITEXT = ROOT / "data" / "wikitext2_test.json"


def synthetic_text(n_words=400_000, seed=0):
    rng = random.Random(seed)
    sents, count = [], 0
    while count < n_words:
        k = rng.randint(6, 12)
        sents.append(" ".join(rng.choice(_WORDS) for _ in range(k)).capitalize() + ".")
        count += k
    return " ".join(sents)


def load_wikitext_lines(cfg=None, split="test"):
    """wikitext-2-raw rows. Order: config text_file (.txt or .json list), saved local copy, Hugging Face hub."""
    path = (cfg or {}).get("text_file")
    for p in [path, str(LOCAL_WIKITEXT)]:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                rows = json.load(f) if p.endswith(".json") else f.read().split("\n")
            return rows, f"wikitext-2-raw-v1 ({split}, local copy)"
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return list(ds["text"]), f"wikitext-2-raw-v1 ({split})"


def save_local_wikitext():
    """Download once and save the exact dataset rows as JSON, so later runs need no network."""
    lines, src = load_wikitext_lines()
    LOCAL_WIKITEXT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_WIKITEXT.write_text(json.dumps(lines), encoding="utf-8")
    return LOCAL_WIKITEXT


def filler_text(cfg=None):
    """Returns (text, source_name). Headings dropped, wikitext tokenisation artefacts cleaned."""
    try:
        raw, src = load_wikitext_lines(cfg)
        lines = [l.strip() for l in raw if l.strip() and not l.strip().startswith("=")]
        text = " ".join(lines)
        for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
            text = text.replace(a, b)
        return text, src
    except Exception as e:
        print(f"[text] wikitext unavailable ({type(e).__name__}); using synthetic filler")
        return synthetic_text(), "synthetic"


def perplexity_text(cfg=None):
    """Raw wikitext-2 test text for perplexity, or (None, None) if unavailable."""
    try:
        raw, src = load_wikitext_lines(cfg)
        return "\n\n".join(raw), src
    except Exception as e:
        print(f"[text] wikitext unavailable for perplexity ({type(e).__name__})")
        return None, None

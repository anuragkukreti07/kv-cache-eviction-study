"""Download everything a run needs (model weights, tokenizer, wikitext-2) and pin the model revisions.

After this succeeds, runs can work offline (run_server.sh sets HF_HUB_OFFLINE=1), so no time in a GPU slot is lost
to network checks, and a run cannot silently pick up a changed model.

  python scripts/prefetch.py --config configs/server_3b.yaml --config configs/server_1b.yaml
"""
import argparse
import sys
import time

import _boot  # noqa: F401

from kvstudy.config import ROOT, load_config
from kvstudy.modelio import resolve_revision

MARKER = ROOT / ".prefetch_ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    ok = True
    for path in args.config:
        cfg = load_config(path)
        name = cfg["model"]
        t = time.time()
        try:
            rev = resolve_revision(cfg, name)
            if rev is None:
                raise RuntimeError("could not resolve the model revision (no internet access to huggingface.co?)")
            local = snapshot_download(name, revision=rev,
                                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"])
            print(f"[ok] {name} @ {rev[:10]} -> {local} ({time.time() - t:.0f} s)", flush=True)
        except Exception as e:
            ok = False
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    try:
        from kvstudy.workloads.text import save_local_wikitext, LOCAL_WIKITEXT
        if LOCAL_WIKITEXT.exists():
            print(f"[ok] wikitext-2 already saved at {LOCAL_WIKITEXT}")
        else:
            p = save_local_wikitext()
            print(f"[ok] wikitext-2 saved to {p}")
    except Exception as e:
        ok = False
        print(f"[FAIL] wikitext-2: {type(e).__name__}: {str(e)[:300]}")
    if ok:
        MARKER.write_text("ok\n")
        print("[prefetch] everything is local; runs can now work offline")
    elif MARKER.exists():
        MARKER.unlink()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

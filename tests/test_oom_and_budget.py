"""CPU test of the out-of-memory and time-budget logic in the sweep and needle stages (no GPU needed).

Fakes an out-of-memory error whenever batch x context > LIMIT tokens and checks that:
  * the first OOM per batch is logged once, and larger contexts for that batch are skipped,
  * a stage stopped by its time budget resumes later without redoing finished points.
"""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import torch  # noqa: E402

from kvstudy import modelio  # noqa: E402

LIMIT = 1024
real = modelio.greedy_generate


def fake(model, ids, max_new, dev, eos_ids=None):
    if ids.shape[0] * ids.shape[1] > LIMIT:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
    return real(model, ids, max_new, dev, eos_ids)


modelio.greedy_generate = fake
CFG = "tests/test_oom.yaml"
Path(CFG).write_text(Path("tests/test_cpu.yaml").read_text().replace("run_name: test_cpu", "run_name: test_oom")
                     .replace("sweep: {contexts: [256, 512, 1024], batches: [1, 2], decode_tokens: 6, trials: 2, max_minutes: 5}",
                              "sweep: {contexts: [256, 512, 1024, 2048], batches: [1, 2, 4], decode_tokens: 4, trials: 3, max_minutes: 5}"))
out = ROOT / "results" / "test_oom"
if out.exists():
    import shutil
    shutil.rmtree(out)


def run(script, minutes=None):
    if minutes is not None:
        os.environ["KV_DEADLINE"] = str(__import__("time").time() + 60 * minutes)
    else:
        os.environ.pop("KV_DEADLINE", None)
    sys.argv = [script, "--config", CFG]
    try:
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
    except SystemExit:
        pass


def rows(name):
    return [json.loads(l) for l in open(out / name)]


run("s1_sweep.py", minutes=0.03)           # ~2 s: stops after a few points
n_first = len(rows("baseline_sweep.jsonl"))
run("s1_sweep.py")                           # resume to completion
r = rows("baseline_sweep.jsonl")
ok = [x for x in r if x["status"] == "ok"]
oom = [x for x in r if x["status"] == "oom"]
keys = [(x["batch"], x["context"], x["trial"]) for x in ok]
assert len(keys) == len(set(keys)), "a finished point was measured twice after resume"
assert sorted((x["batch"], x["context"]) for x in oom) == [(1, 2048), (2, 1024), (4, 512)], oom
for b, c, t in keys:
    assert b * c <= LIMIT
assert len(ok) == 3 * (3 + 2 + 1), len(ok)
print(f"[ok] sweep: {n_first} rows before the budget stop, {len(ok)} ok + {len(oom)} oom after resume, no duplicates")

LIMIT = 300                                   # needle: contexts 256 ok, 512 out of memory
run("s3_needle.py")
n = rows("baseline_needle.jsonl")
assert all(x["context"] == 256 for x in n if x["status"] == "ok")
assert sum(x["status"] == "oom" for x in n) == 1
print(f"[ok] needle: {sum(x['status'] == 'ok' for x in n)} prompts at 256, one OOM at 512, larger contexts skipped")

import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path, quick=False):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    q = cfg.pop("quick", {})
    if quick:
        cfg = _merge(copy.deepcopy(cfg), q)
    cfg["_quick"] = bool(quick)
    name = cfg.get("run_name", "baseline") + ("_quick" if quick else "")
    cfg["_results_dir"] = str(ROOT / "results" / name)
    Path(cfg["_results_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg

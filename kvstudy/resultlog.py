"""Append-only JSONL result files; one line per run, skip configs already done."""
import json
from pathlib import Path


class ResultLog:
    def __init__(self, path, env=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = env or {}
        self._rows = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # half-written last line after a crash

    def find(self, **key):
        return [r for r in self._rows if all(r.get(k) == v for k, v in key.items())]

    def has(self, **key):
        return bool(self.find(**key))

    def append(self, rec):
        rec = {**rec, **self.env}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        self._rows.append(rec)

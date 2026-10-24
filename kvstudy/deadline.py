"""Time budget for GPU slots.

run_all.py sets KV_DEADLINE (unix seconds) for the whole run; each stage may also have its own
`max_minutes` in the config. Stages check `expired()` before starting each new configuration, so a slot
that is about to end stops cleanly with every finished result already on disk.
"""
import os
import time


def global_deadline():
    v = os.environ.get("KV_DEADLINE")
    return float(v) if v else None


class Budget:
    def __init__(self, max_minutes=None, name="stage"):
        self.name = name
        self.t0 = time.time()
        cands = [d for d in (global_deadline(), self.t0 + 60 * max_minutes if max_minutes else None) if d]
        self.deadline = min(cands) if cands else None
        self.stopped = False

    def remaining_s(self):
        return None if self.deadline is None else self.deadline - time.time()

    def expired(self, margin_s=0.0):
        if self.deadline is not None and time.time() + margin_s >= self.deadline:
            if not self.stopped:
                print(f"[budget] {self.name}: time budget reached, stopping cleanly (finished results are saved)",
                      flush=True)
            self.stopped = True
            return True
        return False

    def describe(self):
        r = self.remaining_s()
        return "no time limit" if r is None else f"{r / 60:.1f} min available"

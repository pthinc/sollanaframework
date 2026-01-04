# memory_traces.py
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

PHI = (1.0 + 5**0.5) / 2.0  # altın oran

@dataclass
class BehaviorTrace:
    id: str
    N0: float = 1.0
    lambda_: float = 0.01
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    decay_rate: float = 0.01
    golden_score: float = 0.0
    meta: dict = field(default_factory=dict)

    def age(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return now - self.last_used

    def strength(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        elapsed = now - self.last_used
        return self.N0 * math.exp(-self.lambda_ * elapsed)

    def compute_golden(self) -> float:
        self.golden_score = (self.usage_count * (1.0 - self.decay_rate)) / PHI
        return self.golden_score

    def touch(self, delta_N: float = 0.0, now: Optional[float] = None):
        now = now if now is not None else time.time()
        cur_str = self.strength(now)
        self.N0 = max(0.0, cur_str + float(delta_N))
        self.last_used = now
        self.usage_count += 1
        self.compute_golden()

class MemoryLedger:
    def __init__(self, prune_strength_threshold: float = 1e-4, golden_threshold: float = 0.5):
        self.traces: Dict[str, BehaviorTrace] = {}
        self.prune_strength_threshold = prune_strength_threshold
        self.golden_threshold = golden_threshold

    def ensure_trace(self, trace_id: str, **kwargs):
        if trace_id not in self.traces:
            self.traces[trace_id] = BehaviorTrace(id=trace_id, **kwargs)

    def commit_behavior(self, trace_id: str, delta_N: float, decay_rate: Optional[float] = None, now: Optional[float] = None):
        now = now if now is not None else time.time()
        self.ensure_trace(trace_id)
        tr = self.traces[trace_id]
        if decay_rate is not None:
            tr.decay_rate = decay_rate
        tr.touch(delta_N=delta_N, now=now)

    def sweep_prune(self, now: Optional[float] = None):
        now = now if now is not None else time.time()
        to_delete = []
        for tid, tr in self.traces.items():
            if tr.golden_score < self.golden_threshold and tr.strength(now) < self.prune_strength_threshold:
                to_delete.append(tid)
        for tid in to_delete:
            del self.traces[tid]

    def recompute_all_golden(self):
        for tr in self.traces.values():
            tr.compute_golden()

# decay_core.py
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class DecayTrace:
    id: str
    N0: float
    lambda_: float
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def age(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        return now - self.created_at

    def strength(self, now: Optional[float] = None) -> float:
        t = self.age(now)
        return self.N0 * math.exp(-self.lambda_ * t)

    def update(self, delta_N: float, now: Optional[float] = None):
        if now is None:
            now = time.time()
        current = self.strength(now)
        new_N0 = current + delta_N
        self.N0 = new_N0
        self.created_at = now
        self.last_updated = now

class BehavioralMemory:
    def __init__(self, prune_threshold: float = 1e-4):
        self.traces: Dict[str, DecayTrace] = {}
        self.prune_threshold = prune_threshold

    def add_trace(self, trace_id: str, strength: float, lambda_: float, meta: dict = None):
        self.traces[trace_id] = DecayTrace(id=trace_id, N0=strength, lambda_=lambda_, meta=meta or {})

    def get_strength(self, trace_id: str, now: Optional[float] = None) -> float:
        trace = self.traces.get(trace_id)
        if trace is None:
            return 0.0
        return trace.strength(now)

    def sweep_prune(self, now: Optional[float] = None):
        if now is None:
            now = time.time()
        to_delete = [tid for tid, tr in self.traces.items() if tr.strength(now) < self.prune_threshold]
        for tid in to_delete:
            del self.traces[tid]

    def decay_penalty(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        return sum(tr.strength(now) for tr in self.traces.values())

    def list_traces(self) -> Dict[str, Tuple[float, float]]:
        now = time.time()
        return {tid: (tr.strength(now), tr.lambda_) for tid, tr in self.traces.items()}

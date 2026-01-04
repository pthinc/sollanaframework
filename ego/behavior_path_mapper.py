# behavior_path_mapper.py
import time
import math
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from collections import deque

PHI = (1.0 + 5**0.5) / 2.0

def x_t_scalar(t: float, clamp_max: float = 20.0) -> float:
    t_c = min(t, clamp_max)
    return float(math.tanh(math.exp(t_c) - math.pi))

@dataclass
class PathStep:
    ts: float
    module: str
    params: Dict[str, float]
    phi: float

class BehaviorPathMapper:
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 transforms: Optional[Dict[str, Callable[[float, float], float]]] = None,
                 decay_lambda: float = 0.01,
                 history_limit: int = 1000,
                 persist_dir: str = "paths",
                 memory=None,
                 pattern_tracker=None,
                 character_map_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.weights = weights or {
            "attention": 1.0,
            "match_prob": 1.0,
            "context_weight": 1.0,
            "activation": 1.0,
            "ethical": 1.0,
            "anomaly_penalty": -2.0
        }
        self.transforms = transforms or {
            "attention": lambda v, t: v,
            "match_prob": lambda v, t: math.log(max(v, 1e-9)),
            "context_weight": lambda v, t: v,
            "activation": lambda v, t: x_t_scalar(v),
            "ethical": lambda v, t: (1.0 if v in (1, "approved", "true") else 0.0),
            "anomaly_penalty": lambda v, t: float(-abs(v))
        }
        self.decay_lambda = decay_lambda
        self.history: deque = deque(maxlen=history_limit)
        self.persist_dir = persist_dir
        self.memory = memory
        self.pattern_tracker = pattern_tracker
        self.character_map_hook = character_map_hook

    def _compute_phi(self, params: Dict[str, float], ts: float) -> float:
        s = 0.0
        for k, v in params.items():
            if k not in self.weights or k not in self.transforms:
                continue
            fi = self.transforms[k](v, ts)
            s += float(self.weights[k]) * float(fi)
        return float(s)

    def record_step(self, behavior_id: str, module: str, params: Dict[str, float], ts: Optional[float] = None):
        ts = ts or time.time()
        phi = self._compute_phi(params, ts)
        step = PathStep(ts=ts, module=module, params=params.copy(), phi=phi)
        self.history.append((behavior_id, step))
        # integrate with TemporalMemory and PatternTracker if available
        if self.memory is not None:
            trace_id = f"path_{behavior_id}_{int(ts*1000)}"
            try:
                self.memory.trigger_behavior(trace_id, context=params.get("context","unknown"),
                                             delta_N=phi, decay_rate=params.get("decay_rate", 0.01), now=ts)
            except Exception:
                pass
        if self.pattern_tracker is not None:
            try:
                self.pattern_tracker.record(behavior_id, context_match=params.get("match_prob", 1.0), delta=phi, now=ts)
            except Exception:
                pass
        # optional character hook
        if self.character_map_hook is not None:
            try:
                stats = {"phi": phi, "ts": ts, "params": params}
                self.character_map_hook(behavior_id, stats)
            except Exception:
                pass
        return step

    def cumulative_phi(self, behavior_id: str, now: Optional[float] = None) -> float:
        now = now or time.time()
        total = 0.0
        for bid, step in list(self.history):
            if bid != behavior_id:
                continue
            age = now - step.ts
            weight = math.exp(-self.decay_lambda * age)
            total += step.phi * weight
        return float(total)

    def export_path(self, behavior_id: str) -> Dict[str, Any]:
        steps = [step for bid, step in self.history if bid == behavior_id]
        return {
            "behavior_id": behavior_id,
            "steps": [{"ts": s.ts, "module": s.module, "params": s.params, "phi": s.phi} for s in steps],
            "cumulative_phi": self.cumulative_phi(behavior_id),
            "exported_at": time.time()
        }

    def persist_path(self, behavior_id: str, to_file: Optional[str] = None):
        obj = self.export_path(behavior_id)
        path = to_file or f"{self.persist_dir}/{behavior_id}.path.bce"
        import os, tempfile
        os.makedirs(self.persist_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.persist_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
        return path

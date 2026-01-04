# decay_and_path.py
import time
import math
import json
from typing import Dict, List, Callable, Optional

EPS = 1e-9
PHI = (1.0 + 5**0.5) / 2.0

class DecayModel:
    def __init__(self, base_lambda: float = 0.01):
        self.base_lambda = float(base_lambda)

    def decay(self, t: float, lambda_: Optional[float] = None) -> float:
        lam = float(lambda_ if lambda_ is not None else self.base_lambda)
        return 1.0 - math.exp(-lam * t)

    def adjusted_lambda(self, lambda_: float, ethical_acceptance: float, alpha: float = 1.0) -> float:
        e = float(max(0.0, min(1.0, ethical_acceptance)))
        return float(lambda_ + alpha * (1.0 - e))

class PathTrace:
    def __init__(self, behavior_id: str, decay_model: DecayModel, decay_lambda: float = 0.01):
        self.behavior_id = behavior_id
        self.decay_model = decay_model
        self.decay_lambda = float(decay_lambda)
        self.steps: List[Dict] = []

    def add_step(self, module: str, params: Dict[str, float], ts: Optional[float] = None, weight: float = 1.0, transform: Optional[Callable[[str, float], float]] = None):
        ts = ts or time.time()
        phi = 0.0
        for k, v in params.items():
            val = float(v)
            fi = transform(k, val) if transform is not None else val
            phi += float(weight) * float(fi)
        step = {"ts": ts, "module": module, "params": dict(params), "phi": phi}
        self.steps.append(step)
        return step

    def cumulative_phi(self, now: Optional[float] = None, decay_lambda: Optional[float] = None) -> float:
        now = now or time.time()
        lam = float(decay_lambda if decay_lambda is not None else self.decay_lambda)
        total = 0.0
        for s in self.steps:
            age = max(0.0, now - s["ts"])
            weight = math.exp(-lam * age)
            total += s["phi"] * weight
        return float(total)

    def composite_phi(self) -> float:
        return float(sum(s["phi"] for s in self.steps))

    def to_record(self) -> Dict:
        return {
            "behavior_id": self.behavior_id,
            "steps": list(self.steps),
            "composite_phi": self.composite_phi(),
            "last_cumulative_phi": self.cumulative_phi()
        }

# Utility transforms and a default f_i set
def default_transform(key: str, v: float) -> float:
    if key == "match_prob":
        return math.log(max(v, EPS))
    if key == "activation":
        return math.tanh(math.exp(min(v, 20.0)) - math.pi)
    if key in ("attention", "context_weight"):
        return float(v)
    if key == "ethical":
        return 1.0 if v in (1, "approved", "true") else 0.0
    return float(v)

# Example: create a trace, add steps, compute cumulative phi and decay
if __name__ == "__main__":
    dm = DecayModel(base_lambda=0.02)
    trace = PathTrace("greet_001", dm, decay_lambda=0.02)
    trace.add_step("tokenize", {"attention":0.8, "match_prob":0.7, "activation":0.0}, weight=1.0, transform=default_transform)
    time.sleep(0.01)
    trace.add_step("attention", {"attention":0.82, "context_weight":0.9, "activation":0.1}, weight=1.0, transform=default_transform)
    time.sleep(0.01)
    trace.add_step("ethics", {"ethical":"approved", "anomaly_penalty":0.0}, weight=1.0, transform=default_transform)
    now = time.time()
    rec = trace.to_record()
    rec["cumulative_phi_now"] = trace.cumulative_phi(now)
    print(json.dumps(rec, indent=2))

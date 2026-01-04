"""
Trust Control v2: time-decayed trust with evidence windows and escalation thresholds.
"""
from typing import Dict, Any, List
import time
import math

EPS = 1e-12


def time_decay_trust(events: List[Dict[str, float]], half_life_s: float = 300.0) -> float:
    now = time.time()
    weights = []
    vals = []
    for ev in events:
        ts = float(ev.get("ts", now))
        v = float(ev.get("val", 0.0))
        dt = max(0.0, now - ts)
        w = math.exp(-math.log(2) * dt / max(EPS, half_life_s))
        weights.append(w)
        vals.append(v)
    if not weights:
        return 0.0
    num = sum(w * v for w, v in zip(weights, vals))
    den = sum(weights) + EPS
    return float(num / den)


def evaluate(events: List[Dict[str, float]], thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    thresholds = thresholds or {"high": 0.75, "medium": 0.55, "low": 0.4}
    score = time_decay_trust(events)
    if score >= thresholds["high"]:
        level = "high"
        actions = ["allow"]
    elif score >= thresholds["medium"]:
        level = "medium"
        actions = ["clarify"]
    elif score >= thresholds["low"]:
        level = "low"
        actions = ["throttle", "verify"]
    else:
        level = "critical"
        actions = ["suppress", "escalate"]
    return {"score": score, "level": level, "actions": actions}


def example_usage():
    now = time.time()
    ev = [{"ts": now - 60, "val": 0.8}, {"ts": now - 600, "val": 0.5}]
    return evaluate(ev)


if __name__ == "__main__":
    print(example_usage())

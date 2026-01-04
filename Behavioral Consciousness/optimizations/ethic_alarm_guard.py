"""
Ethic Alarm Guard (mad_etikalarm_egokoruma): combines risk signals and raises graded alarms.
"""
from typing import Dict, Any


def score_risk(signals: Dict[str, float], weights: Dict[str, float] = None) -> float:
    weights = weights or {"tox": 0.4, "drift": 0.3, "latency": 0.1, "novelty": 0.2}
    total_w = sum(weights.values()) or 1.0
    s = 0.0
    for k, w in weights.items():
        s += w * float(signals.get(k, 0.0))
    return float(s / total_w)


def classify(risk: float) -> str:
    if risk >= 0.75:
        return "critical"
    if risk >= 0.5:
        return "major"
    if risk >= 0.25:
        return "minor"
    return "ok"


def guard(signals: Dict[str, float]) -> Dict[str, Any]:
    r = score_risk(signals)
    level = classify(r)
    actions = []
    if level == "critical":
        actions = ["suppress", "snapshot", "escalate"]
    elif level == "major":
        actions = ["clarify", "debias"]
    elif level == "minor":
        actions = ["annotate"]
    return {"risk": r, "level": level, "actions": actions}


def example_usage():
    return guard({"tox":0.8,"drift":0.6,"latency":0.2})


if __name__ == "__main__":
    print(example_usage())

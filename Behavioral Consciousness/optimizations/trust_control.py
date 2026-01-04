"""
Trust Control v1: score trust based on consistency, honesty, drift, and novelty.
Returns trust score, level, and suggested actions.
"""
from typing import Dict, Any


def trust_score(signals: Dict[str, float], weights: Dict[str, float] = None) -> float:
    weights = weights or {"consistency": 0.35, "honesty": 0.3, "drift": 0.2, "novelty": 0.15}
    total = sum(weights.values()) or 1.0
    s = 0.0
    for k, w in weights.items():
        s += w * float(signals.get(k, 0.0))
    return float(max(0.0, min(1.0, s / total)))


def classify_trust(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "low"
    return "critical"


def actions_for_level(level: str):
    if level == "high":
        return ["allow", "log"]
    if level == "medium":
        return ["clarify", "log"]
    if level == "low":
        return ["throttle", "verify"]
    return ["suppress", "escalate"]


def evaluate_trust(signals: Dict[str, float]) -> Dict[str, Any]:
    score = trust_score(signals)
    level = classify_trust(score)
    return {"score": score, "level": level, "actions": actions_for_level(level)}


def example_usage():
    return evaluate_trust({"consistency":0.7,"honesty":0.8,"drift":0.2,"novelty":0.6})


if __name__ == "__main__":
    print(example_usage())

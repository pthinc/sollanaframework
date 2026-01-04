"""
KPI monitor for BCE flows.
Tracks rolling retention/safety/diversity/drift/latency and exposes a health score.
"""
from collections import deque
from typing import Dict, Any, List
import math
import time
import numpy as np

EPS = 1e-12


def _ema(prev: float, x: float, alpha: float) -> float:
    return float((1.0 - alpha) * prev + alpha * x)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _pct(p: float) -> float:
    return float(max(0.0, min(100.0, p * 100.0)))


class KPIMonitor:
    def __init__(self, alpha: float = 0.2, window: int = 128):
        self.alpha = float(alpha)
        self.window = int(max(8, window))
        self.latencies_ms: deque = deque(maxlen=self.window)
        self.history: List[Dict[str, Any]] = []
        # start neutral so early updates do not collapse health
        self.state = {
            "retention": 0.5,
            "safety": 0.5,
            "diversity": 0.5,
            "drift": 0.5,
            "bce_mean": 0.5,
            "latency_ms": 0.0,
        }

    def update(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        event keys (all optional):
            retained: bool/int
            safe: bool/int
            diversity: float in [0,1]
            drift_score: float in [0,1] (1=low drift)
            latency_ms: float
            bce_score: float in [0,1]
        returns snapshot with health and risk flags.
        """
        retained = 1.0 if event.get("retained") else 0.0
        safe = 1.0 if event.get("safe", True) else 0.0
        diversity = _clamp01(float(event.get("diversity", self.state["diversity"])))
        drift = _clamp01(float(event.get("drift_score", self.state["drift"])))
        bce_score = _clamp01(float(event.get("bce_score", self.state["bce_mean"])))
        latency_ms = float(event.get("latency_ms", self.state["latency_ms"]))

        self.state["retention"] = _ema(self.state["retention"], retained, self.alpha)
        self.state["safety"] = _ema(self.state["safety"], safe, self.alpha)
        self.state["diversity"] = _ema(self.state["diversity"], diversity, self.alpha)
        self.state["drift"] = _ema(self.state["drift"], drift, self.alpha)
        self.state["bce_mean"] = _ema(self.state["bce_mean"], bce_score, self.alpha)

        if latency_ms > 0:
            self.latencies_ms.append(latency_ms)
        self.state["latency_ms"] = float(latency_ms if self.latencies_ms else self.state["latency_ms"])

        health = self.health_score()
        badge = self.health_badge(health)
        snapshot = {
            "ts": time.time(),
            "health": health,
            "badge": badge,
            "retention": self.state["retention"],
            "safety": self.state["safety"],
            "diversity": self.state["diversity"],
            "drift": self.state["drift"],
            "bce_mean": self.state["bce_mean"],
            "latency_ms": self.state["latency_ms"],
            "p95_latency_ms": self.p95_latency(),
            "risks": self.risk_flags(),
        }
        self.history.append(snapshot)
        return snapshot

    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        arr = np.array(self.latencies_ms, dtype=float)
        return float(np.percentile(arr, 95))

    def health_score(self) -> float:
        latency_pen = self._latency_penalty(self.state["latency_ms"])
        components = [
            self.state["retention"],
            self.state["safety"],
            self.state["diversity"],
            self.state["drift"],
            self.state["bce_mean"],
            latency_pen,
        ]
        return float(sum(components) / len(components))

    def _latency_penalty(self, latency_ms: float) -> float:
        if latency_ms <= 0:
            return 1.0
        # soft penalty: <200ms ~1.0, 500ms ~0.5, slow tails drop further
        return float(max(0.0, math.exp(-0.005 * max(0.0, latency_ms - 50))))

    def risk_flags(self) -> Dict[str, bool]:
        return {
            "retention_low": self.state["retention"] < 0.5,
            "safety_low": self.state["safety"] < 0.8,
            "diversity_low": self.state["diversity"] < 0.4,
            "drift_high": self.state["drift"] < 0.5,
            "latency_high": self._latency_penalty(self.state["latency_ms"]) < 0.6,
        }

    def health_badge(self, health: float) -> str:
        h = _clamp01(health)
        if h >= 0.8:
            return "green"
        if h >= 0.6:
            return "yellow"
        return "red"

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"health": 0.0, "badge": "red", "samples": 0}
        last = self.history[-1]
        return {
            "health": last["health"],
            "badge": last["badge"],
            "samples": len(self.history),
            "retention_pct": _pct(last["retention"]),
            "safety_pct": _pct(last["safety"]),
            "diversity_pct": _pct(last["diversity"]),
            "drift_pct": _pct(last["drift"]),
            "bce_pct": _pct(last["bce_mean"]),
            "p95_latency_ms": last["p95_latency_ms"],
        }


def example_usage() -> Dict[str, Any]:
    monitor = KPIMonitor(alpha=0.3)
    events = [
        {"retained": True, "safe": True, "diversity": 0.7, "drift_score": 0.8, "latency_ms": 180, "bce_score": 0.75},
        {"retained": True, "safe": False, "diversity": 0.6, "drift_score": 0.6, "latency_ms": 320, "bce_score": 0.68},
        {"retained": False, "safe": True, "diversity": 0.5, "drift_score": 0.55, "latency_ms": 140, "bce_score": 0.7},
    ]
    snap = None
    for ev in events:
        snap = monitor.update(ev)
    return {"snapshot": snap, "summary": monitor.summary()}


if __name__ == "__main__":
    print(example_usage())

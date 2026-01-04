"""
Path scoring and activation helpers (doc-aligned).
- activation_curve: x(t) = tanh(e^t - pi)
- path_score: combines BCE, trust, decay risk and trace length with golden ratio weight
"""
from __future__ import annotations
import numpy as np
from typing import Any

PHI = 1.618033988749895
EPS = 1e-8


def activation_curve(t: float) -> float:
    return float(np.tanh(np.exp(float(t)) - np.pi))


def path_score(bce: float, decay_risk: float, trust_score: float, trace_len: int = 1) -> float:
    bce_s = max(0.0, bce)
    decay = max(EPS, 1.0 + decay_risk)
    trust_s = max(0.0, trust_score)
    length_boost = np.log1p(max(1, trace_len))
    score = ((PHI * bce_s) + trust_s) / decay
    score = score * length_boost
    return float(score)


__all__ = ["activation_curve", "path_score", "PHI"]

# dynamic_behavior.py
"""Dynamic behavior utilities with backend awareness (torch preferred)."""

import math
import time
from typing import Optional

from backends import ensure_backend

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

EPS = 1e-9
GOLDEN_THRESHOLD = 1.0 / ((1.0 + 5**0.5) / 2.0)  # ~0.618

def x_t(t, backend_name: Optional[str] = None):
    backend = ensure_backend(backend_name)
    if backend.name.startswith("torch") and _TORCH_AVAILABLE:
        t_clamped = torch.clamp(t, max=20.0)
        return torch.tanh(torch.exp(t_clamped) - math.pi)
    import numpy as np

    t_arr = np.array(t)
    t_clamped = np.minimum(t_arr, 20.0)
    return np.tanh(np.exp(t_clamped) - math.pi)

def adapt_prob_and_weight(P: float, W: float, data_quality: float, alpha: float = 0.1, beta: float = 0.1):
    Pp = min(1.0, max(EPS, P + alpha * (1.0 - data_quality)))
    Wp = W * (1.0 - beta * (1.0 - data_quality))
    return Pp, Wp

def generate_behavior_scalar(attention: float, match_prob: float, context_weight: float, timestamp: float, data_quality: float,
                             h_scale: float = 1.0, k_scale: float = 1.0, F_scale: float = 1.0,
                             backend_name: Optional[str] = None):
    Pp, Wp = adapt_prob_and_weight(match_prob, context_weight, data_quality)
    xt = x_t(timestamp, backend_name=backend_name)
    energy = h_scale * attention
    entropy = k_scale * math.log(Pp + EPS)
    transfer = F_scale * Wp
    return float(xt * (energy + entropy + transfer))

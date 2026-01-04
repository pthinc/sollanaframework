# activation.py
"""Activation helpers with torch/NumPy backends."""

import math
import numpy as np
from typing import Optional

from backends import ensure_backend

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def x_t_np(t, clamp: float = 20.0):
    t_c = np.minimum(t, clamp)
    return np.tanh(np.exp(t_c) - math.pi)


def x_t_torch(t, clamp: float = 20.0):
    t_c = torch.clamp(t, max=clamp)
    return torch.tanh(torch.exp(t_c) - math.pi)


def x_t(t, clamp: float = 20.0, backend_name: Optional[str] = None):
    backend = ensure_backend(backend_name)
    if backend.name.startswith("torch") and _TORCH_AVAILABLE:
        return x_t_torch(t, clamp=clamp)
    return x_t_np(t, clamp=clamp)


__all__ = ["x_t", "x_t_np", "x_t_torch"]

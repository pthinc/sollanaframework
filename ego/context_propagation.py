# context_propagation.py
"""Context propagation with backend awareness (torch preferred, NumPy fallback)."""

from typing import Optional

from backends import ensure_backend


try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

SI_FARADAY = 96485.33212

def propagate_context_scalar(neuron_outputs, context_weights, F_scale: float = 1e-3):
    propagated = []
    for i, out in enumerate(neuron_outputs):
        transfer = context_weights[i] * SI_FARADAY * F_scale
        propagated.append(out + transfer)
    return propagated

def propagate_context_vectorized(outputs, context_weights, F_scale: float = 1e-3, backend_name: Optional[str] = None):
    """
    outputs: (batch, num_neurons)
    context_weights: (batch, num_neurons) or (num_neurons,)
    returns: (batch, num_neurons)
    """
    backend = ensure_backend(backend_name)
    Fc = SI_FARADAY * F_scale

    if backend.name.startswith("torch") and _TORCH_AVAILABLE:
        cw = context_weights if context_weights.shape == outputs.shape else context_weights.view(1, -1).expand_as(outputs)
        return outputs + cw * Fc

    import numpy as np

    outputs_np = np.array(outputs)
    cw_np = np.array(context_weights)
    if cw_np.shape != outputs_np.shape:
        cw_np = cw_np.reshape(1, -1)
        cw_np = np.broadcast_to(cw_np, outputs_np.shape)
    return outputs_np + cw_np * Fc


__all__ = ["propagate_context_scalar", "propagate_context_vectorized"]

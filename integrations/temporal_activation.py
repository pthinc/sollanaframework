# temporal_activation.py
"""Temporal activation with multi-backend (torch preferred, TF/NumPy fallback)."""
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    keras = None  # type: ignore
    _TF_AVAILABLE = False

import numpy as np


def _x_t_torch(t):
    return torch.tanh(torch.exp(t) - math.pi)


def _x_t_tf(t):
    return tf.math.tanh(tf.math.exp(t) - math.pi)


def _x_t_np(t):
    return np.tanh(np.exp(t) - math.pi)


if _TORCH_AVAILABLE:

    class TemporalActivation(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, hidden, t):
            scale = _x_t_torch(t).to(hidden.dtype).view(-1, 1)
            return hidden * scale

elif _TF_AVAILABLE:

    class TemporalActivation(keras.layers.Layer):  # type: ignore
        def call(self, inputs, **kwargs):  # type: ignore
            hidden, t = inputs
            scale = tf.reshape(_x_t_tf(t), (-1, 1))
            return tf.cast(hidden, tf.float32) * tf.cast(scale, tf.float32)

else:

    class TemporalActivation:
        def __call__(self, hidden, t):
            scale = np.reshape(_x_t_np(t), (-1, 1))
            return hidden * scale


__all__ = ["TemporalActivation"]

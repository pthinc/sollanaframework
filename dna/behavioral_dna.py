# behavioral_dna.py
"""Behavioral DNA activation with multi-backend support (torch preferred, TF/NumPy fallback)."""
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
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

    class BehavioralDNA(nn.Module):
        """
        BehavioralDNA hesaplayıcısı (PyTorch)
        Girdi: attention A, probability P, weight W, zaman t (torch.Tensor)
        """

        def __init__(self, h: float = 1.0, k: float = 1.0, F: float = 1.0, learnable: bool = False, eps: float = 1e-9):
            super().__init__()
            self.eps = eps
            if learnable:
                self.h = nn.Parameter(torch.tensor(float(h)))
                self.k = nn.Parameter(torch.tensor(float(k)))
                self.F = nn.Parameter(torch.tensor(float(F)))
            else:
                self.register_buffer("h", torch.tensor(float(h)))
                self.register_buffer("k", torch.tensor(float(k)))
                self.register_buffer("F", torch.tensor(float(F)))

        def forward(self, A: torch.Tensor, P: torch.Tensor, W: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore
            safe_P = torch.clamp(P, min=self.eps, max=1.0)
            term = self.h * A + self.k * torch.log(safe_P) + self.F * W
            return _x_t_torch(t) * term

elif _TF_AVAILABLE:

    class BehavioralDNA(keras.layers.Layer):  # type: ignore
        """TensorFlow/Keras implementation when torch is unavailable."""

        def __init__(self, h: float = 1.0, k: float = 1.0, F: float = 1.0, learnable: bool = False, eps: float = 1e-9, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            if learnable:
                self.h = tf.Variable(float(h), trainable=True, dtype=tf.float32)
                self.k = tf.Variable(float(k), trainable=True, dtype=tf.float32)
                self.F = tf.Variable(float(F), trainable=True, dtype=tf.float32)
            else:
                self.h = tf.constant(float(h), dtype=tf.float32)
                self.k = tf.constant(float(k), dtype=tf.float32)
                self.F = tf.constant(float(F), dtype=tf.float32)

        def call(self, inputs, **kwargs):  # type: ignore
            A, P, W, t = inputs
            safe_P = tf.clip_by_value(P, clip_value_min=self.eps, clip_value_max=1.0)
            term = self.h * A + self.k * tf.math.log(safe_P) + self.F * W
            return _x_t_tf(t) * term

else:

    class BehavioralDNA:
        """Lightweight NumPy fallback for environments without torch/tf.

        This is intended for compatibility/testing; gradients are not supported.
        """

        def __init__(self, h: float = 1.0, k: float = 1.0, F: float = 1.0, learnable: bool = False, eps: float = 1e-9):
            self.eps = eps
            self.h = float(h)
            self.k = float(k)
            self.F = float(F)
            self.learnable = learnable

        def __call__(self, A, P, W, t):
            safe_P = np.clip(P, self.eps, 1.0)
            term = self.h * A + self.k * np.log(safe_P) + self.F * W
            return _x_t_np(t) * term


__all__ = ["BehavioralDNA"]

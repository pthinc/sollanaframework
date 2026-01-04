# rl_reward_shaping.py
"""Reward shaping helper with backend-aware tensor creation."""

from typing import Optional, Tuple

from backends import ensure_backend


def behavior_reward_shaped(env_reward, A, P, W, t, coder, alpha: float = 1.0,
                           clip: Tuple[float, float] = (-1.0, 1.0), backend_name: Optional[str] = None):
    backend = ensure_backend(backend_name)

    if backend.name.startswith("torch"):
        import torch

        D = coder(torch.tensor(A), torch.tensor(P), torch.tensor(W), torch.tensor(t)).item()
    elif backend.name.startswith("tensorflow"):
        import tensorflow as tf  # type: ignore

        D = float(coder(tf.convert_to_tensor(A), tf.convert_to_tensor(P), tf.convert_to_tensor(W), tf.convert_to_tensor(t)))
    else:
        import numpy as np

        D = float(coder(np.array(A), np.array(P), np.array(W), np.array(t)))

    shaped = env_reward + alpha * D
    return max(min(shaped, clip[1]), clip[0])

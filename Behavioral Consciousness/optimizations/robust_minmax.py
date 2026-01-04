"""
Adversarial robustness helpers (Cicikus Robust abstraction).
- Evasion objective wrapper
- Poisoning placeholder
- LoRA-regularized loss helper
"""

from typing import Callable, Dict, Any
import numpy as np

EPS = 1e-12


def evasion_objective(model_fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 0.1, norm_p: int = 2) -> Dict[str, Any]:
    """
    model_fn: returns detector score (higher safer)
    x: input vector
    returns adversarial delta and target score direction (minimize score)
    """
    x = np.asarray(x, dtype=float)
    delta = np.zeros_like(x)
    grad = np.sign(x) if norm_p == 1 else x / (np.linalg.norm(x) + EPS)
    delta = -eps * grad
    adv_x = x + delta
    score = float(model_fn(adv_x)) if callable(model_fn) else 0.0
    return {"adv_x": adv_x, "delta": delta, "score": score}


def poisoning_objective(train_set: np.ndarray, k: int = 1) -> Dict[str, Any]:
    """Placeholder: select k points to flip/perturb."""
    if train_set.shape[0] == 0:
        return {"indices": [], "delta": None}
    idx = list(range(min(k, train_set.shape[0])))
    return {"indices": idx, "delta": None}


def lora_regularized_loss(base_loss: float, A_norm: float, B_norm: float, lam: float = 1e-3) -> float:
    return float(base_loss + lam * (A_norm + B_norm))

# feature_weighting.py
"""Feature weighting utility (NumPy only)."""

import numpy as np
from typing import Tuple


def apply_behavioral_weights(X: np.ndarray, context_weights: np.ndarray) -> np.ndarray:
    return X * context_weights.reshape(1, -1)


__all__ = ["apply_behavioral_weights"]

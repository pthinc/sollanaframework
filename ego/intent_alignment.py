# intent_alignment.py
"""Intent alignment using NumPy; backend-agnostic."""

import numpy as np
from typing import Tuple


def compute_alignment(intent_vecs, phi_matrix, eta_vec) -> Tuple[float, ...]:
    # intent_vecs shape (K,d), phi_matrix shape (N,d)
    proj = np.dot(phi_matrix, intent_vecs.T)  # (N,K)
    weighted = (proj * eta_vec.reshape(-1, 1)).sum(axis=0)
    return tuple(weighted / (np.linalg.norm(intent_vecs, axis=1) + 1e-9))


__all__ = ["compute_alignment"]

"""
Taste / companionship manifold utilities (secretkesifler-inspired).
- Taste vector T = [n, d, e, o, h]
- Decay suppression on provoke (delta * C vector)
- Similarity and freedom/divergence scoring
"""

from typing import List, Dict, Any
import numpy as np

EPS = 1e-12


def taste_vector(n: float, d: float, e: float, o: float, h: float) -> np.ndarray:
    T = np.array([n, d, e, o, h], dtype=float)
    T = np.clip(T, 0.0, 1.0)
    return T


def apply_provoke(T: np.ndarray, delta: float, C: np.ndarray = None) -> np.ndarray:
    C_vec = C if C is not None else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return np.clip(T + delta * C_vec, 0.0, 1.0)


def taste_similarity(T_a: np.ndarray, T_b: np.ndarray) -> float:
    na = float(np.linalg.norm(T_a) + EPS)
    nb = float(np.linalg.norm(T_b) + EPS)
    return float(np.dot(T_a, T_b) / (na * nb))


def manifold_divergence(T_matrix: np.ndarray) -> float:
    if T_matrix.shape[0] < 2:
        return 0.0
    cov = np.cov(T_matrix, rowvar=False)
    return float(np.trace(cov))


def freedom_mask(T_matrix: np.ndarray, trace_thresh: float = 0.02, sim_thresh: float = 0.2) -> Dict[int, bool]:
    if T_matrix.shape[0] == 0:
        return {}
    sims = np.dot(T_matrix, T_matrix.T) / ((np.linalg.norm(T_matrix, axis=1)[:,None]+EPS)*(np.linalg.norm(T_matrix, axis=1)[None,:]+EPS))
    cov_trace = manifold_divergence(T_matrix)
    result = {}
    for i in range(T_matrix.shape[0]):
        mean_sim = float(np.mean(sims[i]))
        result[i] = bool(cov_trace > trace_thresh and mean_sim > sim_thresh)
    return result


def to_dict(T: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(["n","d","e","o","h"], T.tolist())}

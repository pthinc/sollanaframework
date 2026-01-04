"""
Lightweight IIT/GWT helpers for BCE system.
Heuristic, fast metrics (not full IIT/GWT computations).
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

EPS = 1e-12


def _safe_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= EPS:
        return np.full_like(x, 1.0 / max(1, x.size))
    return x / s


def iit_integrated_information(prob_matrix: np.ndarray) -> Dict[str, Any]:
    """Simple phi surrogate: joint entropy minus mean part entropy."""
    probs = _safe_probs(prob_matrix.ravel())
    H_joint = -np.sum(probs * np.log2(np.clip(probs, EPS, 1.0)))
    parts = np.array_split(probs, max(1, probs.size // 2))
    H_parts = np.mean([
        -np.sum(p * np.log2(np.clip(p, EPS, 1.0))) if p.size > 0 else 0.0
        for p in parts
    ])
    phi = max(0.0, H_joint - H_parts)
    return {"phi": float(phi), "H_joint": float(H_joint), "H_parts": float(H_parts)}


def gwt_broadcast_score(activations: np.ndarray, top_k: int = 4) -> Dict[str, Any]:
    """Workspace ignition proxy from activations."""
    a = np.asarray(activations, dtype=float).ravel()
    if a.size == 0:
        return {"broadcast": 0.0, "ignition": 0.0, "sparsity": 0.0}
    norm = np.linalg.norm(a) + EPS
    a_norm = a / norm
    k = min(top_k, a_norm.size)
    top_vals = np.partition(a_norm, -k)[-k:]
    broadcast = float(top_vals.mean())
    sparsity = float(np.mean(a_norm == 0.0))
    ignition = float(np.mean(a_norm > np.percentile(a_norm, 75)))
    return {
        "broadcast": broadcast,
        "ignition": ignition,
        "sparsity": sparsity,
        "topk_mean": broadcast,
    }


__all__ = ["iit_integrated_information", "gwt_broadcast_score"]

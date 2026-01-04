"""
Classic Robust Coder: deterministic hashing + clamp/clip for safety; simpler than ART variant.
"""
from typing import Dict, Any
import hashlib
import numpy as np

EPS = 1e-12


def encode(x: np.ndarray, clip: float = 1.0) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    clipped = np.clip(x, -clip, clip)
    h = hashlib.sha256(clipped.tobytes()).hexdigest()
    return {"clipped": clipped, "hash": h}


def similarity_hash(h1: str, h2: str) -> float:
    # simple Hamming-like score on hex
    same = sum(c1 == c2 for c1, c2 in zip(h1, h2))
    return float(same) / max(1, len(h1))


def robust_compare(x: np.ndarray, y: np.ndarray, clip: float = 1.0) -> Dict[str, Any]:
    ex = encode(x, clip)
    ey = encode(y, clip)
    sim = similarity_hash(ex["hash"], ey["hash"])
    return {"sim": sim, "x_hash": ex["hash"], "y_hash": ey["hash"]}


def example_usage():
    a = np.array([0.1, 0.2])
    b = np.array([0.1, 0.19])
    return robust_compare(a, b)


if __name__ == "__main__":
    print(example_usage())

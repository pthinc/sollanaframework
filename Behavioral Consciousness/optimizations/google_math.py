"""
GoogleMaths-inspired heuristic: rank items by score and damping (PageRank-lite) with diversity boost.
"""
from typing import List, Tuple
import numpy as np

EPS = 1e-12


def rank_with_damping(scores: List[float], damping: float = 0.85) -> List[float]:
    n = len(scores)
    if n == 0:
        return []
    s = np.array(scores, dtype=float)
    s_range = float(s.max() - s.min()) if s.size else 0.0
    s = (s - s.min()) / (s_range + EPS)
    base = np.full(n, (1.0 - damping) / n)
    pr = np.copy(base)
    for _ in range(10):
        pr = base + damping * s / (s.sum() + EPS)
    return pr.tolist()


def diversify(scores: List[float], embeddings: List[List[float]], alpha: float = 0.2) -> List[float]:
    if not embeddings:
        return scores
    X = np.array(embeddings, dtype=float)
    sims = np.matmul(X, X.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(X, axis=1, keepdims=True).T + EPS)
    penalty = sims.mean(axis=1)
    return [float(max(0.0, s - alpha * p)) for s, p in zip(scores, penalty)]


def google_math_rank(items: List[Tuple[str, float]], embeddings: List[List[float]]) -> List[Tuple[str, float]]:
    ids, sc = zip(*items) if items else ([], [])
    damped = rank_with_damping(list(sc))
    div = diversify(damped, embeddings)
    ranked = list(zip(ids, div))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def example_usage():
    items = [("a",0.2),("b",0.5),("c",0.9)]
    emb = [[1,0],[0.1,0.9],[0.9,0.1]]
    return google_math_rank(items, emb)


if __name__ == "__main__":
    print(example_usage())

"""
Bayes math utilities inspired by bayesmath.md: simple prior/likelihood updates, evidence fusion, and surprise.
"""
from typing import Dict, List
import math

EPS = 1e-12


def bayes_update(prior: float, likelihood: float, evidence: float = 1.0) -> float:
    denom = evidence if evidence != 0 else (prior * likelihood + EPS)
    return float((prior * likelihood) / (denom + EPS))


def normalize_probs(ps: List[float]) -> List[float]:
    s = sum(ps) + EPS
    return [float(p / s) for p in ps]


def fuse_evidences(priors: List[float], likelihoods: List[float]) -> List[float]:
    fused = [p * l for p, l in zip(priors, likelihoods)]
    return normalize_probs(fused)


def surprise(p: float) -> float:
    p = min(max(p, EPS), 1.0)
    return float(-math.log(p))


def kl_div(p: List[float], q: List[float]) -> float:
    p = normalize_probs(p)
    q = normalize_probs(q)
    return float(sum(pi * math.log((pi + EPS) / (qi + EPS)) for pi, qi in zip(p, q)))


def example_usage() -> Dict[str, float]:
    prior = 0.6; lik = 0.7
    post = bayes_update(prior, lik)
    fused = fuse_evidences([0.6,0.4],[0.7,0.5])
    return {"post": post, "fused0": fused[0], "surprise": surprise(post)}


if __name__ == "__main__":
    print(example_usage())

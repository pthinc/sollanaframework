# bce_selection.py
import math
import bisect
import random
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def compute_weights(mu_task: float, mu_emotion: float, mu_engage: float, eps: float = 1e-8) -> Dict[str, float]:
    w_det = (mu_task * 0.6 + mu_engage * 0.4) * (1.0 - mu_emotion * 0.5)
    w_stoch = (mu_emotion * 0.7 + (1.0 - mu_task) * 0.3) * (1.0 - mu_engage * 0.4)
    total = w_det + w_stoch + eps
    return {"w_det": float(w_det / total), "w_stoch": float(w_stoch / total)}

class DeterministicSelector:
    """
    Deterministic selection using sorted score list and binary search thresholding.
    Candidates must be list of dicts with 'behavior_id' and 'score'.
    """
    def __init__(self, min_score: float = 0.0):
        self.min_score = float(min_score)

    def select(self, candidates: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        sorted_c = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        lo, hi = 0, len(sorted_c)
        # binary search for first item below min_score to skip low-quality group
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_c[mid].get("score", 0.0) >= self.min_score:
                lo = mid + 1
            else:
                hi = mid
        cutoff = lo
        selected = sorted_c[:max(1, min(top_k, cutoff))]
        return selected

class BlumBlumShub:
    """
    Simple Blum Blum Shub generator for reproducible strong randomness.
    Use small primes only for demo. For production choose large safe primes.
    """
    def __init__(self, p: int, q: int, seed: Optional[int] = None):
        self.n = p * q
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(os.urandom(32)).digest()[:8], "big")
        self.state = seed % self.n

    def next_bit(self) -> int:
        self.state = pow(self.state, 2, self.n)
        return self.state & 1

    def rand_uniform(self) -> float:
        # accumulate 32 bits
        val = 0
        for _ in range(32):
            val = (val << 1) | self.next_bit()
        return val / float(2**32 - 1)

class WeightedSampler:
    """
    Weighted sampler using prefix sums. Accepts deterministic RNG or BBS wrapper.
    """
    def __init__(self, weights: List[float]):
        self.weights = np.array(weights, dtype=float)
        total = float(self.weights.sum())
        if total <= 0:
            self.prefix = np.cumsum(np.ones_like(self.weights))
            self.total = float(self.prefix[-1])
        else:
            self.prefix = np.cumsum(self.weights)
            self.total = float(self.prefix[-1])

    def sample_index(self, rng_uniform: Optional[callable] = None) -> int:
        rnd = rng_uniform() if rng_uniform is not None else random.random
        r = rnd() * self.total
        idx = bisect.bisect_left(self.prefix, r)
        return int(min(idx, len(self.weights) - 1))

class StochasticSelector:
    """
    Stochastic selection builds a weighted sampler from candidate novelty or diversity scores
    and samples using Blum Blum Shub for reproducibility.
    """
    def __init__(self, p: int = 383, q: int = 503, seed: Optional[int] = None):
        self.bbs = BlumBlumShub(p, q, seed)

    def select(self, candidates: List[Dict[str, Any]], k: int = 1) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        weights = [max(0.0, c.get("novelty", 0.01) + 1e-6) for c in candidates]
        sampler = WeightedSampler(weights)
        def rngf():
            return self.bbs.rand_uniform
        picked = []
        used = set()
        attempts = 0
        while len(picked) < k and attempts < len(candidates) * 5:
            idx = sampler.sample_index(rng_uniform=rngf)
            attempts += 1
            if idx in used:
                continue
            used.add(idx)
            picked.append(candidates[idx])
        return picked

class HybridSelector:
    """
    Combine deterministic and stochastic outputs using fuzzy weights.
    Final scoring mixes Phi_det and Phi_stoch values if available.
    """
    def __init__(self, det_selector: DeterministicSelector, stoch_selector: StochasticSelector):
        self.det = det_selector
        self.stoch = stoch_selector

    def select(self, candidates: List[Dict[str, Any]], mu_task: float, mu_emotion: float, mu_engage: float, k: int = 1) -> List[Dict[str, Any]]:
        weights = compute_weights(mu_task, mu_emotion, mu_engage)
        w_det, w_stoch = weights["w_det"], weights["w_stoch"]
        det = self.det.select(candidates, top_k=max(1, k))
        stoch = self.stoch.select(candidates, k)
        combined = {}
        for c in candidates:
            bid = c.get("behavior_id")
            phi_det = c.get("phi_det", c.get("score", 0.0))
            phi_stoch = c.get("phi_stoch", c.get("novelty", 0.0))
            combined_score = w_det * float(phi_det) + w_stoch * float(phi_stoch)
            combined[bid] = {**c, "combined_score": combined_score}
        sorted_comb = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        return sorted_comb[:k]

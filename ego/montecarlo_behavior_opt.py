# montecarlo_behavior_opt.py
import time, math, random, numpy as np
from typing import List, Dict, Callable, Optional

class MonteCarloOptimizer:
    def __init__(self,
                 kappa_fn: Callable[[float], float],
                 omega_fn: Callable[[float], float],
                 gravity: float = 1.0,
                 seed: Optional[int] = None):
        self.kappa_fn = kappa_fn
        self.omega_fn = omega_fn
        self.G = float(gravity)
        self.rng = random.Random(seed)

    def _eta(self, phi_vec: np.ndarray, context_vec: Optional[np.ndarray]) -> float:
        if context_vec is None:
            return float(np.linalg.norm(phi_vec))
        num = float(np.dot(phi_vec, context_vec))
        den = float((np.linalg.norm(phi_vec) * np.linalg.norm(context_vec)) + 1e-12)
        return max(0.0, min(1.0, num/den))

    def _entropy(self, phi_vec: np.ndarray) -> float:
        p = np.asarray(phi_vec, dtype=float)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p)/len(p)
        else:
            p = p / (s + 1e-12)
        ent = -float((p * np.log(p + 1e-12)).sum())
        return ent

    def score_candidate(self,
                        cand: Dict,
                        t: float,
                        context_vec: Optional[np.ndarray],
                        snapshot_count: float) -> float:
        phi_vec = np.asarray(cand.get("phi_vec", np.zeros(1)), dtype=float)
        phi_scalar = float(cand.get("phi", 0.0))
        E = float(cand.get("ethical", 1.0))
        eta = self._eta(phi_vec, context_vec)
        kappa = float(self.kappa_fn(t))
        omega = float(self.omega_fn(t))
        entropy = float(self._entropy(phi_vec))
        xi = float(self.rng.gauss(0.0, 0.01))
        score = kappa * eta * E * phi_scalar - omega * self.G * float(snapshot_count) + xi * entropy
        return float(score)

    def optimize(self,
                 candidates: List[Dict],
                 now: Optional[float] = None,
                 context_vec: Optional[np.ndarray] = None,
                 snapshot_count: float = 1.0,
                 n_iters: int = 256,
                 ethical_check: Optional[Callable[[Dict], bool]] = None,
                 anomaly_check: Optional[Callable[[Dict], float]] = None,
                 anom_threshold: float = 0.6) -> Dict:
        now = now or time.time()
        best = None
        best_score = -1e9
        for _ in range(n_iters):
            cand = self.rng.choice(candidates)
            if ethical_check and not ethical_check(cand):
                continue
            if anomaly_check:
                anom = float(anomaly_check(cand))
                if anom > anom_threshold:
                    continue
            s = self.score_candidate(cand, now, context_vec, snapshot_count)
            # local perturbation exploration
            if self.rng.random() < 0.2:
                s += self.rng.gauss(0, 0.02) * self._entropy(np.asarray(cand.get("phi_vec", [1.0])))
            if s > best_score:
                best_score = s
                best = {**cand, "mc_score": s, "ts": now}
        return best or {}

# örnek kappa ve omega fonksiyonları
def kappa_linear_increase(t): return 0.5 + 0.0001 * (t % 3600)
def omega_decay(t): return max(0.0, 0.5 - 0.00005 * (t % 3600))

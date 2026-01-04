# salinum_flavor_opt.py
import math, time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

PHI = 1.6180339887498948
EPS = 1e-12

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    return float(max(0.0, min(1.0, (np.dot(a, b) / (na * nb) + 1.0) / 2.0)))  # map [-1,1] -> [0,1]

def amplitude(psi_vec: np.ndarray, norm_vec: np.ndarray, n: float = 1.0) -> float:
    sim = cos_sim(psi_vec, norm_vec)
    return float((PHI ** float(n)) * sim)

def frequency(R_ij: float) -> float:
    return float(math.pi * PHI * max(0.0, min(1.0, 1.0 - float(R_ij))))

def phase(norm_gaps: List[float]) -> float:
    return float(sum(norm_gaps))

def balance(approval: float, critical: float, flavor: float, approval_weight: float = 1.0) -> float:
    # approval, critical, flavor in [0,1]; approval_weight controls emphasis on approval vs flavor
    a_term = approval_weight * float(approval) * (1.0 - float(critical))
    f_term = (1.0 - approval_weight) * float(flavor)
    return float(max(0.0, a_term + f_term))

def salinum_t(t: float,
              psi_vec: np.ndarray,
              norm_vec: np.ndarray,
              n: float,
              R_ij: float,
              norm_gaps: List[float],
              approval: float,
              critical: float,
              flavor: float,
              approval_weight: float = 1.0) -> float:
    A = amplitude(psi_vec, norm_vec, n)
    B = frequency(R_ij)
    C = phase(norm_gaps)
    D = balance(approval, critical, flavor, approval_weight)
    return float(A * math.sin(B * float(t) + C) + D)

# Simple grid optimizer for n and approval_weight maximizing a target metric
def optimize_params(history: List[Dict[str, Any]],
                    candidate_actions: List[Dict[str, Any]],
                    n_grid: List[float] = None,
                    aw_grid: List[float] = None,
                    metric_fn: Optional[callable] = None) -> Dict:
    """
    history: recent samples (unused for simplistic optimizer but available)
    candidate_actions: list of {"psi_vec","norm_vec","R_ij","norm_gaps","approval","critical","flavor"}
    metric_fn: function(list of (action,score))->float higher is better (defaults to sum of top-3)
    Returns best {n, approval_weight, rankings}
    """
    n_grid = n_grid or [0.5, 1.0, 1.5, 2.0]
    aw_grid = aw_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
    if metric_fn is None:
        def metric_fn(ranked):  # ranked: list of scores
            return sum(sorted(ranked, reverse=True)[:3])
    best = {"score": -1e9}
    for n in n_grid:
        for aw in aw_grid:
            scores = []
            for c in candidate_actions:
                s = salinum_t(t=time.time(), psi_vec=np.array(c["psi_vec"]), norm_vec=np.array(c["norm_vec"]),
                              n=n, R_ij=c.get("R_ij",0.5), norm_gaps=c.get("norm_gaps",[]),
                              approval=c.get("approval",0.5), critical=c.get("critical",0.0), flavor=c.get("flavor",0.0),
                              approval_weight=aw)
                scores.append(s)
            val = metric_fn(scores)
            if val > best["score"]:
                best = {"n": n, "approval_weight": aw, "score": val, "scores": scores}
    return best

# Example helper: produce time series for single action
def timeseries_for_action(action: Dict[str,Any], n: float, approval_weight: float, t0: float = 0.0, steps:int=100, dt:float=1.0):
    ts = [t0 + i*dt for i in range(steps)]
    vals = [salinum_t(t, np.array(action["psi_vec"]), np.array(action["norm_vec"]), n,
                       action.get("R_ij",0.5), action.get("norm_gaps",[]),
                       action.get("approval",0.5), action.get("critical",0.0), action.get("flavor",0.0),
                       approval_weight) for t in ts]
    return ts, vals

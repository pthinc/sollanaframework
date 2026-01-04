# salinim_module.py
import time, math
import numpy as np
from typing import List, Dict, Tuple, Optional

EPS = 1e-12

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    return float(np.dot(a, b) / (na * nb))

def amplitude(psi_vec: np.ndarray, norm_vec: np.ndarray) -> float:
    return float(max(0.0, min(1.0, (cos_sim(psi_vec, norm_vec) + 1.0) / 2.0)))

def alpha_decay(decay_t: float, recovery_t: float) -> float:
    return float(-decay_t + recovery_t)

def freq_from_intervals(interval_seconds: float) -> float:
    if interval_seconds <= 0:
        return 0.0
    return 1.0 / float(interval_seconds)

def omega_from_freq(f: float) -> float:
    return 2.0 * math.pi * float(f)

def phase_from_norm_mismatch(norm_mismatch: float, scale: float = 1.0) -> float:
    return float(np.clip(norm_mismatch, 0.0, 1.0) * scale * math.pi)

def salinim_at_t(t: float,
                 psi_vec: np.ndarray,
                 norm_vec: np.ndarray,
                 decay_t: float,
                 recovery_t: float,
                 interval_seconds: float,
                 norm_mismatch: float) -> float:
    A = amplitude(psi_vec, norm_vec)
    alpha = alpha_decay(decay_t, recovery_t)
    f = freq_from_intervals(interval_seconds)
    omega = omega_from_freq(f)
    phi = phase_from_norm_mismatch(norm_mismatch)
    expo = math.exp(alpha * t)  # alpha can be negative
    return float(A * expo * math.sin(omega * t + phi))

# parameter estimation on windowed samples
def estimate_parameters(history: List[Dict]) -> Dict:
    """
    history elements: {"ts":float, "psi_vec":array, "norm_vec":array, "decay":float, "recovery":float}
    returns averages and inferred interval_seconds median
    """
    if not history:
        return {"A":0.0, "alpha":0.0, "interval":0.0, "freq":0.0, "phi":0.0}
    # amplitude average
    As = []
    alphas = []
    times = []
    norm_mismatches = []
    for h in history:
        psi = np.asarray(h["psi_vec"], dtype=float)
        norm = np.asarray(h["norm_vec"], dtype=float)
        As.append(amplitude(psi, norm))
        alphas.append(alpha_decay(h.get("decay",0.0), h.get("recovery",0.0)))
        times.append(h["ts"])
        nm = float(h.get("norm_mismatch", 0.0))
        norm_mismatches.append(nm)
    # estimate typical interval as median difference
    times = sorted(times)
    intervals = [times[i+1]-times[i] for i in range(len(times)-1)] if len(times)>1 else [60.0]
    median_interval = float(np.median(intervals)) if intervals else 60.0
    freq = freq_from_intervals(median_interval)
    phi = float(np.median(norm_mismatches)) if norm_mismatches else 0.0
    return {"A": float(np.mean(As)), "alpha": float(np.mean(alphas)), "interval": median_interval, "freq": freq, "phi": phi}

# simple grid optimizer for alpha scaling and interval nudges
def grid_optimize_sug(candidates: List[Dict], alpha_grid: List[float], beta_grid: List[float]) -> Dict:
    """
    candidates: list of {"action":str, "resonance":float, "context_consistency":float}
    objective SugScore = alpha * R + beta * C
    returns best config and ranked suggestions
    """
    best = {"score": -1.0}
    for a in alpha_grid:
        for b in beta_grid:
            if a + b <= 0:
                continue
            aa, bb = a/(a+b), b/(a+b)
            scores = [(c, aa*c.get("resonance",0.0) + bb*c.get("context_consistency",0.0)) for c in candidates]
            scores.sort(key=lambda x: x[1], reverse=True)
            total = sum(s for (_,s) in scores[:min(5,len(scores))])  # top-k aggregate
            if total > best["score"]:
                best = {"alpha": aa, "beta": bb, "score": total, "ranked": scores}
    return best

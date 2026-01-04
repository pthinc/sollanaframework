# salinim_core.py
import time, math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from optimizations.decay_reconstruction import policy_step
    DECAY_POL_AVAILABLE = True
except Exception:
    policy_step = None
    DECAY_POL_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except Exception:
    DBSCAN = None
    SKLEARN_AVAILABLE = False

EPS = 1e-12
PI = math.pi

# ---- Aktivasyon fonksiyonları ----
def activation_core(t: float, alpha: float = 1.0, char_const: float = PI) -> float:
    """x(t) = tanh(e^{alpha*t} - char_const)"""
    val = math.exp(alpha * t)
    return math.tanh(val - char_const)

def free_oscillation_test(t: float, alpha: float = 1.0) -> float:
    """x_free(t) = tanh(e^{alpha*t})"""
    return math.tanh(math.exp(alpha * t))

# ---- Self vector ve süreklilik ----
def compute_self_vector(samples: List[Dict[str, float]]) -> float:
    """
    samples: list of {"ts":float, "C":float, "D":float, "N":float, "R":float}
    returns scalar S(t) = sum C*(1-D)*N*R over samples
    """
    total = 0.0
    for s in samples:
        C = float(s.get("C", 0.0))
        D = float(s.get("D", 0.0))
        N = float(s.get("N", 0.0))
        R = float(s.get("R", 0.0))
        total += C * (1.0 - D) * N * R
    return float(total)

def compute_self_continuity(snapshot: List[Dict[str, Any]]) -> float:
    """
    snapshot: list of {"ts":, "S":} sorted by ts
    SC(t) ~= integral of gradient of S = cumulative sum of dS * dt
    returns scalar continuity measure (higher => more continuous)
    """
    if len(snapshot) < 2:
        return 0.0
    cont = 0.0
    for i in range(1, len(snapshot)):
        dS = snapshot[i]["S"] - snapshot[i-1]["S"]
        dt = max(1e-6, snapshot[i]["ts"] - snapshot[i-1]["ts"])
        cont += dS * dt
    return float(cont)

def compute_self_drift(S_now: float, S_prev: float) -> float:
    return float(abs(S_now - S_prev))


# ---- Decay/Reconstruction orchestration (optional) ----
def apply_decay_policy(state: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
    if not DECAY_POL_AVAILABLE:
        return {"score": state.get("score", 1.0), "reason": "policy_unavailable"}
    try:
        return policy_step(state, signals)
    except Exception as e:
        return {"score": state.get("score", 1.0), "error": str(e)}

# ---- Kısmi bilinç kümeleme (örn. zaman-temelli) ----
def partial_conscious_clustering(values: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> List[int]:
    """
    values: shape (N, d) embeddings or feature vectors of x(t) samples
    returns cluster labels array (len N)
    """
    if values.shape[0] == 0 or not SKLEARN_AVAILABLE or DBSCAN is None:
        return []
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = model.fit_predict(values)
    return labels.tolist()

# ---- Örnek simülasyon ----
def simulate_trace(num_points: int = 50, alpha: float = 1.0):
    ts0 = time.time()
    samples = []
    for i in range(num_points):
        t = i / max(1.0, num_points-1)
        ts = ts0 + i
        x = activation_core(t, alpha=alpha)
        # synthetic components for S vector
        C = 0.5 + 0.5 * x
        D = max(0.0, 0.1 * (1.0 - x))
        N = 0.8
        R = max(0.0, 0.5 + 0.5 * x)
        samples.append({"ts": ts, "C": C, "D": D, "N": N, "R": R, "x": x})
    return samples

# oksipital_reflex.py
import time, math
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

EPS = 1e-12

# -------------------------
# Hook placeholders - replace in integration
# -------------------------
def snapshot_hook(reason: str) -> str:
    return f"snapshot_{int(time.time()*1000)}"

def quarantine_hook(payload: Dict[str,Any]) -> None:
    print("QUARANTINE:", payload)

def emotional_shield_hook(payload: Dict[str,Any]) -> None:
    print("EMOTIONAL SHIELD:", payload)

def decay_suppression_hook(payload: Dict[str,Any]) -> None:
    print("DECAY SUPPRESSION:", payload)

def flavor_suggestion_hook(payload: Dict[str,Any]) -> None:
    print("FLAVOR SUGGESTION:", payload)

def telemetry_hook(event: Dict[str,Any]) -> None:
    print("TELEMETRY:", event)

# -------------------------
# Fuzzy C Means implementation
# -------------------------
def _dist_matrix(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # squared euclidean distances shape (N, K)
    # X: (N, d), C: (K, d)
    XX = np.sum(X*X, axis=1, keepdims=True)  # (N,1)
    CC = np.sum(C*C, axis=1, keepdims=True).T  # (1,K)
    XC = X.dot(C.T)  # (N,K)
    D2 = np.maximum(0.0, XX + CC - 2*XC)
    D = np.sqrt(D2 + EPS)
    return D

def fuzzy_c_means(X: np.ndarray,
                  K: int = 3,
                  m: float = 2.0,
                  max_iter: int = 150,
                  tol: float = 1e-5,
                  init_centers: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns memberships U (N,K) and centers C (K,d)
    """
    N, d = X.shape
    if init_centers is None:
        # kmeans++ style init: random samples
        idx = np.random.choice(N, K, replace=False)
        C = X[idx].astype(float)
    else:
        C = init_centers.copy().astype(float)
    U = np.zeros((N,K), dtype=float)
    for it in range(max_iter):
        D = _dist_matrix(X, C)  # (N,K)
        # avoid divide by zero: if any distance zero, set membership 1 for that center
        zero_mask = D < 1e-8
        if np.any(zero_mask):
            U = np.zeros_like(D)
            U[zero_mask] = 1.0
        else:
            expo = 2.0 / (m - 1.0)
            # compute U using formula
            denom = (D[:, :, None] / D[:, None, :]) ** expo  # (N,K,K)
            denom_sum = np.sum(denom, axis=2)  # (N,K)
            U = 1.0 / np.maximum(EPS, denom_sum)
        U_m = U ** m
        C_new = (U_m.T @ X) / (np.sum(U_m.T, axis=1, keepdims=True) + EPS)  # (K,d)
        shift = np.linalg.norm(C_new - C)
        C = C_new
        if shift < tol:
            break
    return U, C

# -------------------------
# Oksipital Reflex Engine
# -------------------------
class OksipitalReflexEngine:
    def __init__(self,
                 K: int = 3,
                 m: float = 2.0,
                 cluster_weights: Optional[List[float]] = None,
                 high_thresh: float = 0.8,
                 mid_thresh: float = 0.4,
                 low_thresh: float = 0.2):
        self.K = int(K); self.m = float(m)
        self.cluster_weights = np.array(cluster_weights or [0.2, 0.5, 1.0], dtype=float)  # e.g. [light, mid, critical]
        self.high_thresh = float(high_thresh)
        self.mid_thresh = float(mid_thresh)
        self.low_thresh = float(low_thresh)

    def evaluate(self,
                 B: List[np.ndarray],
                 context: Dict[str,Any] = None
                ) -> Dict[str,Any]:
        """
        B: list of bozulma vektörleri (N, d)
        returns decision dict and actions performed
        """
        context = context or {}
        X = np.vstack(B).astype(float)  # (N,d)
        N, d = X.shape
        # fuzzy clustering
        U, C = fuzzy_c_means(X, K=self.K, m=self.m)
        # compute reflex score per item and global reflex score
        # for each item i: Rf_i = sum_j u_ij * w_j
        # global Rf = max_i Rf_i  (or mean)
        Rf_items = (U * self.cluster_weights[None, :]).sum(axis=1)  # (N,)
        Rf_global = float(np.max(Rf_items))
        # determine severity bucket
        actions = []
        telemetry = {"Rf_global": Rf_global, "Rf_items_sample": Rf_items.tolist(), "K": self.K}
        # decision logic
        if Rf_global >= self.high_thresh:
            # critical: emotional shield + snapshot + quarantine + human review flag
            snap = snapshot_hook("oksipital_critical")
            quarantine_hook({"reason":"critical_disruption","context":context,"score":Rf_global})
            emotional_shield_hook({"reason":"critical_disruption","impact":Rf_global})
            actions.append("emotional_shield")
            actions.append("snapshot")
            actions.append("quarantine")
            telemetry["action"] = "critical"
        elif Rf_global >= self.mid_thresh:
            # medium: increase decay suppression, stronger monitoring
            decay_suppression_hook({"reason":"medium_disruption","impact":Rf_global})
            actions.append("decay_suppression")
            telemetry["action"] = "medium"
        elif Rf_global >= self.low_thresh:
            # light: flavor suggestions and gentle nudges
            flavor_suggestion_hook({"reason":"light_disruption","impact":Rf_global})
            actions.append("flavor_suggestion")
            telemetry["action"] = "light"
        else:
            telemetry["action"] = "none"
        telemetry_hook(telemetry)
        return {"Rf_global": Rf_global, "Rf_items": Rf_items.tolist(), "centers": C.tolist(), "actions": actions, "memberships_shape": U.shape}

# -------------------------
# Demo usage
# -------------------------
if __name__ == "__main__":
    # create synthetic bozulma vectors for demo
    # dimension corresponds to features like context_jump, tone_shift, repetition, urgency, topic_inconsistency
    np.random.seed(0)
    B_low = np.random.normal(loc=0.05, scale=0.02, size=(5,6))
    B_mid = np.random.normal(loc=0.3, scale=0.05, size=(4,6))
    B_high = np.random.normal(loc=0.8, scale=0.08, size=(3,6))
    B_all = np.vstack([B_low, B_mid, B_high])
    engine = OksipitalReflexEngine()
    res = engine.evaluate([b for b in B_all], context={"session":"demo"})
    print("Result:", res["Rf_global"], "Actions:", res["actions"])

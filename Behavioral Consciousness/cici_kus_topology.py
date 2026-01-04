# cici_kus_topology.py
# Requires: numpy
import time, math
import numpy as np
from typing import Dict, List, Tuple

EPS = 1e-12

# --- Helpers ---
def clip01(x): return float(max(0.0, min(1.0, x)))

def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v) + EPS
    return v / norm

# --- Tat vector factory and cici tampon ---
def make_tat(n=0.5,d=0.2,e=0.3,o=0.1,h=0.1) -> np.ndarray:
    v = np.array([n,d,e,o,h], dtype=float)
    v = np.clip(v, 0.0, 1.0)
    return v

# cici tampon vector (mizah, sabır, yönlendirme) mapped into Tat dims
CICI_TAMPON = np.array([0.4, 0.1, 0.3, 0.2, 0.0], dtype=float)  # example

# --- Update rules ---
def decay_update(T: np.ndarray, tau_decay: float=0.01, noise_scale: float=0.0) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    T = T * (1.0 - tau_decay)
    if noise_scale > 0:
        T += np.random.normal(scale=noise_scale, size=T.shape)
    return np.clip(T, 0.0, 1.0)

def provoke_update(T: np.ndarray, delta: float, cici=CICI_TAMPON) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Tp = T + delta * np.asarray(cici, dtype=float)
    return np.clip(Tp, 0.0, 1.0)

def ema_update(T_old: np.ndarray, T_new: np.ndarray, alpha: float=0.15) -> np.ndarray:
    return np.clip((1-alpha)*T_old + alpha*T_new, 0.0, 1.0)

# --- Salınım ---
def S_of_t(T: np.ndarray, omega: float, phi: float, t: float) -> float:
    # component-wise salınım summed into scalar echo
    return float(np.dot(T, np.sin(omega * t + phi)))

# --- Similarity and adjacency ---
def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + EPS
    return (X @ X.T) / (norms @ norms.T)

# --- Local covariance trace as divergence proxy ---
def local_cov_trace(i: int, X: np.ndarray, neighbors: List[int]) -> float:
    if len(neighbors) <= 1: return 0.0
    local = X[neighbors]  # k x d
    cov = np.cov(local, rowvar=False)  # d x d
    # return trace (sum variances) as spread measure
    return float(np.trace(cov))

# --- Freedom detection ---
def detect_freedom(i: int, X: np.ndarray, k_neighbors: int=8, sim_threshold: float=0.2, trace_thresh: float=0.05) -> Tuple[bool, float]:
    sims = cosine_similarity_matrix(X)[i]
    # choose top k neighbors excluding self
    idxs = np.argsort(-sims)
    neighbors = [j for j in idxs if j != i][:k_neighbors]
    # compute mean similarity
    mean_sim = float(np.mean([sims[j] for j in neighbors])) if neighbors else 0.0
    # compute local cov trace
    trace = local_cov_trace(i, X, neighbors) if neighbors else 0.0
    freedom_score = trace  # normalized externally
    is_free = (freedom_score > trace_thresh) and (mean_sim > sim_threshold)
    return is_free, freedom_score

# --- "Kuş gibi sevişme" detection ---
def detect_sevisme_set(X: np.ndarray, sim_low: float=0.25, sim_high: float=0.8, neighbor_k: int=8, ratio_thresh: float=0.6) -> List[int]:
    N = X.shape[0]
    sims = cosine_similarity_matrix(X)
    sevisme_ids = []
    for i in range(N):
        neigh = np.argsort(-sims[i])[1:neighbor_k+1]
        good = [j for j in neigh if sim_low <= sims[i,j] <= sim_high]
        if len(good) / max(1, len(neigh)) >= ratio_thresh:
            sevisme_ids.append(i)
    return sevisme_ids

# --- Example small run ---
if __name__ == "__main__":
    # create 20 users tat vectors
    T = np.array([make_tat(n=0.5+0.1*np.random.randn(), d=0.2+0.05*np.random.randn(),
                           e=0.3+0.1*np.random.randn(), o=0.1+0.05*np.random.randn(),
                           h=0.1+0.05*np.random.randn()) for _ in range(20)])
    # decay step
    T = np.array([decay_update(t, tau_decay=0.01) for t in T])
    # simulate a provoke on user 3
    T[3] = provoke_update(T[3], delta=0.2)
    # compute freedom and sevisme
    frees = [detect_freedom(i, T) for i in range(len(T))]
    sevisme = detect_sevisme_set(T)
    print("Freedom flags:", [f[0] for f in frees])
    print("Sevisme ids:", sevisme)

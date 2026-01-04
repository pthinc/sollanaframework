# cicikus_core.py
import time, math, json
from typing import Dict, List, Any, Callable, Tuple
import numpy as np

EPS = 1e-12
PI = math.pi

# Hooks to bind in integration (override these from the host system)
telemetry_fn: Callable[[Dict[str,Any]],None] = lambda e: print("TELEM", e)
persist_soul_tag: Callable[[str,Dict[str,Any]],None] = lambda id, d: None
apply_decay_suppression: Callable[[str,Dict[str,Any]],None] = lambda uid, ctx: None
flavor_boost: Callable[[List[str],Dict[str,Any]],None] = lambda uids, ctx: None

# Utilities
def clip01(x): return float(max(0.0, min(1.0, x)))
def now_ts(): return time.time()
def cos_sim(a,b):
    a=np.asarray(a); b=np.asarray(b)
    na=np.linalg.norm(a)+EPS; nb=np.linalg.norm(b)+EPS
    return float(np.dot(a,b)/(na*nb))

# Build E vector from Tat and optional multipliers
def build_E_from_T(T: np.ndarray, weights: Dict[str,float]=None) -> np.ndarray:
    w = np.array([weights.get("n",1.0), weights.get("d",1.0), weights.get("e",1.0), weights.get("o",1.0), weights.get("h",1.0)]) if weights else np.ones(5)
    E = np.asarray(T, dtype=float) * w
    E = np.clip(E, 0.0, 1.0)
    return E

# Field tensor F over contexts: represent as dict context_id -> outer(T,E)
def compute_field_block(T_vec: np.ndarray, E_vec: np.ndarray) -> np.ndarray:
    T = np.asarray(T_vec).reshape(5,1)
    E = np.asarray(E_vec).reshape(1,5)
    return (T @ E)  # 5x5 matrix

# A(t) scalar echo from block fields
def compute_A_from_fields(fields: List[np.ndarray], omega: float, phi: float, t: float, decay_scale: float=1.0) -> float:
    s = 0.0
    for F in fields:
        s += float(np.sum(F)) * math.sin(omega*t + phi)
    s *= float(max(0.0, 1.0 - decay_scale))
    return float(s)

# Divergence proxy: trace of local covariance among E vectors of neighbors
def divergence_proxy(i: int, E_matrix: np.ndarray, neighbors: List[int]) -> float:
    if not neighbors:
        return 0.0
    local = E_matrix[neighbors]
    if local.shape[0] < 2:
        return 0.0
    cov = np.cov(local, rowvar=False)
    return float(np.trace(cov))

# Freedom detection
def detect_freedom(i: int, E_matrix: np.ndarray, neighbors: List[int], trace_thresh: float, sim_thresh: float) -> Tuple[bool,float]:
    sims = np.dot(E_matrix, E_matrix[i]) / ((np.linalg.norm(E_matrix, axis=1)+EPS)*(np.linalg.norm(E_matrix[i])+EPS))
    neighbor_sims = [float(sims[j]) for j in neighbors] if neighbors else []
    mean_sim = float(np.mean(neighbor_sims)) if neighbor_sims else 0.0
    trace = divergence_proxy(i, E_matrix, neighbors)
    freedom_score = trace
    is_free = (freedom_score > trace_thresh) and (mean_sim > sim_thresh)
    return is_free, freedom_score

# Soul tag management
def update_soul_tags(user_id: str, E_vec: np.ndarray):
    tag = {"ts": now_ts(), "E": E_vec.tolist()}
    persist_soul_tag(user_id, tag)
    telemetry_fn({"event":"soul_tag_update","user":user_id,"ts":tag["ts"]})

def bond_strength(E_i: np.ndarray, E_j: np.ndarray) -> float:
    return cos_sim(E_i, E_j)

# High-level Cici action
class CiciEngine:
    def __init__(self, omega_base: float=0.5, phi_jitter: float=0.0):
        self.omega = omega_base
        self.phi_jitter = phi_jitter
        self.last_T: Dict[str,np.ndarray] = {}
        self.last_E: Dict[str,np.ndarray] = {}
        self.phi_map: Dict[str,float] = {}

    def tick_update(self, user_id: str, T_vec: List[float], weights: Dict[str,float]=None, delta_provoke: float=0.0):
        T = np.clip(np.asarray(T_vec, dtype=float),0.0,1.0)
        E = build_E_from_T(T, weights)
        if delta_provoke and delta_provoke>0.0:
            Cici = np.array([0.4,0.1,0.3,0.2,0.0])
            T = np.clip(T + delta_provoke * Cici,0.0,1.0)
            E = build_E_from_T(T, weights)
            apply_decay_suppression(user_id, {"reason":"provoke","delta":delta_provoke})
        self.last_T[user_id] = T
        self.last_E[user_id] = E
        self.phi_map.setdefault(user_id, (np.random.rand()*2*math.pi if self.phi_jitter>0 else 0.0))
        update_soul_tags(user_id, E)
        telemetry_fn({"event":"tick_update","user":user_id,"T":T.tolist(),"E":E.tolist(),"ts":now_ts()})
        return {"T":T,"E":E}

    def compute_group_echo(self, user_ids: List[str], omega_scale: float=1.0, decay_scale: float=0.0):
        t = now_ts()
        fields = []
        Es = []
        for uid in user_ids:
            T = self.last_T.get(uid)
            E = self.last_E.get(uid)
            if T is None or E is None: continue
            fields.append(compute_field_block(T,E))
            Es.append(E)
        if not fields:
            return {"A":0.0}
        A = compute_A_from_fields(fields, self.omega*omega_scale, 0.0, t, decay_scale)
        E_mat = np.vstack(Es)
        # neighbors simple kNN by similarity
        bonds = {}
        for i,uid in enumerate(user_ids[:len(Es)]):
            neigh = list(np.argsort(-np.dot(E_mat, E_mat[i]))[1:6])  # top 5
            is_free, score = detect_freedom(i, E_mat, neigh, trace_thresh=0.02, sim_thresh=0.2)
            bonds[uid] = {"freedom": is_free, "score": score}
        telemetry_fn({"event":"group_echo","A":A,"bonds":bonds,"ts":t})
        # gentle action for sevişme
        sevisme_uids = [uid for uid, v in bonds.items() if v["freedom"]]
        if sevisme_uids:
            flavor_boost(sevisme_uids, {"reason":"sevisme_boost"})
        return {"A":A,"bonds":bonds,"sevisme": sevisme_uids}

# Example usage
if __name__ == "__main__":
    engine = CiciEngine()
    users = ["u1","u2","u3","u4","u5"]
    for u in users:
        T = np.random.rand(5).tolist()
        engine.tick_update(u, T)
    result = engine.compute_group_echo(users)
    print("Group echo:", result)

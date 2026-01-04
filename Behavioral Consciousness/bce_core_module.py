# bce_core_module.py
"""
BCE core: computes BCE_i(t), Output_i(t), Z_i(t), N_match_i(t).
Dependencies: numpy, sklearn
"""

import time, math, json, os, random
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    DBSCAN = None
    BayesianGaussianMixture = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    from optimizations.master_law import MasterLawScorer
    MASTER_LAW_AVAILABLE = True
except Exception:
    MasterLawScorer = None
    MASTER_LAW_AVAILABLE = False

EPS = 1e-12
PHI = (1.0 + 5**0.5) / 2.0

# -------------------------
# Helpers
# -------------------------
def normalize_coeffs(coeffs: Dict[str, float]) -> Dict[str, float]:
    s = sum(abs(v) for v in coeffs.values()) + EPS
    return {k: float(v) / s for k, v in coeffs.items()}

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# -------------------------
# 1) Mini-RL (micro bandit)
# -------------------------
class MicroRL:
    def __init__(self, n: int = 4, lr: float = 0.05, eps: float = 0.05):
        self.n = n
        self.values = np.full(n, 1.0, dtype=float)
        self.lr = lr
        self.eps = eps
    def choose(self):
        if random.random() < self.eps:
            return random.randrange(self.n)
        return int(np.argmax(self.values))
    def update(self, idx: int, reward: float):
        self.values[idx] += self.lr * (reward - self.values[idx])
    def score(self) -> float:
        # produce a micro-RL signal in [0,1]
        v = np.clip(self.values.mean(), 0.0, 10.0)
        return float(sigmoid((v - 1.0)))
        
# -------------------------
# 2) IsolationForest wrapper
# -------------------------
class IsoAnomaly:
    def __init__(self):
        self.model = None
    def fit(self, X: np.ndarray):
        if not SKLEARN_AVAILABLE:
            return {"fitted": False, "reason": "sklearn_unavailable"}
        if X.shape[0] < 16:
            return {"fitted": False}
        self.model = IsolationForest(contamination=0.01, random_state=0)
        self.model.fit(X)
        return {"fitted": True}
    def anomaly_score(self, x: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        s = self.model.decision_function(x.reshape(1, -1))[0]
        # map to [0,1] anomaly (higher -> less anomalous)
        return float(1.0 - sigmoid(-s))
        
# -------------------------
# 3) Dcycle (user typology -> resonance pipeline)
# -------------------------
def dcycle_signal(user_type: str, phi_vec: np.ndarray, char_vector: np.ndarray, interaction_features: Dict[str, float]) -> float:
    """
    Lightweight heuristic:
      - match user_type via simple mapping
      - compute cosine similarity between phi_vec and char_vector
      - incorporate interaction intensity
    returns normalized [0,1] score
    """
    # user_type influence (simple lexicon)
    map_score = {"bağ_kurucu": 1.0, "görev_odaklı": 0.7, "yaratıcı": 0.9}
    ut = map_score.get(user_type, 0.6)
    # cosine similarity
    if np.linalg.norm(phi_vec) < EPS or np.linalg.norm(char_vector) < EPS:
        cos = 0.0
    else:
        cos = float(np.dot(phi_vec, char_vector) / (np.linalg.norm(phi_vec)*np.linalg.norm(char_vector)))
        cos = (cos + 1.0) / 2.0  # map [-1,1] -> [0,1]
    engage = float(interaction_features.get("engagement", 0.5))
    return float(np.clip(ut * 0.4 + cos * 0.4 + engage * 0.2, 0.0, 1.0))

# -------------------------
# 4) DBSCAN clustering wrapper (behavioral clustering)
# -------------------------
class BehaviorClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, tau: float = 0.6, sigma: float = 1.0):
        self.eps = eps
        self.min_samples = min_samples
        self.tau = tau
        self.sigma = sigma
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.labels = None
        self.Xs = None

    def fit(self, X: np.ndarray):
        if X.shape[0] < 4 or not SKLEARN_AVAILABLE or self.scaler is None:
            self.labels = np.array([-1]*X.shape[0])
            self.Xs = X
            return self.labels
        self.Xs = self.scaler.fit_transform(X)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.Xs)
        self.labels = self.model.labels_
        return self.labels

    def cluster_membership(self, X: np.ndarray) -> np.ndarray:
        if self.labels is None:
            self.fit(X)
        if self.labels is None or len(self.labels)==0:
            return np.zeros((X.shape[0],), dtype=float)
        labels = self.labels
        uniq, counts = np.unique(labels[labels!=-1], return_counts=True) if labels is not None else ([], [])
        if len(counts)==0:
            return np.zeros((labels.shape[0],), dtype=float)
        max_count = float(np.max(counts)) + EPS
        size_map = {int(k):float(c)/max_count for k,c in zip(uniq, counts)}
        mu = np.zeros_like(labels, dtype=float)
        for i,lbl in enumerate(labels):
            if lbl == -1:
                mu[i] = 0.0
            else:
                mu[i] = size_map.get(int(lbl), 0.0)
        return mu

    def cluster_risk(self, X: np.ndarray) -> float:
        mu = self.cluster_membership(X)
        if mu.size == 0:
            return 1.0
        # risk is fraction below tau
        risk = float(np.mean(mu < self.tau))
        return float(np.clip(risk, 0.0, 1.0))
        
# -------------------------
# 5) Self-Auto Discovery (simple daily checks)
# -------------------------
class SelfAutoDiscovery:
    def __init__(self):
        pass
    def run(self) -> Dict[str, Any]:
        # lightweight checks (placeholders)
        return {"ts": time.time(), "issues": [], "summary": "ok"}
        
# -------------------------
# 6) Bayesian Memory (BGMM)
# -------------------------
class BayesianMemory:
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.model: Optional[BayesianGaussianMixture] = None
    def fit(self, X: np.ndarray):
        if X.shape[0] < 8 or not SKLEARN_AVAILABLE:
            return {"fitted": False}
        self.model = BayesianGaussianMixture(n_components=self.n_components, random_state=0, max_iter=300)
        self.model.fit(X)
        return {"fitted": True}
    def context_score(self, phi_vec: np.ndarray, context_vec: Optional[np.ndarray]) -> float:
        if self.model is None:
            # fallback: cosine with context
            if context_vec is None or np.linalg.norm(phi_vec) < EPS or np.linalg.norm(context_vec) < EPS:
                return 0.5
            cos = float(np.dot(phi_vec, context_vec) / (np.linalg.norm(phi_vec)*np.linalg.norm(context_vec)))
            return float((cos + 1.0) / 2.0)
        # posterior weighting as proxy
        try:
            # responsibility-proxy: how well phi_vec fits model
            ll = float(self.model.score_samples(phi_vec.reshape(1,-1))[0])
            return float(sigmoid(ll))
        except Exception:
            return 0.5

# -------------------------
# 7) Fuzzy continuity (character/resonance)
# -------------------------
def fuzzy_continuity(phi_vec: np.ndarray, history_count: int, decay_rate: float) -> float:
    # combine recency (via decay_rate), usage counts and vector norm
    norm = float(np.tanh(np.linalg.norm(phi_vec)))
    count_scale = 1.0 / (1.0 + math.exp(-0.1*(history_count - 5)))
    decay_scale = 1.0 - float(np.clip(decay_rate, 0.0, 1.0))
    return float(np.clip(0.5*norm + 0.3*count_scale + 0.2*decay_scale, 0.0, 1.0))

# -------------------------
# 8) Adaptive Ethics (simple dynamic score)
# -------------------------
class AdaptiveEthics:
    def __init__(self):
        # store per-behavior or per-cluster ethical multipliers
        self.rules = {}
    def score(self, behavior_meta: Dict[str,Any]) -> float:
        # behavior_meta may contain 'ethical_tag' or 'sensitivity'
        tag = behavior_meta.get("ethical_tag", "approved")
        if tag in ("approved","ok","true"):
            return 1.0
        if tag in ("caution",):
            return 0.6
        return 0.0
    def propose_update(self, cluster_text: str) -> Dict[str,Any]:
        # placeholder for human-review proposal
        return {"proposal": "review", "proto": cluster_text}

# -------------------------
# Top-level BCE Engine
# -------------------------
class BCEEngine:
    def __init__(self, coeffs: Optional[Dict[str,float]] = None):
        # keys: rl, iso, dcycle, dbscan, auto, bayes, fuzzy, ethics, master
        default = {"rl":1.0, "iso":1.0, "dcycle":1.0, "dbscan":1.0, "auto":1.0, "bayes":1.0, "fuzzy":1.0, "ethics":1.0, "master":1.0}
        self.coeffs = normalize_coeffs(coeffs or default)
        # components
        self.rl = MicroRL(n=4)
        self.iso = IsoAnomaly()
        self.bayes = BayesianMemory(n_components=6)
        self.cluster = BehaviorClustering()
        self.self_auto = SelfAutoDiscovery()
        self.ethic = AdaptiveEthics()
        self.master = MasterLawScorer() if MASTER_LAW_AVAILABLE else None
    def compute_BCE(self, behavior: Dict[str,Any], context: Dict[str,Any]) -> float:
        """
        behavior: {phi_vec, phi_scalar, history_count, decay_rate, meta}
        context: {user_type, context_vec, interaction_features, recent_matrix_for_iso, clustering_matrix}
        """
        phi_vec = np.asarray(behavior.get("phi_vec", np.zeros(1)), dtype=float)
        phi_scalar = float(behavior.get("phi", 0.0))
        history_count = int(behavior.get("history_count", 0))
        decay_rate = float(behavior.get("decay_rate", 0.01))
        meta = behavior.get("meta", {})
        # 1 RL signal
        rl_score = self.rl.score()
        # 2 Isolation
        X_iso = np.asarray(context.get("recent_matrix_for_iso", np.zeros((0, phi_vec.shape[0]))))
        if X_iso.size:
            try:
                self.iso.fit(X_iso)
            except Exception:
                pass
        iso_s = self.iso.anomaly_score(phi_vec)
        # 3 Dcycle
        user_type = context.get("user_type", "unknown")
        char_vec = np.asarray(context.get("char_vector", np.zeros_like(phi_vec)), dtype=float)
        dcycle_s = dcycle_signal(user_type, phi_vec, char_vec, context.get("interaction_features", {}))
        # 4 DBSCAN risk
        X_cluster = np.asarray(context.get("clustering_matrix", np.zeros((0, phi_vec.shape[0]))))
        if X_cluster.size:
            self.cluster.fit(X_cluster)
        db_risk = 1.0 - (1.0 - self.cluster.cluster_risk(X_cluster))
        db_s = float(np.clip(1.0 - db_risk, 0.0, 1.0))
        # 5 Auto discovery
        auto_s = 1.0 if context.get("allow_auto", False) else 0.0
        # 6 Bayesian memory
        try:
            bayes_s = self.bayes.context_score(phi_vec, context.get("context_vec"))
        except Exception:
            bayes_s = 0.5
        # 7 fuzzy continuity
        fuzzy_s = fuzzy_continuity(phi_vec, history_count, decay_rate)
        # 8 ethics
        ethics_s = self.ethic.score(meta)
        # 9 Master Law (token pathting) if configured
        master_s = 0.0
        master_details = None
        if self.master and context.get("master_candidates"):
            try:
                mres = self.master.score(context.get("master_candidates", []), context.get("master_context", {}))
                probs = mres.get("probs") or []
                master_s = float(max(probs)) if probs else 0.0
                master_details = mres
            except Exception:
                master_s = 0.0
        comp = {
            "rl": rl_score,
            "iso": iso_s,
            "dcycle": dcycle_s,
            "dbscan": db_s,
            "auto": auto_s,
            "bayes": bayes_s,
            "fuzzy": fuzzy_s,
            "ethics": ethics_s,
            "master": master_s
        }
        # weighted sum
        w = self.coeffs
        BCE_val = sum(w[k] * comp[k] for k in comp.keys())
        # scale by phi_scalar (behavior strength)
        BCE_val = float(BCE_val * float(phi_scalar))
        if master_details is not None:
            comp["master_details"] = master_details
        return BCE_val, comp
    def compute_output(self, bce_val: float, behavior: Dict[str,Any]) -> float:
        """
        Output_i(t) = BCE_i(t) * (1 - D_i(t)) * R_i(t) * C_i(t)
        """
        decay = float(behavior.get("decay_level", 0.0))
        resonance = float(behavior.get("resonance", 1.0))
        char_sal = float(behavior.get("char_sal", 1.0))
        return float(bce_val * (1.0 - decay) * resonance * char_sal)
    def compute_Z(self, kappa: float, C: float, R: float, N: float, D: float) -> float:
        return float(kappa * (C + R + N) * (1.0 - D))
    def compute_Nmatch(self, user_type: str, C: float, E: float, D: float, S: float) -> float:
        # phi(U,C,E) simplified as weighted combo
        ut_map = {"bağ_kurucu":1.0, "görev_odaklı":0.7, "yaratıcı":0.9}
        ut = ut_map.get(user_type, 0.6)
        phi = float(0.4*ut + 0.4*C + 0.2*E)
        return float(phi * (1.0 - D) * S)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    engine = BCEEngine()
    # fake behavior + context
    b = {
        "phi_vec": np.random.rand(12).tolist(),
        "phi": 0.9,
        "history_count": 8,
        "decay_rate": 0.02,
        "meta": {"ethical_tag": "approved"},
        "decay_level": 0.05,
        "resonance": 0.8,
        "char_sal": 0.9
    }
    ctx = {
        "user_type": "bağ_kurucu",
        "context_vec": np.random.rand(12),
        "char_vector": np.random.rand(12),
        "recent_matrix_for_iso": np.random.rand(40,12),
        "clustering_matrix": np.random.rand(60,12),
        "interaction_features": {"engagement": 0.7},
        "allow_auto": True
    }
    bce_val, comps = engine.compute_BCE(b, ctx)
    out = engine.compute_output(bce_val, b)
    Z = engine.compute_Z(kappa=0.5, C=b.get("char_sal",1.0), R=b.get("resonance",1.0), N=0.7, D=b.get("decay_level",0.0))
    Nmatch = engine.compute_Nmatch(ctx["user_type"], C=b.get("char_sal",1.0), E=comps["ethics"], D=b.get("decay_level",0.0), S=0.9)
    print("BCE:", bce_val)
    print("components:", json.dumps(comps, indent=2))
    print("Output:", out)
    print("Z:", Z, "N_match:", Nmatch)

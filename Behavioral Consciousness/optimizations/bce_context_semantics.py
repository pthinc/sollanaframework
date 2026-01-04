# bce_context_semantics.py
# Requirements: numpy, scikit-learn, jsonschema
import time, os, json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from jsonschema import validate as js_validate, ValidationError

EPS = 1e-12

# -------------------------
# Short term context window
# -------------------------
class ShortContextWindow:
    def __init__(self, size: int = 32):
        self.size = int(size)
        self.buffer: List[Dict[str, Any]] = []

    def push(self, behavior: Dict[str, Any]):
        self.buffer.append(behavior)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def recent(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def summary(self) -> Dict[str, Any]:
        # simple statistical summary over phi vectors if present
        vecs = [np.asarray(b.get("phi_vec", []), dtype=float) for b in self.buffer if b.get("phi_vec") is not None]
        if not vecs:
            return {"count": len(self.buffer)}
        M = np.vstack([v for v in vecs if v.size])
        mean = M.mean(axis=0).tolist()
        std = M.std(axis=0).tolist()
        return {"count": len(self.buffer), "mean_dim": len(mean), "mean": mean, "std": std}

# -------------------------
# Long term checkpoint manager
# -------------------------
class CheckpointManager:
    def __init__(self, dirpath: str = "context_checkpoints", pca_dim: Optional[int] = 64):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
        self.pca_dim = pca_dim
        self.checkpoints: List[Dict[str, Any]] = []  # entries: {ts, id, emb}
        self.pca = PCA(n_components=pca_dim) if pca_dim is not None else None
        self._fitted = False

    def make_checkpoint(self, behaviors: List[Dict[str, Any]], context_id: str):
        # compute summary embedding (mean of phi_vec)
        vecs = [np.asarray(b["phi_vec"], dtype=float) for b in behaviors if "phi_vec" in b and b["phi_vec"]]
        if not vecs:
            emb = np.zeros(self.pca_dim or 1).tolist()
        else:
            M = np.vstack(vecs)
            emb = M.mean(axis=0)
            if self.pca is not None:
                if not self._fitted and M.shape[0] >= max(2, self.pca.n_components):
                    self.pca.fit(M)
                    self._fitted = True
                if self._fitted:
                    emb = self.pca.transform(emb.reshape(1,-1))[0]
            emb = emb.tolist()
        entry = {"ts": time.time(), "context_id": context_id, "emb": emb, "version": 1}
        self.checkpoints.append(entry)
        fname = os.path.join(self.dirpath, f"ckpt_{context_id}_{int(entry['ts'])}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entry, f)
        return entry

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return list(self.checkpoints)

# -------------------------
# Cross-behavior attention
# -------------------------
def cross_attention_score(ei: np.ndarray, ej: np.ndarray, time_weight: float = 1.0, dt: float = 0.0) -> float:
    # cosine similarity times temporal decay weight w_ij = exp(-lambda * dt)
    sim = float(np.clip(np.dot(ei, ej) / (np.linalg.norm(ei)+EPS) / (np.linalg.norm(ej)+EPS), -1.0, 1.0))
    lam = 0.001 * time_weight
    w = float(np.exp(-lam * dt))
    return float(sim * w)

class CrossBehaviorAttention:
    def __init__(self, lambda_time: float = 0.001):
        self.lambda_time = float(lambda_time)

    def attention_matrix(self, checkpoints: List[Dict[str, Any]], now: Optional[float] = None) -> np.ndarray:
        now = now or time.time()
        N = len(checkpoints)
        if N == 0:
            return np.zeros((0,0))
        E = np.vstack([np.asarray(c["emb"], dtype=float) for c in checkpoints])
        sims = cosine_similarity(E, E)
        times = np.array([now - c["ts"] for c in checkpoints], dtype=float)
        dt = np.abs(times.reshape(-1,1) - times.reshape(1,-1))
        W = np.exp(-self.lambda_time * dt)
        A = sims * W
        return A

# -------------------------
# Schema validation
# -------------------------
class SchemaValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            js_validate(instance=obj, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, str(e)

# -------------------------
# Rule engine (simple)
# -------------------------
class RuleEngine:
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        # rules: list of {"name":str, "fn": callable(behavior)->bool, "action":str}
        self.rules = rules or []

    def add_rule(self, name: str, fn, action: str):
        self.rules.append({"name": name, "fn": fn, "action": action})

    def evaluate(self, behavior: Dict[str, Any]) -> List[str]:
        actions = []
        for r in self.rules:
            try:
                if r["fn"](behavior):
                    actions.append(r["action"])
            except Exception:
                pass
        return actions

# -------------------------
# Normative and ethical filter
# -------------------------
class NormFilter:
    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)
        self.norm_profiles: Dict[str, Dict[str, float]] = {}  # norm_id -> stats

    def score_context_consistency(self, cur_emb: np.ndarray, prev_emb: np.ndarray) -> float:
        sim = float(safe_cosine(cur_emb, prev_emb))
        return float((sim + 1.0) / 2.0)

    def dynamic_threshold(self, E_hist: List[float], gamma: float = 1.0) -> float:
        import numpy as np
        arr = np.array(E_hist, dtype=float)
        mu = float(arr.mean()) if arr.size else 0.0
        sigma = float(arr.std()) if arr.size else 0.0
        return float(mu + gamma * sigma)

    def reject_if_violates(self, e_ctx: float, threshold: float) -> bool:
        return e_ctx > threshold

# -------------------------
# Context versioning & semantic matching
# -------------------------
class ContextVersioning:
    def __init__(self):
        self.versions: Dict[str, List[Dict[str, Any]]] = {}  # context_id -> list of {version, emb, ts}

    def register(self, context_id: str, emb: List[float], version_note: str = ""):
        vlist = self.versions.setdefault(context_id, [])
        ver = len(vlist) + 1
        entry = {"version": ver, "emb": emb, "ts": time.time(), "note": version_note}
        vlist.append(entry)
        return entry

def semantic_match(ci: Dict[str, Any], cj: Dict[str, Any], version_compat: bool = True) -> float:
    ei = np.asarray(ci["emb"], dtype=float)
    ej = np.asarray(cj["emb"], dtype=float)
    sim = float(safe_cosine(ei, ej))
    if version_compat:
        vi = ci.get("version", 1)
        vj = cj.get("version", 1)
        delta = 1.0 if vi == vj else 0.9  # soft penalty if versions differ
    else:
        delta = 1.0
    return float(sim * delta)

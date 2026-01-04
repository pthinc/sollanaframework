# BCE_performance_modules.py
"""
BCE Performance Modules
- RecalibrationFeedbackModule
- UserTypologyIntegration (with Emotional Feedback Loop)
- SelfAutoDiscoveryModule (sandboxed system introspection)
- ContextualMemoryEngine (Bayesian + fuzzy confidence)
- AdaptiveEthicsTypologyAlignment

Requirements: numpy, scipy, scikit-learn, python-dateutil, fuzzywuzzy
"""

import os
import time
import math
import json
import random
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dateutil import parser as dateparser
from fuzzywuzzy import fuzz

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BCE.Perf")

# -------------------------
# 1) Recalibration Feedback Module
# -------------------------
class MicroBanditAgent:
    """
    Micro RL-like bandit agent for tiny weight adjustments.
    Keeps CPU/GPU effect minimal by performing bounded updates
    and a strict per-interval budget.
    API:
      - choose(candidates): returns index
      - update(chosen_idx, reward)
      - step_budget_reset() each calibration window
    """
    def __init__(self, n_arms: int, lr: float = 0.05, init: float = 1.0, max_updates_per_window: int = 5):
        self.n = n_arms
        self.lr = float(lr)
        self.values = np.full(self.n, float(init), dtype=float)
        self.counts = np.zeros(self.n, dtype=int)
        self.updates_done = 0
        self.max_updates = int(max_updates_per_window)
        self.eps = 0.05  # small exploration

    def choose(self) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n)
        return int(np.argmax(self.values))

    def update(self, arm_idx: int, reward: float):
        if self.updates_done >= self.max_updates:
            return
        self.counts[arm_idx] += 1
        # simple incremental update (like bandit)
        self.values[arm_idx] += self.lr * (reward - self.values[arm_idx])
        self.updates_done += 1

    def reset_window(self):
        self.updates_done = 0

@dataclass
class RecalibrationFeedbackModule:
    """
    - Monitors resource usage samples (user-supplied)
    - Runs IsolationForest on recent metrics to produce anomaly/decay signals
    - Uses MicroBanditAgent for micro-updates to behavior selection weights
    - Produces daily logs and threshold suggestions
    """
    bandit: MicroBanditAgent
    isolation_window: int = 1024
    resource_buffer: Dict[str, List[float]] = field(default_factory=lambda: {"cpu": [], "gpu": [], "npu": []})
    iso_model: Optional[IsolationForest] = None
    anomaly_threshold: float = 0.5  # fraction of contamination suspect
    max_buffer: int = 2048
    out_dir: str = "recal_logs"

    def push_sample(self, cpu: float, gpu: float = 0.0, npu: float = 0.0):
        """Push a new resource usage sample (normalized 0..1)."""
        for k, v in (("cpu", cpu), ("gpu", gpu), ("npu", npu)):
            buf = self.resource_buffer.setdefault(k, [])
            buf.append(float(v))
            if len(buf) > self.max_buffer:
                buf.pop(0)

    def compute_anomaly(self) -> Dict[str, Any]:
        """Fit IsolationForest on combined recent samples (lightweight fit)."""
        # assemble feature vectors of last N
        keys = list(self.resource_buffer.keys())
        lengths = [len(self.resource_buffer[k]) for k in keys]
        if min(lengths) < 16:
            return {"ready": False}
        n = min(lengths, self.isolation_window)
        X = []
        for i in range(-n, 0):
            row = [self.resource_buffer[k][i] for k in keys]
            X.append(row)
        X = np.asarray(X)
        # fit small iso forest
        self.iso_model = IsolationForest(contamination=0.01, random_state=0)
        self.iso_model.fit(X)
        scores = self.iso_model.decision_function(X)  # higher is normal
        # detect fraction below median threshold
        anomaly_mask = scores < np.percentile(scores, 10)
        frac = float(np.mean(anomaly_mask))
        suggestion = {"anomaly_fraction": frac, "anomaly": frac > self.anomaly_threshold}
        return {"ready": True, "keys": keys, "n": n, "scores_mean": float(np.mean(scores)), **suggestion}

    def micro_recalibrate(self, candidate_rewards: List[float]):
        """Given small list of candidate rewards (len == bandit.n), update bandit a bit."""
        if len(candidate_rewards) != self.bandit.n:
            raise ValueError("candidate_rewards length must equal bandit arms")
        # budgeted updates: sample one arm (or choose best) and update with reward
        arm = self.bandit.choose()
        reward = float(candidate_rewards[arm])
        self.bandit.update(arm, reward)
        return {"arm": arm, "reward": reward, "values": self.bandit.values.tolist()}

    def daily_report(self) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        report = {"ts": time.time(), "bandit_values": self.bandit.values.tolist()}
        anomaly = self.compute_anomaly()
        report["anomaly_summary"] = anomaly
        # auto threshold suggestion example (very simple)
        if anomaly.get("ready"):
            if anomaly.get("anomaly"):
                report["suggestion"] = "reduce_update_rate"
            else:
                report["suggestion"] = "ok"
        fname = os.path.join(self.out_dir, f"recal_report_{int(time.time())}.json")
        atomic_write(fname, json.dumps(report, ensure_ascii=False, indent=2))
        return report

# -------------------------
# 2) User Typology Integration + Emotional Feedback Loop
# -------------------------
@dataclass
class UserTypology:
    """
    - aggregates per-user behavior vectors (Phi) and assigns a type cluster
    - uses DBSCAN for clustering; produces per-user emotional resonance and decay tracking
    """
    phi_store: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None
    cluster_eps: float = 0.5
    cluster_min_samples: int = 5
    reduced_dim: Optional[int] = 16

    def add_user_sample(self, user_id: str, phi_vec: np.ndarray, timestamp: Optional[float] = None):
        self.phi_store.setdefault(user_id, []).append(np.asarray(phi_vec, dtype=float))

    def build_feature_matrix(self) -> Tuple[np.ndarray, List[Tuple[str,int]]]:
        rows = []
        keys = []
        for uid, vecs in self.phi_store.items():
            # aggregate: mean + std features
            arr = np.vstack(vecs)
            feat = np.concatenate([np.mean(arr, axis=0), np.std(arr, axis=0)])
            rows.append(feat)
            keys.append((uid, len(vecs)))
        X = np.vstack(rows) if rows else np.zeros((0, 0))
        return X, keys

    def run_clustering(self) -> Dict[str, Any]:
        X, keys = self.build_feature_matrix()
        if X.size == 0:
            return {"clusters": {}, "meta": {}}
        # reduce dim
        if self.reduced_dim and X.shape[1] > self.reduced_dim:
            self.pca = PCA(n_components=self.reduced_dim, random_state=0)
            Xr = self.pca.fit_transform(X)
        else:
            Xr = X
        self.scaler = StandardScaler().fit(Xr)
        Xs = self.scaler.transform(Xr)
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(Xs)
        labels = db.labels_
        result = {}
        for (uid, cnt), lab in zip(keys, labels):
            result[uid] = {"label": int(lab), "samples": int(cnt)}
        return {"clusters": result, "labels": labels.tolist()}

    def user_resonance(self, user_id: str, cluster_map: Dict[str,int]) -> float:
        # simple heuristic: fuzz ratio between user aggregated centroid and cluster prototype names if available
        # here compute novelty/resonance as normalized sample count
        count = len(self.phi_store.get(user_id, []))
        return min(1.0, count / 50.0)

# -------------------------
# 3) Self-Auto Discovery Module (sandboxed)
# -------------------------
@dataclass
class SelfAutoDiscoveryModule:
    """
    - Runs daily short (bounded) discovery tasks inside a sandbox
    - Does static analysis, internal metric sweep, and proposes updates
    - External scraping is NOT performed automatically; placeholder for manual action
    """
    run_minutes: float = 5.0
    out_dir: str = "self_discovery"
    safe_checks: List[Callable[[], Dict[str,Any]]] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        # default checks
        self.safe_checks.append(self._check_module_sizes)
        self.safe_checks.append(self._check_recent_patterns)

    def _check_module_sizes(self) -> Dict[str,Any]:
        # inspect file sizes in cwd limited to python files
        entries = []
        for fn in os.listdir("."):
            if fn.endswith(".py"):
                try:
                    st = os.stat(fn)
                    entries.append({"file": fn, "size": st.st_size})
                except Exception:
                    pass
        return {"module_files": entries}

    def _check_recent_patterns(self) -> Dict[str,Any]:
        # look for patterns/ directory and list recent files
        out = []
        p = "patterns"
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p))[-20:]:
                out.append(fn)
        return {"recent_patterns": out}

    def run_discovery(self) -> Dict[str,Any]:
        """
        Run quick sandboxed discovery tasks and return a report.
        If a check indicates risk (heuristic), mark for ethics review.
        """
        start = time.time()
        report = {"ts": start, "checks": [], "notes": []}
        for check in self.safe_checks:
            try:
                res = check()
                report["checks"].append(res)
            except Exception as e:
                report["checks"].append({"error": str(e)})
        # heuristics: if many pattern files flagged, suggest review
        recent = report["checks"][-1].get("recent_patterns", [])
        if len(recent) > 50:
            report["notes"].append("many_patterns: suggest ethics review")
        # persist
        fname = os.path.join(self.out_dir, f"discovery_{int(start)}.json")
        atomic_write(fname, json.dumps(report, ensure_ascii=False, indent=2))
        return report

# -------------------------
# 4) Contextual Memory Engine (Bayesian + Fuzzy)
# -------------------------
@dataclass
class ContextualMemoryEngine:
    """
    - Accepts behavior phi vectors and feedback labels (optional)
    - Performs BayesianGaussianMixture clustering to assign soft labels
    - Computes fuzzy confidence combining posterior probs, sample counts, and time decay
    - Exposes protect() to mark traces that deserve decay protection
    """
    n_components: int = 4
    bgm: Optional[BayesianGaussianMixture] = None
    last_fit_time: Optional[float] = None
    min_samples_fit: int = 16
    memory_protection: Dict[str, float] = field(default_factory=dict)  # trace_id -> protect_until_ts

    def fit(self, phi_matrix: np.ndarray):
        if phi_matrix.shape[0] < self.min_samples_fit:
            return {"fitted": False, "reason": "not_enough_samples"}
        self.bgm = BayesianGaussianMixture(n_components=self.n_components, random_state=0, max_iter=500)
        self.bgm.fit(phi_matrix)
        self.last_fit_time = time.time()
        return {"fitted": True, "n_components": int(self.bgm.n_components)}

    def assign_posteriors(self, phi_matrix: np.ndarray) -> np.ndarray:
        if self.bgm is None:
            raise RuntimeError("BGMM not fitted")
        return self.bgm.predict_proba(phi_matrix)  # (N, K)

    def fuzzy_confidence(self, posteriors: np.ndarray, counts: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """
        posteriors: (N,K)
        counts: sample counts per behavior (N,)
        ages: age in seconds since last use (N,)
        returns fuzzy confidence in [0,1] per behavior
        """
        # posterior max
        max_post = np.max(posteriors, axis=1)
        # count_scale via logistic
        count_scale = 1.0 / (1.0 + np.exp(-0.1 * (counts - 5.0)))
        # age_scale (recent higher)
        age_scale = 1.0 / (1.0 + (ages / (3600.0 * 24.0)))  # daily scale
        conf = max_post * count_scale * age_scale
        return conf

    def protect_trace(self, trace_id: str, duration_s: float = 3600.0):
        self.memory_protection[trace_id] = time.time() + float(duration_s)

    def is_protected(self, trace_id: str) -> bool:
        ts = self.memory_protection.get(trace_id)
        return bool(ts and ts > time.time())

# -------------------------
# 5) Adaptive Ethics + Typology Alignment
# -------------------------
@dataclass
class AdaptiveEthicsTypologyAlignment:
    """
    - Aligns cluster->user types and proposes ethical filter updates
    - Compares cluster prototypes with norm definitions (norm_map)
    - Generates human-reviewable proposals and logs
    """
    norm_map: Dict[str, List[str]] = field(default_factory=dict)  # norm_id -> exemplar tokens or keywords
    proposal_dir: str = "ethic_proposals"

    def __post_init__(self):
        os.makedirs(self.proposal_dir, exist_ok=True)

    def compute_similarity(self, cluster_prototype_text: str, norm_keywords: List[str]) -> float:
        """Use fuzzy string matching as a lightweight semantic proxy."""
        scores = [fuzz.partial_ratio(cluster_prototype_text, kw) for kw in norm_keywords]
        return float(np.max(scores) / 100.0) if scores else 0.0

    def propose_updates(self, clusters: Dict[str, Dict], cluster_prototypes: Dict[str, str], user_typology: Dict[str,Any]) -> Dict[str,Any]:
        """
        clusters: cluster_id -> metadata (eth_ratio, decay_resist)
        cluster_prototypes: cluster_id -> textual prototype
        returns proposals list with confidence
        """
        proposals = []
        for cid, meta in clusters.items():
            proto = cluster_prototypes.get(cid, "")
            best = None
            for norm_id, keywords in self.norm_map.items():
                sim = self.compute_similarity(proto, keywords)
                if best is None or sim > best[1]:
                    best = (norm_id, sim)
            if best and best[1] < 0.5:
                # low alignment => propose an ethics rule strengthening or human review
                prop = {"cluster": cid, "proto": proto, "best_norm": best[0], "sim": best[1], "action": "human_review"}
                proposals.append(prop)
            else:
                # optionally propose soft mapping
                proposals.append({"cluster": cid, "best_norm": best[0] if best else None, "sim": best[1] if best else 0.0, "action": "map"})
        # persist
        fname = os.path.join(self.proposal_dir, f"ethic_proposals_{int(time.time())}.json")
        atomic_write(fname, json.dumps(proposals, ensure_ascii=False, indent=2))
        return {"proposals": proposals}

# -------------------------
# Demo usage
# -------------------------
def _demo():
    print("=== BCE Performance Modules Demo ===")
    # 1 Recalibration
    bandit = MicroBanditAgent(n_arms=3)
    recal = RecalibrationFeedbackModule(bandit)
    for _ in range(40):
        recal.push_sample(cpu=random.random()*0.2, gpu=random.random()*0.1, npu=random.random()*0.05)
    print("Anomaly summary:", recal.compute_anomaly())
    print("Micro recalibration:", recal.micro_recalibrate([0.1, 0.2, 0.05]))
    print("Daily report:", recal.daily_report())

    # 2 Typology
    typ = UserTypology()
    for u in ["alice","bob","carol"]:
        for _ in range(8):
            typ.add_user_sample(u, np.random.randn(8))
    clusters = typ.run_clustering()
    print("Typology clusters:", clusters.get("clusters"))

    # 3 Self discovery
    sad = SelfAutoDiscoveryModule()
    rep = sad.run_discovery()
    print("Discovery checks:", rep.get("checks")[:1])

    # 4 Contextual Memory
    cme = ContextualMemoryEngine()
    phi = np.random.rand(40, 12)
    print("CME fit:", cme.fit(phi))
    post = cme.assign_posteriors(phi) if cme.bgm is not None else None
    if post is not None:
        conf = cme.fuzzy_confidence(post, counts=np.random.randint(1,10,size=phi.shape[0]), ages=np.random.rand(phi.shape[0])*3600)
        print("CME confidences sample:", conf[:5])

    # 5 Adaptive Ethics
    aet = AdaptiveEthicsTypologyAlignment(norm_map={"safety":["do not harm","privacy"], "help":["assist","help","support"]})
    props = aet.propose_updates(clusters={"0":{"eth_ratio":0.9}}, cluster_prototypes={"0":"friendly assist and help user"})
    print("Ethic proposals:", props)

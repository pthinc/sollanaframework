# oksipitalist.py
# Requirements: numpy, sklearn
import time, json, math, os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

# ---------- Config ----------
ALPHA_GRID = np.linspace(0.1,1.0,10)
BETA_GRID  = np.linspace(0.1,1.0,10)
DBSCAN_EPS_GRID = np.linspace(0.2,1.0,9)
DBSCAN_MIN_GRID = list(range(3,11))
RA_THRESHOLD = 0.10
DRIFT_THRESHOLD = 0.15
RESOURCE_CAP = 1.0  # normalized units
FALLBACK_MODEL = "small-model"

# ---------- Hooks (attach real services) ----------
def persist_clusters(clusters: Dict[str,Any]):
    # simple default persistence to a local JSON snapshot
    try:
        path = os.path.join(os.getcwd(), "oksipitalist_clusters.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False)
    except Exception as e:
        emit_telemetry({"event":"persist_error","error":str(e)})
def emit_telemetry(event: Dict[str,Any]): print("TELEM", event)
def throttle_service(reason: str): print("THROTTLE:", reason)
def switch_to_fallback(name: str): print("SWITCH FALLBACK:", name)
def human_alert(payload: Dict[str,Any]): print("HUMAN ALERT:", payload)

# ---------- Bayesian weight model ----------
class BayesianSugList:
    def __init__(self):
        # store per-suglist stats
        self.stats: Dict[str, Dict[str,float]] = {}
    def ensure(self, sid: str, alpha=0.5, beta=0.5):
        if sid not in self.stats:
            self.stats[sid] = {"alpha":alpha,"beta":beta,"hit":0.0,"miss":0.0}
    def record_hit(self, sid: str):
        self.ensure(sid); self.stats[sid]["hit"] += 1.0
    def record_miss(self, sid: str):
        self.ensure(sid); self.stats[sid]["miss"] += 1.0
    def prob(self, sid: str) -> float:
        s = self.stats.get(sid)
        if not s: return 0.0
        num = s["alpha"] + s["hit"]
        den = s["alpha"] + s["beta"] + s["hit"] + s["miss"]
        return float(num / (den + 1e-12))
    def grid_tune(self, labeled: List[Tuple[str,int]], alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID):
        # labeled: list of (sug_id, label 1/0) from validation
        best = None; best_score=-1
        for a in alpha_grid:
            for b in beta_grid:
                # compute score on labels
                score=0.0
                for sid,lab in labeled:
                    self.ensure(sid,a,b)
                    p = self.prob(sid)
                    # log‑loss negative as proxy or simple accuracy
                    pred = 1 if p>0.5 else 0
                    if pred==lab: score+=1
                if score>best_score:
                    best_score=score; best=(a,b,score)
        return best

# ---------- DBSCAN sweep & recluster ----------
def dbscan_sweep(embeddings: np.ndarray, eps_grid=DBSCAN_EPS_GRID, min_grid=DBSCAN_MIN_GRID):
    best=None; best_metric=-1
    # metric: number of clusters minus noise ratio balanced with silhouette surrogate via intra/external distances
    for eps in eps_grid:
        for m in min_grid:
            model = DBSCAN(eps=float(eps), min_samples=int(m)).fit(embeddings)
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels==-1)/len(labels)
            # simplistic score: prefer more clusters and less noise
            score = n_clusters - noise_ratio
            if score>best_metric:
                best_metric=score; best=(eps,m,labels,score)
    return best

# ---------- Reflex telemetry monitor ----------
class ReflexMonitor:
    def __init__(self):
        self.unnecessary=0
        self.total=0
    def report_activation(self, unnecessary: bool):
        self.total += 1
        if unnecessary: self.unnecessary += 1
        ra = self.unnecessary / max(1,self.total)
        emit_telemetry({"ra":ra,"total":self.total})
        if ra > RA_THRESHOLD:
            human_alert({"event":"reflex_tuning_needed","ra":ra})

# ---------- Resource planner ----------
class ResourcePlanner:
    def __init__(self, cap=RESOURCE_CAP, gamma_reserve=0.2):
        self.cap=float(cap); self.gamma=float(gamma_reserve)
    def estimate_C(self, cpu_list: List[float], mem_list: List[float]) -> float:
        return float(sum(cpu_list) + sum(mem_list))
    def enforce(self, cpu_list: List[float], mem_list: List[float], expected_spike: float=0.0):
        C = self.estimate_C(cpu_list, mem_list)
        emit_telemetry({"resource_usage":C})
        if C > self.cap:
            throttle_service("usage_exceeded")
            switch_to_fallback(FALLBACK_MODEL)
        # reserve
        reserve = self.gamma * expected_spike
        return {"C":C,"reserve":reserve}

# ---------- Drift control for 4096 token window ----------
def context_drift_score(emb_before: np.ndarray, emb_after: np.ndarray) -> float:
    # cosine similarity inverse
    num = float(np.dot(emb_before, emb_after))
    den = (np.linalg.norm(emb_before)+EPS)*(np.linalg.norm(emb_after)+EPS)
    sim = num/den
    drift = 1.0 - sim
    return float(drift)

# ---------- High-level orchestrator ----------
class OksipitalistModule:
    def __init__(self):
        self.bayes = BayesianSugList()
        self.reflex = ReflexMonitor()
        self.rplanner = ResourcePlanner()
        self.last_cluster_map = None
    def process_suglists(self, sug_embeddings: np.ndarray, sug_ids: List[str]):
        # recluster daily or upon call
        best = dbscan_sweep(sug_embeddings)
        eps, min_s, labels, score = best
        cluster_map = {sid:int(l) for sid,l in zip(sug_ids, labels)}
        # compute alignment delta with last map
        if self.last_cluster_map is not None:
            pre = set(self.last_cluster_map.keys())
            post = set(cluster_map.keys())
            inter = len(pre & post)
            delta_a = inter / max(1, len(post))
        else:
            delta_a = 1.0
        persist_clusters({"eps":eps,"min_samples":min_s,"labels":labels.tolist(),"delta_a":delta_a})
        self.last_cluster_map = cluster_map
        emit_telemetry({"event":"recluster_done","eps":eps,"min_samples":min_s,"delta_a":delta_a})
        return cluster_map
    def handle_activation(self, unnecessary_flag: bool):
        self.reflex.report_activation(unnecessary_flag)
    def resource_check(self, cpu_list, mem_list, expected_spike=0.0):
        return self.rplanner.enforce(cpu_list, mem_list, expected_spike)
    def vulnerability_recluster_on_drift(self, emb_before, emb_after, threshold=DRIFT_THRESHOLD):
        drift = context_drift_score(emb_before, emb_after)
        emit_telemetry({"event":"drift_check","drift":drift})
        if drift > threshold:
            # retrain clusters and tune bayes weights
            # placeholder: trigger recluster external call
            human_alert({"event":"drift_recluster_needed","drift":drift})
            return True
        return False

# ---------- Example usage ----------
if __name__ == "__main__":
    mod = OksipitalistModule()
    # create dummy sug embeddings and ids
    N=100; d=32
    X = np.random.randn(N,d)
    ids = [f"s{i}" for i in range(N)]
    cluster_map = mod.process_suglists(X, ids)
    mod.handle_activation(unnecessary_flag=True)
    mod.resource_check([0.2,0.3],[0.1,0.4], expected_spike=0.5)
    # simulate drift
    emb_b = np.random.randn(d)
    emb_a = emb_b*0.95 + 0.05*np.random.randn(d)
    mod.vulnerability_recluster_on_drift(emb_b, emb_a)

# bce_graph_clustering.py
import time, math
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

class GraphClustering:
    def __init__(self, sim_threshold: float = 0.7, agglom_k: int = 8):
        self.sim_threshold = sim_threshold
        self.agglom_k = agglom_k
        self.history = []  # list of (ts, node_id, phi_vec, C, D, R)

    def ingest(self, node_id: str, phi_vec: np.ndarray, C: float, D: float, R: float, ts: Optional[float] = None):
        ts = ts or time.time()
        self.history.append((ts, node_id, np.asarray(phi_vec, dtype=float), float(C), float(D), float(R)))

    def build_graph_score(self, window_seconds: float = 3600.0, now: Optional[float] = None) -> float:
        now = now or time.time()
        nodes = [h for h in self.history if h[0] >= now - window_seconds]
        if not nodes:
            return 0.0
        total = 0.0
        for (_, _, _, C, D, R) in nodes:
            total += C * (1.0 - D) * R
        return float(total)

    def detect_evolutionary_jump(self, prev_score: float, window_seconds: float = 3600.0, threshold: float = 0.2) -> Tuple[bool, float, float]:
        now = time.time()
        s_now = self.build_graph_score(window_seconds, now)
        delta = s_now - prev_score
        triggered = delta > threshold
        return triggered, s_now, delta

    def cluster_behavior_nodes(self, n_clusters: Optional[int] = None):
        # cluster recent phi vectors by agglomerative on cosine similarity
        if not self.history:
            return {}
        X = np.vstack([h[2] for h in self.history])
        # cosine affinity via precomputed distances: convert to euclidean surrogate with PCA is optional
        k = n_clusters or min(self.agglom_k, max(2, len(X)//2))
        model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        try:
            labels = model.fit_predict(X)
        except Exception:
            # fallback: single cluster
            labels = np.zeros(len(X), dtype=int)
        clusters = {}
        for (rec, lab) in zip(self.history, labels):
            clusters.setdefault(int(lab), []).append(rec)
        return clusters

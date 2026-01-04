# superego.py
import math
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def compute_intersections(norm_map, cluster_map):
    # norm_map: norm_id -> set(behavior_ids)
    # cluster_map: cluster_id -> set(behavior_ids)
    intersects = {}
    for n_id, n_set in norm_map.items():
        for c_id, c_set in cluster_map.items():
            key = (n_id, c_id)
            intersects[key] = len(n_set & c_set)
    return intersects

def compute_superego(norm_map, cluster_map, cluster_meta, norm_meta, alpha=1.0):
    # cluster_meta[cluster_id] = {"eth_ratio":0.9, "decay_resist":0.8}
    # norm_meta[norm_id] = {"critical": True/False}
    inter = compute_intersections(norm_map, cluster_map)
    score = 0.0
    norm_factor = 0.0
    for (n_id, c_id), cnt in inter.items():
        if cnt == 0: continue
        eth = cluster_meta.get(c_id, {}).get("eth_ratio", 1.0)
        decay_r = cluster_meta.get(c_id, {}).get("decay_resist", 1.0)
        crit = 1.5 if norm_meta.get(n_id, {}).get("critical", False) else 1.0
        w = cnt * eth * decay_r * crit
        score += w
        norm_factor += cnt
    if norm_factor <= 0:
        return 0.0
    normalized = score / (norm_factor + 1e-9)
    return sigmoid(alpha * (normalized - 0.5))  # center at 0.5 for sensitivity

def superego_status(value):
    if value >= 0.9: return "healthy"
    if value >= 0.5: return "at_risk"
    if value >= 0.2: return "fragmented"
    return "nascent"

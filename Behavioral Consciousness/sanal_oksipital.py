# sanal_oksipital.py
"""
Prototype for Sanal Oksipital
Requires: numpy, sklearn
"""
import time, os, json, math
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import heapq

EPS = 1e-12

# -------------------------
# MiniBatch clustering manager
# -------------------------
class ClusterManager:
    def __init__(self, n_clusters=32, embed_dim=128, batch_size=256, lambda_flavor=0.1, lambda_emotion=0.1):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        self.fitted = False
        self.mu = np.zeros((n_clusters, embed_dim))
        self.lambda_flavor = lambda_flavor
        self.lambda_emotion = lambda_emotion
        self.embed_dim = embed_dim

    def partial_fit(self, X_batch: np.ndarray):
        if X_batch.shape[0] < 2:
            return
        if not self.fitted:
            # initialize with first batch
            self.kmeans.partial_fit(X_batch)
            self.fitted = True
            try:
                self.mu = self.kmeans.cluster_centers_
            except Exception:
                self.mu = np.zeros((self.n_clusters, self.embed_dim))
            return
        self.kmeans.partial_fit(X_batch)
        self.mu = self.kmeans.cluster_centers_

    def stabilize_centers(self, flavor_buffer_func: Callable[[int], np.ndarray], emotion_imprint_func: Callable[[int], np.ndarray]):
        # augment centers with flavor+emotion imprints
        for j in range(self.n_clusters):
            fb = flavor_buffer_func(j) if callable(flavor_buffer_func) else np.zeros(self.embed_dim)
            ei = emotion_imprint_func(j) if callable(emotion_imprint_func) else np.zeros(self.embed_dim)
            self.mu[j] = self.mu[j] + self.lambda_flavor * fb + self.lambda_emotion * ei

# -------------------------
# Kalman + PID decay estimator
# -------------------------
class SimpleKalman:
    def __init__(self, q=1e-5, r=1e-2):
        self.q = q  # process noise
        self.r = r  # measurement noise
        self.x = 0.0
        self.P = 1.0

    def update(self, z):
        # predict
        self.P += self.q
        # update
        K = self.P / (self.P + self.r + EPS)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

class PID:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, dt=1.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.dt = dt
        self.I = 0.0; self.prev = 0.0
    def step(self, e):
        self.I += e * self.dt
        D = (e - self.prev) / (self.dt + EPS)
        out = self.kp*e + self.ki*self.I + self.kd*D
        self.prev = e
        return out

# -------------------------
# Memory layers
# -------------------------
class MemoryLayers:
    def __init__(self, short_len=8):
        self.short = []  # list of embeddings (recent)
        self.long = []   # list of checkpoints (mean embeddings)
        self.short_len = short_len

    def push_short(self, emb: np.ndarray):
        self.short.append(emb)
        if len(self.short) > self.short_len:
            self.short.pop(0)

    def checkpoint(self):
        if not self.short:
            return None
        mean_emb = np.mean(np.vstack(self.short), axis=0)
        self.long.append({"ts": time.time(), "emb": mean_emb})
        return mean_emb

# -------------------------
# Adaptive attention
# -------------------------
def adaptive_attention(short_embs: List[np.ndarray], long_embs: List[Dict[str,Any]], lambda_time=0.001):
    if not short_embs or not long_embs:
        return np.array([])
    Q = np.vstack(short_embs)
    C = np.vstack([e["emb"] for e in long_embs])
    sims = Q.dot(C.T) / (np.linalg.norm(Q, axis=1)[:,None] * (np.linalg.norm(C, axis=1)[None,:] + EPS) + EPS)
    # time weighting
    now = time.time()
    times = np.array([now - e["ts"] for e in long_embs])
    dt = np.abs(times[None,:] - np.zeros_like(times)[None,:])  # proxy
    W = np.exp(-lambda_time * dt)
    A = sims * W
    return A  # shape (len(short), len(long))

# -------------------------
# Bayesian tag selection
# -------------------------
class TagManager:
    def __init__(self, theta=0.7, alpha=0.6):
        self.priors = {}  # tag -> prior weight
        self.theta = theta
        self.alpha = alpha
        self.blacklist = set()

    def update_prior(self, tag, observed):
        pprior = self.priors.get(tag, 0.1)
        self.priors[tag] = self.alpha * pprior + (1.0 - self.alpha) * observed

    def validate(self, tag, context_vec, tag_vec_fn: Callable[[str], np.ndarray]):
        if tag in self.blacklist:
            return False
        tag_vec = tag_vec_fn(tag)
        sim = np.dot(tag_vec, context_vec) / ( (np.linalg.norm(tag_vec)+EPS) * (np.linalg.norm(context_vec)+EPS) )
        if sim < self.theta:
            self.blacklist.add(tag)
            return False
        return True

# -------------------------
# Anomaly pipeline
# -------------------------
class AnomalyPipeline:
    def __init__(self, contamination=0.01):
        self.iso = IsolationForest(contamination=contamination)
        self.fitted = False

    def fit(self, X: np.ndarray):
        if X.shape[0] < 8:
            return
        self.iso.fit(X)
        self.fitted = True

    def score(self, x: np.ndarray) -> float:
        if not self.fitted:
            return 0.0
        return float(self.iso.decision_function(x.reshape(1,-1))[0])

# -------------------------
# Simple priority queue for anomalies or rollbacks
# -------------------------
class PriorityQueue:
    def __init__(self):
        self.heap = []
    def push(self, priority, payload):
        heapq.heappush(self.heap, (priority, payload))
    def pop(self):
        if not self.heap: return None
        return heapq.heappop(self.heap)

def mini_forget_step(items, theta, gamma):
    """
    items: list of {"id", "vec", "relevance","resonance","context_fit","active_weight"}
    returns updated items and archive list
    """
    archive = []
    for it in items:
        Q = it["relevance"] + it["resonance"] + it["context_fit"]
        if Q <= theta:
            it["active_weight"] *= gamma
            it.setdefault("decay_steps", 0)
            it["decay_steps"] += 1
            if it["decay_steps"] > 5 and it["active_weight"] < 1e-3:
                archive.append(it["id"])
    return items, archive

def meta_decay_update(w, dt, lam, eta, trait_update):
    w_new = w * math.exp(-lam * dt) + eta * trait_update
    return max(0.0, w_new)

d_hat = kalman.update(D_obs)
corr = pid.step(D_obs - d_hat)
apply_correction(corr)  # e.g., reduce exploration, increase snapshot

iso = IsolationForest(contamination=0.01).fit(X_train)
score = iso.decision_function(x_test.reshape(1,-1))[0]
if score < anomaly_threshold:
    priority_queue.push(priority=score, payload=event)


# -------------------------
# Example orchestration
# -------------------------
if __name__ == "__main__":
    # brief demo of flow
    emb_dim = 64
    cm = ClusterManager(n_clusters=8, embed_dim=emb_dim)
    ml = MemoryLayers(short_len=6)
    km = SimpleKalman(); pid = PID()
    tagm = TagManager(); anom = AnomalyPipeline()

    # fake stream
    rng = np.random.RandomState(0)
    for i in range(120):
        emb = rng.randn(emb_dim)
        ml.push_short(emb)
        if i % 16 == 0:
            ck = ml.checkpoint()
            if ck is not None:
                cm.partial_fit(np.vstack([ck + 0.01 * rng.randn(emb_dim) for _ in range(32)]))
        # decay measurement sim
        D = max(0.0, 0.05 * (1.0 - math.cos(i/8.0)))
        Dk = km.update(D)
        correction = pid.step(D - Dk)
        # anomaly engine feed
        if i % 20 == 0 and len(ml.long) > 2:
            X = np.vstack([e["emb"] for e in ml.long])
            anom.fit(X)
        # simple telemetry print
        if i % 30 == 0:
            print("i",i,"D",D,"Dk",Dk,"corr",correction)
    print("Demo done")


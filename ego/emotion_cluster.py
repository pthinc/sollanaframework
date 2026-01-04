# emotion_cluster.py (özet)
from sklearn.preprocessing import StandardScaler
import umap, hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cluster_phis(phi_matrix):
    X = StandardScaler().fit_transform(phi_matrix)
    Xr = umap.UMAP(n_components=32).fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20).fit(Xr)
    labels = clusterer.labels_
    return Xr, labels, clusterer

def soft_assignment(phi_vecs, centers, M_scores, beta=5.0):
    sims = cosine_similarity(phi_vecs, centers)
    scaled = beta * sims * np.expand_dims(M_scores, axis=1)
    ex = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    return probs

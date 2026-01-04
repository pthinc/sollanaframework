# cluster_emotions.py (özet)
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import math

def prepare_phi_matrix(phi_list):
    X = np.vstack(phi_list)                   # shape (N, d)
    X = StandardScaler().fit_transform(X)
    return X

def reduce_dim(X, n_components=32, random_state=0):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X), reducer

def batch_cluster(X_reduced, min_cluster_size=20):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    labels = clusterer.fit_predict(X_reduced)
    return labels, clusterer

def compute_centroids(X, labels):
    unique = sorted(set(labels) - {-1})
    centers = []
    for c in unique:
        centers.append(X[labels == c].mean(axis=0))
    return np.vstack(centers), unique

def soft_assignment(phi_vecs, centers, M_scores, beta=5.0):
    sims = cosine_similarity(phi_vecs, centers)            # (N, K)
    scaled = beta * sims * np.expand_dims(M_scores, axis=1)
    ex = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    return probs

def assign_to_centroids(phi_vec, centers):
    sims = cosine_similarity(phi_vec.reshape(1,-1), centers).ravel()
    idx = sims.argmax()
    return idx, sims[idx]

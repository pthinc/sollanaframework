# behavior_clustering.py
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

ROOT = Path("data")
ROOT.mkdir(exist_ok=True)

def _save_json(obj: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def cluster_behaviors_batch(embeddings: np.ndarray,
                            n_init_components: int = 8,
                            max_components: int = 16,
                            random_state: int = 0) -> Dict[str, Any]:
    """
    Batch fit BGMM, return labels, centers (cluster means), model metadata.
    embeddings shape: (N, D)
    """
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type="dirichlet_process",
        random_state=random_state,
        max_iter=500
    )
    bgm.fit(embeddings)
    labels = bgm.predict(embeddings)
    # compute cluster centroids as weighted means of components
    # fallback: use component means
    centers = bgm.means_.tolist()
    meta = {
        "algo": "BayesianGaussianMixture",
        "n_components": int(bgm.n_components),
        "converged": bool(getattr(bgm, "converged_", False)),
        "fitted_at": time.time()
    }
    return {"labels": labels.tolist(), "centers": centers, "meta": meta, "model": bgm}

def persist_clusters(labels: List[int], centers: List[List[float]], out_prefix: str = "behavior_clusters"):
    ROOT.mkdir(exist_ok=True)
    clusters_path = ROOT / f"{out_prefix}.json"
    centers_path = ROOT / f"{out_prefix}_centers.json"
    _save_json({"labels": labels}, clusters_path)
    _save_json({"centers": centers, "saved_at": time.time()}, centers_path)
    return clusters_path, centers_path

def fit_mini_batch_kmeans(embeddings: np.ndarray, k: int = 8, batch_size: int = 1024, random_state: int = 0) -> Tuple[MiniBatchKMeans, List[List[float]]]:
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=random_state)
    mbk.fit(embeddings)
    centers = mbk.cluster_centers_.tolist()
    return mbk, centers

def assign_to_centers(embeddings: np.ndarray, centers: List[List[float]]) -> np.ndarray:
    C = np.asarray(centers)
    # squared euclidean distance assignment
    dists = np.sum((embeddings[:, None, :] - C[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    return labels

def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        if len(set(labels.tolist())) > 1:
            out["silhouette"] = float(silhouette_score(embeddings, labels))
            out["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
        else:
            out["silhouette"] = -1.0
            out["calinski_harabasz"] = -1.0
    except Exception:
        out["silhouette"] = -1.0
        out["calinski_harabasz"] = -1.0
    return out

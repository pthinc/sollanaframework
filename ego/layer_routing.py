# layer_routing.py
"""Layer routing with numpy + sklearn; backend-agnostic (no torch dependency)."""

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import BayesianGaussianMixture
from typing import Dict, Any, Optional

class ConsciousnessLayerManager:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {"ego": 0.7, "superego": 0.85}
        self.context_anchor: Dict[str, np.ndarray] = {}

    def route(self, behavior_score: float, context_match: float, ethical_tag: Any) -> str:
        if behavior_score < self.thresholds["ego"]:
            return "id"
        if context_match >= self.thresholds["ego"]:
            if self._ethically_approved(ethical_tag):
                return "superego" if context_match >= self.thresholds["superego"] else "ego"
            return "ego"
        return "id"

    def _ethically_approved(self, ethical_tag: Any) -> bool:
        if isinstance(ethical_tag, str):
            return ethical_tag.lower() in ("approved", "true", "ok")
        if isinstance(ethical_tag, (int, float)):
            return float(ethical_tag) >= 0.5
        return False

def is_contextual(behavior_vector: np.ndarray, context_vector: np.ndarray, threshold: float = 0.7) -> bool:
    if behavior_vector is None or context_vector is None:
        return False
    behavior_vector = behavior_vector.reshape(1, -1)
    context_vector = context_vector.reshape(1, -1)
    sim = float(cosine_similarity(behavior_vector, context_vector)[0][0])
    return sim > threshold

def ethical_filter(behavior_embeddings: np.ndarray, feedback_labels: Dict[int, str]) -> str:
    if behavior_embeddings.shape[0] == 0:
        return "rejected"
    bgm = BayesianGaussianMixture(n_components=2, random_state=0, max_iter=500)
    bgm.fit(behavior_embeddings)
    preds = bgm.predict(behavior_embeddings)
    cluster = int(preds[0])
    label = feedback_labels.get(cluster, "false")
    return "approved" if label in ("true", "approved") else "rejected"

# Örnek kullanım
if __name__ == "__main__":
    mgr = ConsciousnessLayerManager()
    behavior_score = 0.8
    behavior_vec = np.random.randn(512)
    context_vec = behavior_vec * 0.95
    context_match = float(cosine_similarity(behavior_vec.reshape(1,-1), context_vec.reshape(1,-1))[0][0])
    ethical_tag = "approved"
    layer = mgr.route(behavior_score, context_match, ethical_tag)
    print("Routed to", layer)

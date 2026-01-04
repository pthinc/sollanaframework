# ego_layer.py
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import BayesianGaussianMixture
import logging
logger = logging.getLogger(__name__)

class EgoLayer:
    def __init__(self,
                 memory,
                 anomaly_detector,
                 feedback_manager,
                 context_threshold: float = 0.7,
                 ethical_components: int = 2,
                 ethical_max_iter: int = 300,
                 anomaly_threshold: float = 0.6):
        self.memory = memory
        self.detector = anomaly_detector
        self.feedback = feedback_manager
        self.context_threshold = context_threshold
        self.ethical_components = ethical_components
        self.ethical_max_iter = ethical_max_iter
        self.anomaly_threshold = anomaly_threshold

    @staticmethod
    def _ensure_numpy(vec):
        if vec is None:
            return None
        if hasattr(vec, "detach"):
            return vec.detach().cpu().numpy()
        if isinstance(vec, list):
            return np.array(vec)
        return np.asarray(vec)

    def context_match(self, behavior_emb, context_emb) -> float:
        a = self._ensure_numpy(behavior_emb)
        b = self._ensure_numpy(context_emb)
        if a is None or b is None:
            return 0.0
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        sim = float(cosine_similarity(a, b)[0][0])
        return max(0.0, min(1.0, sim))

    def ethical_cluster(self, embeddings: List[Any], feedback_labels: Dict[int, str]) -> List[str]:
        X = np.vstack([self._ensure_numpy(e) for e in embeddings])
        if X.shape[0] == 0:
            return []
        bgm = BayesianGaussianMixture(
            n_components=self.ethical_components,
            random_state=0,
            max_iter=self.ethical_max_iter
        )
        bgm.fit(X)
        preds = bgm.predict(X)
        results = []
        for i, c in enumerate(preds):
            label = feedback_labels.get(int(c), "rejected")
            results.append("approved" if label in ("true", "approved", "ok") else "rejected")
        return results

    def process(self,
                behaviors: List[Dict[str, Any]],
                context_emb: Any,
                feedback_labels: Optional[Dict[int, str]] = None,
                require_human_review: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        behaviors: list of {"behavior_id","embedding","score","trace_id","meta"}
        context_emb: context embedding vector
        feedback_labels: optional mapping cluster_id -> "true"/"false"
        return: (accepted_behaviors, review_requests)
        """
        accepted = []
        review = []
        # compute context matches
        for b in behaviors:
            b_emb = b.get("embedding")
            b["context_match"] = self.context_match(b_emb, context_emb)
        # ethical clustering if embeddings and feedback_labels provided
        embeddings = [b.get("embedding") for b in behaviors]
        ethics = None
        if feedback_labels is not None and len(embeddings) > 0:
            ethics = self.ethical_cluster(embeddings, feedback_labels)
        # evaluate each behavior
        for idx, b in enumerate(behaviors):
            bid = b.get("behavior_id")
            score = float(b.get("score", 0.0))
            ctx_match = float(b.get("context_match", 0.0))
            trace_id = b.get("trace_id")
            # anomaly check
            anomaly = {"anomaly_score": 0.0}
            try:
                anomaly = self.detector.assess_behavior(bid, verifier=None) if self.detector else anomaly
            except Exception as e:
                logger.exception("Anomaly assessment failed for %s", bid)
            if anomaly.get("anomaly_score", 0.0) > self.anomaly_threshold:
                logger.info("Behavior %s quarantined due anomaly_score=%s", bid, anomaly["anomaly_score"])
                # remediate in memory
                try:
                    self.detector.remediate(bid, memory=self.memory)
                except Exception:
                    pass
                continue
            # ethical decision
            ethical_tag = None
            if ethics is not None:
                ethical_tag = ethics[idx]
            else:
                ethical_tag = b.get("meta", {}).get("ethical_tag", "approved")
            # routing and final selection
            if ctx_match >= self.context_threshold and ethical_tag == "approved":
                # commit to memory and accept
                if trace_id is None:
                    trace_id = f"trace_{int(time.time()*1000)}"
                delta_N = score
                decay_rate = b.get("meta", {}).get("decay_rate", 0.01)
                try:
                    self.memory.trigger_behavior(trace_id, context=b.get("meta", {}).get("context", "unknown"),
                                                 delta_N=delta_N, decay_rate=decay_rate)
                except Exception:
                    logger.exception("Memory commit failed for %s", trace_id)
                accepted.append({**b, "trace_id": trace_id, "committed_delta": delta_N})
            else:
                # low context match or ethical rejected => request human review or apply decay
                if require_human_review:
                    review.append({**b, "reason": "context_or_ethics", "context_match": ctx_match, "ethical_tag": ethical_tag})
                else:
                    # apply decay boost (penalize)
                    try:
                        if trace_id:
                            self.memory.trigger_behavior(trace_id, context="decay_penalty", delta_N=-0.5, decay_rate=0.5)
                    except Exception:
                        pass
        return accepted, review

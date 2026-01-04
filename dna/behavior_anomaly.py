# behavior_anomaly.py
import time
import math
from typing import Dict, List, Any, Callable, Optional
import collections
import torch
import torch.nn.functional as F

class BehaviorEvent:
    def __init__(self, behavior_id: str, score: float, context_vec: Optional[torch.Tensor],
                 timestamp: Optional[float] = None, trace_id: Optional[str] = None, meta: Optional[Dict] = None):
        self.behavior_id = behavior_id
        self.score = float(score)
        self.context_vec = context_vec
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.trace_id = trace_id
        self.meta = meta or {}

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return 0.0
    a = a.view(-1).float()
    b = b.view(-1).float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

class BehaviorAnomalyDetector:
    def __init__(self,
                 window_seconds: float = 3600.0,
                 bias_threshold: float = 0.6,
                 repetition_threshold: int = 10,
                 context_drift_threshold: float = 0.4,
                 hallucination_threshold: float = 0.5,
                 score_weights: Optional[Dict[str, float]] = None,
                 alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.window_seconds = window_seconds
        self.bias_threshold = bias_threshold
        self.repetition_threshold = repetition_threshold
        self.context_drift_threshold = context_drift_threshold
        self.hallucination_threshold = hallucination_threshold
        self.score_weights = score_weights or {
            "bias": 1.0,
            "repetition": 1.0,
            "context_drift": 1.0,
            "hallucination": 1.5,
            "confidence_mismatch": 1.0
        }
        self.alert_callback = alert_callback
        self.events: Dict[str, collections.deque] = {}
        self.latest_context_anchor: Dict[str, torch.Tensor] = {}

    def log_event(self, event: BehaviorEvent):
        q = self.events.setdefault(event.behavior_id, collections.deque())
        q.append(event)
        self._prune_queue(event.behavior_id)
        if event.context_vec is not None:
            self.latest_context_anchor[event.behavior_id] = event.context_vec.clone()

    def _prune_queue(self, behavior_id: str):
        q = self.events.get(behavior_id)
        if not q:
            return
        cutoff = time.time() - self.window_seconds
        while q and q[0].timestamp < cutoff:
            q.popleft()

    def _score_bias(self, behavior_id: str) -> float:
        q = self.events.get(behavior_id, [])
        if not q:
            return 0.0
        scores = [e.score for e in q]
        if len(scores) < 2:
            return 0.0
        dispersion = float(torch.tensor(scores).float().std().item())
        norm = dispersion / (max(scores) - min(scores) + 1e-9)
        return min(1.0, norm)

    def _score_repetition(self, behavior_id: str) -> float:
        q = self.events.get(behavior_id, [])
        if not q:
            return 0.0
        count = len(q)
        s = min(1.0, count / max(1.0, self.repetition_threshold))
        return float(s)

    def _score_context_drift(self, behavior_id: str) -> float:
        q = list(self.events.get(behavior_id, []))
        if len(q) < 2:
            return 0.0
        if behavior_id not in self.latest_context_anchor:
            return 0.0
        anchor = self.latest_context_anchor[behavior_id]
        sims = []
        for e in q:
            if e.context_vec is None:
                sims.append(0.0)
            else:
                sims.append(_cosine(anchor, e.context_vec))
        mean_sim = sum(sims) / max(1, len(sims))
        drift = 1.0 - mean_sim
        return float(min(1.0, drift))

    def _score_hallucination(self, behavior_id: str, verifier: Optional[Callable[[BehaviorEvent], float]] = None) -> float:
        q = self.events.get(behavior_id, [])
        if not q:
            return 0.0
        if verifier is None:
            return 0.0
        mismatch_scores = []
        for e in q:
            ver = verifier(e)
            mismatch = max(0.0, e.score - ver)
            mismatch_scores.append(mismatch)
        if not mismatch_scores:
            return 0.0
        mean_mismatch = sum(mismatch_scores) / len(mismatch_scores)
        return float(min(1.0, mean_mismatch))

    def assess_behavior(self, behavior_id: str, verifier: Optional[Callable[[BehaviorEvent], float]] = None) -> Dict[str, float]:
        bias = self._score_bias(behavior_id)
        repetition = self._score_repetition(behavior_id)
        context_drift = self._score_context_drift(behavior_id)
        hallucination = self._score_hallucination(behavior_id, verifier)
        confidence_mismatch = hallucination
        weights = self.score_weights
        total = (weights["bias"] * bias +
                 weights["repetition"] * repetition +
                 weights["context_drift"] * context_drift +
                 weights["hallucination"] * hallucination +
                 weights["confidence_mismatch"] * confidence_mismatch)
        maxw = sum(weights.values()) + 1e-9
        anomaly_score = float(total / maxw)
        result = {
            "bias": bias,
            "repetition": repetition,
            "context_drift": context_drift,
            "hallucination": hallucination,
            "confidence_mismatch": confidence_mismatch,
            "anomaly_score": anomaly_score
        }
        if anomaly_score > 0.5 and self.alert_callback:
            alert = {"behavior_id": behavior_id, "anomaly_score": anomaly_score, "details": result}
            try:
                self.alert_callback(alert)
            except Exception:
                pass
        return result

    def remediate(self, behavior_id: str, memory, actions: Optional[Dict[str, bool]] = None):
        actions = actions or {
            "quarantine": True,
            "decay_boost": True,
            "flag_for_review": True,
            "rollback": False
        }
        if actions.get("quarantine"):
            trace_ids = [e.trace_id for e in self.events.get(behavior_id, []) if e.trace_id]
            for tid in trace_ids:
                try:
                    memory.trigger_behavior(tid, context="quarantine", delta_N=-1.0, decay_rate=1.0)
                except Exception:
                    pass
        if actions.get("decay_boost"):
            for e in self.events.get(behavior_id, []):
                if e.trace_id:
                    try:
                        memory.trigger_behavior(e.trace_id, context="decay_boost", delta_N=-0.5, decay_rate=0.8)
                    except Exception:
                        pass
        if actions.get("flag_for_review") and self.alert_callback:
            self.alert_callback({"behavior_id": behavior_id, "action": "flag_for_review"})
        if actions.get("rollback"):
            if self.alert_callback:
                self.alert_callback({"behavior_id": behavior_id, "action": "rollback_requested"})

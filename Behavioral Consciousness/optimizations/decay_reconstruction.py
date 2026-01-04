"""
Decay and reconstruction policy utilities derived from BCE docs (secretkesifler).

Features:
- Adaptive decay rate using recency/frequency and optional Bayes bias.
- Reconstruction triggers based on hit_rate, drift_score, human_flag.
- Immutable critical guard snapshot helper.
"""

from typing import Dict, Any
import time
import math

EPS = 1e-12


def adaptive_decay(lambda_base: float, recency: float, freq: float, bayes_bias: float = 0.0) -> float:
    """
    lambda_base: nominal decay constant
    recency: [0,1] recent support (higher -> slower decay)
    freq: [0,1] frequency of hits (higher -> slower decay)
    bayes_bias: extra protection weight
    returns adjusted decay rate in [0, 2*lambda_base]
    """
    recency = float(max(0.0, min(1.0, recency)))
    freq = float(max(0.0, min(1.0, freq)))
    bias = float(max(0.0, bayes_bias))
    slow = (1.0 - 0.5*recency) * (1.0 - 0.5*freq)
    adj = lambda_base * slow * (1.0 - 0.3*bias)
    return float(max(0.0, min(2.0*lambda_base, adj)))


def decay_next(s_t: float, lambda_t: float, delta_t: float, support_boost: float = 0.0) -> float:
    """
    s_{t+1} = s_t * exp(-lambda_t * delta_t) + support_boost
    """
    return float(s_t * math.exp(-lambda_t * max(delta_t, 0.0)) + support_boost)


def reconstruction_trigger(hit_rate: float, drift_score: float, human_flag: bool, hit_thr: float = 0.3, drift_thr: float = 0.4) -> bool:
    return bool(hit_rate > hit_thr or drift_score > drift_thr or human_flag)


def reconstruction_gain(cluster_avg: float, eta: float = 0.1, s_min: float = 0.05) -> float:
    if cluster_avg <= 0.0:
        return 0.0
    return float(max(0.0, eta * cluster_avg + s_min))


def immutable_guard(snapshot_fn, item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    snapshot_fn should persist payload immutably; fallback to noop if None.
    """
    if callable(snapshot_fn):
        try:
            snapshot_fn(item_id, payload)
            return {"snapshot": True}
        except Exception as e:
            return {"snapshot": False, "error": str(e)}
    return {"snapshot": False, "reason": "no_fn"}


def policy_step(state: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates decay + possible reconstruction.
    state: {"score": float, "lambda_base": float}
    signals: {"recency":, "freq":, "bayes_bias":, "delta_t":, "support_boost":,
              "hit_rate":, "drift_score":, "human_flag":, "cluster_avg":, "eta":}
    """
    s = float(state.get("score", 1.0))
    lambda_base = float(state.get("lambda_base", 0.1))
    recency = float(signals.get("recency", 0.0))
    freq = float(signals.get("freq", 0.0))
    bayes_bias = float(signals.get("bayes_bias", 0.0))
    delta_t = float(signals.get("delta_t", 1.0))
    support_boost = float(signals.get("support_boost", 0.0))
    lambda_t = adaptive_decay(lambda_base, recency, freq, bayes_bias)
    s_next = decay_next(s, lambda_t, delta_t, support_boost)
    hit_rate = float(signals.get("hit_rate", 0.0))
    drift_score = float(signals.get("drift_score", 0.0))
    human_flag = bool(signals.get("human_flag", False))
    triggered = reconstruction_trigger(hit_rate, drift_score, human_flag)
    recon_gain = 0.0
    if triggered:
        eta = float(signals.get("eta", 0.1))
        cluster_avg = float(signals.get("cluster_avg", 0.0))
        recon_gain = reconstruction_gain(cluster_avg, eta)
        s_next += recon_gain
    return {
        "score": s_next,
        "lambda_t": lambda_t,
        "triggered": triggered,
        "recon_gain": recon_gain
    }

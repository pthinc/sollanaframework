# bce_anomali_plus.py
import numpy as np
from typing import Dict

ANOMALY_TYPES = ["obsessive", "avoidant", "norm_violation", "decay_driven"]

def classify_deviation(delta_vec: np.ndarray, D_vec: np.ndarray, context=None) -> Dict:
    """
    delta_vec: vector of per-dimension deltas normalized
    D_vec: decay contributions per dimension
    Returns dict with 'type', 'prob', 'severity'
    """
    # features
    mean_delta = float(np.mean(np.abs(delta_vec)))
    decay_mean = float(np.mean(D_vec))
    skew = float(np.mean(delta_vec))  # directionality
    # heuristic probabilities
    probs = {"obsessive": 0.0, "avoidant": 0.0, "norm_violation": 0.0, "decay_driven": 0.0}
    # obsessive: high mean_delta + low decay contribution + positive skew (repetition)
    probs["obsessive"] = max(0.0, min(1.0, mean_delta * (1.0 - decay_mean) * (1.0 + max(0.0, skew))))
    # avoidant: high mean_delta but negative skew and low norm match expected
    probs["avoidant"] = max(0.0, min(1.0, mean_delta * (1.0 - decay_mean) * (1.0 + max(0.0, -skew))))
    # norm_violation: if external context flag present or large direction change
    norm_flag = 1.0 if (context and context.get("norm_discrepancy", False)) else 0.0
    probs["norm_violation"] = max(0.0, min(1.0, mean_delta * (0.5 + 0.5*norm_flag)))
    # decay driven: high decay_mean dominates
    probs["decay_driven"] = max(0.0, min(1.0, decay_mean * (1.0 + mean_delta)))
    # normalize
    s = sum(probs.values()) + 1e-9
    for k in probs:
        probs[k] /= s
    # severity score
    severity = float(min(1.0, mean_delta * (1.0 + decay_mean)))
    top = max(probs.items(), key=lambda x: x[1])
    return {"type": top[0], "prob": top[1], "probs": probs, "severity": severity}

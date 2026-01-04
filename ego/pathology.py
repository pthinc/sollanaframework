# pathology.py
import math, time
from collections import deque

THETA_PATH = 0.75

def entropy(vec):
    import numpy as np
    p = np.asarray(vec)
    p = p / (p.sum() + 1e-9)
    return -float((p * (p+1e-12).log()).sum())

def pathology_score(phi_vec, phi_scalar, ethical_acceptance):
    ent = entropy(phi_vec)
    return float(phi_scalar * (1.0 - float(ethical_acceptance)) * ent)

def assess_and_remediate(behavior_id, phi_vec, phi_scalar, ethical_acceptance, memory, human_queue):
    score = pathology_score(phi_vec, phi_scalar, ethical_acceptance)
    if score >= THETA_PATH:
        memory.trigger_behavior(f"quarantine_{behavior_id}", context="pathology", delta_N=-1.0, decay_rate=1.0)
        human_queue.append({"behavior_id": behavior_id, "reason": "pathology", "score": score, "ts": time.time()})
        return {"action":"quarantine", "score": score}
    return {"action":"ok", "score": score}

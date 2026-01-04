# intent.py
import numpy as np

def intent_vector(p_from, p_to):
    d = np.array(p_to) - np.array(p_from)
    norm = np.linalg.norm(d) + 1e-12
    return d / norm, norm

def combined_intent(vectors):
    total = np.sum([v for v, _ in vectors], axis=0)
    mag = np.linalg.norm(total)
    return total, mag

# thought_gen.py
import random
import numpy as np

from xt_activation import x_t_np

def sample_R(seed=None, scale=0.02, novelty=0.5):
    rng = random.Random(seed)
    base = rng.uniform(-1.0, 1.0)
    return base * scale * (1.0 + novelty)

def produce_thought(t, intent_vec, seed=None, scale=0.02, novelty=0.5):
    xt = float(x_t_np(t))
    R = sample_R(seed=seed, scale=scale, novelty=novelty)
    I = np.linalg.norm(intent_vec) if isinstance(intent_vec, (list, np.ndarray)) else float(intent_vec)
    return xt * R * I

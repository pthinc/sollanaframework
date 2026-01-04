import math
import numpy as np
import time

EPS = 1e-9

def decay_scalar(t: float, lam: float) -> float:
    return 1.0 - math.exp(-lam * t)

def context_score(P: np.ndarray, W: np.ndarray, ages: np.ndarray, lam: float) -> np.ndarray:
    d = 1.0 - np.exp(-lam * ages)     # decay(t) = 1 - e^{-λt}
    return P * (1.0 - d) * W

# kullanımı
# P, W, ages aynı boyutta numpy dizileri
# theta_context örn 0.6

from scipy.special import expit as sigmoid

def time_modulator(age: float, tau: float) -> float:
    return 1.0 / (1.0 + math.exp((age - tau)))   # örnek zaman modülatörü

def izlek_alignment(phi: float, context_vec: np.ndarray, path_vec: np.ndarray) -> float:
    # kosinüs benzeri uyum skoru
    num = np.dot(context_vec, path_vec)
    den = (np.linalg.norm(context_vec) * np.linalg.norm(path_vec) + EPS)
    return max(0.0, min(1.0, num / den))

def normalize_B(B: float, age: float, phi: float, tau: float=10.0) -> float:
    return float(sigmoid(B) * time_modulator(age, tau) * izlek_alignment(phi, np.ones(3), np.ones(3)))

def meaning_score(phi: float, B: float, age: float, E: float) -> float:
    normB = normalize_B(B, age, phi)
    return phi * normB * float(E)

def x_t_scalar(t: float, clamp: float=20.0) -> float:
    t_c = min(t, clamp)
    return math.tanh(math.exp(t_c) - math.pi)

def random_variation(seed: int=None, scale: float=0.05, shape=1):
    rng = np.random.RandomState(seed)
    return rng.normal(scale=scale, size=shape)

def discovery_energy(t: float, seed: int=None, scale: float=0.05):
    return x_t_scalar(t) * float(np.abs(random_variation(seed=seed, scale=scale)[0]))

def process_behavior(behavior_id, P, W, age, phi, lam, ethical_acceptance, phi_min, theta_context):
    B = context_score(np.array([P]), np.array([W]), np.array([age]), lam)[0]
    if (1.0 - math.exp(-lam * age)) >= theta_context:
        return {"status":"context_exhausted"}
    E = 1.0 if ethical_acceptance else 0.0
    M = meaning_score(phi, B, age, E)
    if M < phi_min:
        return {"status":"decay_candidate", "meaning": M}
    return {"status":"accepted", "meaning": M}


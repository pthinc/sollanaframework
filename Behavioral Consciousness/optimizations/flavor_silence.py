# flavor_silence.py
import time, math, random
from typing import Dict, List, Tuple
import numpy as np

EPS = 1e-12
PI = math.pi

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS; nb = np.linalg.norm(b) + EPS
    return float(np.dot(a,b) / (na * nb))

def resonance(f_vec: np.ndarray, c_vec: np.ndarray) -> float:
    return cos_sim(f_vec, c_vec)

def resonance_gradient_norm(f_vec: np.ndarray, c_vec: np.ndarray, eps: float = 1e-3, n_directions: int = 8) -> float:
    """
    Numerical estimate of ||∂Res/∂c|| by sampling random unit directions u and central difference.
    Returns average L2 norm across directions (robust proxy).
    """
    c = np.asarray(c_vec, dtype=float)
    norms = []
    for _ in range(n_directions):
        u = np.random.randn(c.size)
        u = u / (np.linalg.norm(u) + EPS)
        r_plus = resonance(f_vec, c + eps * u)
        r_minus = resonance(f_vec, c - eps * u)
        deriv = (r_plus - r_minus) / (2.0 * eps)  # directional derivative
        norms.append(abs(deriv))
    # aggregate directional derivatives into scalar sensitivity
    return float(np.mean(norms))

class FlavorSelector:
    def __init__(self, flavor_prototypes: Dict[str, np.ndarray], superego_check_fn=None):
        """
        flavor_prototypes: {flavor_name: embedding_vector}
        superego_check_fn(token_payload)-> bool returns True if allowed
        """
        self.protos = {k: np.asarray(v, dtype=float) for k,v in flavor_prototypes.items()}
        self.superego_check_fn = superego_check_fn

    def choose_flavor(self, context_vec: np.ndarray) -> Tuple[str, float, float]:
        best = None; best_score = -1.0; best_R0 = 0.0
        for name, fvec in self.protos.items():
            score = resonance_gradient_norm(fvec, context_vec)
            R0 = resonance(fvec, context_vec)
            # optional superego gating (skip if disallowed)
            if callable(self.superego_check_fn):
                allowed = self.superego_check_fn({"flavor":name, "R0":R0})
                if not allowed:
                    continue
            if score > best_score:
                best_score = score; best = name; best_R0 = R0
        return best, best_score, best_R0

class SilenceTampon:
    def __init__(self, selector: FlavorSelector, tone_sigma: float = 0.02, tone_clip: float = 0.08):
        self.selector = selector
        self.tone_sigma = tone_sigma
        self.tone_clip = tone_clip

    def sample_tone(self, flavor_name: str) -> float:
        # controlled variance per flavor; could be per-flavor tuned
        delta = random.gauss(0.0, self.tone_sigma)
        delta = max(-self.tone_clip, min(self.tone_clip, delta))
        return float(delta)

    def handle_silence(self, context_vec: np.ndarray, approval: float, critical: float, mu_before: float = 0.0) -> Dict:
        F0, sens_score, R0 = self.selector.choose_flavor(context_vec)
        if F0 is None:
            # fallback neutral
            return {"flavor": "neutral", "applied": False}
        tone_delta = self.sample_tone(F0)
        # compute tampon value
        tampon = float(approval * (1.0 - critical) + 0.0)  # flavor contribution separate
        psi_open = float(PI * R0 + tone_delta + mu_before)
        # package action
        out = {
            "flavor": F0,
            "sensitivity": sens_score,
            "R0": R0,
            "tone_delta": tone_delta,
            "tampon": tampon,
            "psi_open": psi_open,
            "ts": time.time()
        }
        return out

# flavor_treatment.py
"""
Flavor Tedavi Optimizasyonları
- ExpandedFlavor: çoklu flavor fonksiyonlarının ağırlıklı birleşimi
- AdaptivePIDTampon: decay tamponunu PID ile gerçek zamanlı ayarlar
- VSubsetHealth: alt-kümeler üzerinden sağlık skoru hesaplar
- HybridFlavorController: adaptif + preset flavor karışımı
- Örnek entegrasyon ve hızlı test
Requirements: numpy
"""

import time, math, random
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np

EPS = 1e-12
PI = math.pi

# -------------------------
# Expanded Flavor
# -------------------------
class ExpandedFlavor:
    def __init__(self, flavor_funcs: Dict[str, Callable[[np.ndarray, Dict], float]], initial_weights: Optional[Dict[str, float]] = None):
        """
        flavor_funcs: name -> function(context_vec, meta) -> scalar contribution
        initial_weights: name -> w_k initial
        """
        self.flavor_funcs = flavor_funcs
        self.weights = {k: (initial_weights.get(k, 1.0) if initial_weights else 1.0) for k in flavor_funcs.keys()}
        self.normalize_weights()

    def normalize_weights(self):
        s = sum(abs(v) for v in self.weights.values()) + EPS
        for k in list(self.weights.keys()):
            self.weights[k] = float(self.weights[k]) / s

    def compute(self, context_vec: np.ndarray, emotion_meta: Dict) -> Dict[str, float]:
        contributions = {}
        for name, fn in self.flavor_funcs.items():
            try:
                contributions[name] = float(fn(context_vec, emotion_meta))
            except Exception:
                contributions[name] = 0.0
        # weighted sum
        total = 0.0
        for name, c in contributions.items():
            total += self.weights.get(name, 0.0) * c
        return {"total": float(total), "contributions": contributions, "weights": dict(self.weights)}

    def adapt_weights(self, delta_map: Dict[str, float], lr: float = 0.01):
        for k, delta in delta_map.items():
            if k in self.weights:
                self.weights[k] = max(0.0, self.weights[k] + lr * float(delta))
        self.normalize_weights()

# -------------------------
# Adaptive PID Tampon
# -------------------------
class AdaptivePIDTampon:
    def __init__(self, kp: float = 0.6, ki: float = 0.32, kd: float = 0.405, dt: float = 1.0, integrator_max: float = 10.0):
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.dt = float(dt)
        self.integrator = 0.0
        self.prev_error = 0.0
        self.integrator_max = float(integrator_max)

    def step(self, error: float) -> float:
        self.integrator += error * self.dt
        self.integrator = max(-self.integrator_max, min(self.integrator, self.integrator_max))
        derivative = (error - self.prev_error) / (self.dt + EPS)
        out = self.kp * error + self.ki * self.integrator + self.kd * derivative
        self.prev_error = error
        return float(out)

    def tune(self, reward: float, lr: float = 0.005):
        # small safe updates from self-reward signal
        self.kp = max(0.0, min(5.0, self.kp + lr * reward))
        self.ki = max(0.0, min(1.0, self.ki + lr * 0.1 * reward))
        self.kd = max(0.0, min(1.0, self.kd + lr * 0.01 * reward))

# -------------------------
# V Subset Health Scorer
# -------------------------
class VSubsetHealth:
    def __init__(self, subset_map: Dict[str, List[str]], flavor_resonance_fn: Callable[[str, np.ndarray, Dict], float]):
        """
        subset_map: name -> list of flavor names in subset V_j
        flavor_resonance_fn: (flavor_name, context_vec, meta) -> resonance scalar
        """
        self.subset_map = subset_map
        self.flavor_resonance_fn = flavor_resonance_fn

    def score(self, context_vec: np.ndarray, char_vec: np.ndarray, meta: Dict) -> Dict[str, float]:
        out = {}
        for vname, flavors in self.subset_map.items():
            vals = []
            for f in flavors:
                r = float(self.flavor_resonance_fn(f, context_vec, meta))
                # sensitivity to context via local numeric derivative proxy
                eps = 1e-3
                # directional proxy: small perturbation in char_vec
                pert = char_vec + eps * (np.random.randn(*char_vec.shape))
                r2 = float(self.flavor_resonance_fn(f, pert, meta))
                dr_dc = (r2 - r) / eps
                vals.append(r * abs(dr_dc))
            out[vname] = float(max(vals) if vals else 0.0)
        return out

    def best_subset(self, context_vec: np.ndarray, char_vec: np.ndarray, meta: Dict) -> Tuple[str, float]:
        scores = self.score(context_vec, char_vec, meta)
        if not scores:
            return "", 0.0
        best = max(scores.items(), key=lambda x: x[1])
        return best

# -------------------------
# Hybrid Control Controller
# -------------------------
class HybridFlavorController:
    def __init__(self, adaptive: ExpandedFlavor, preset: ExpandedFlavor, gamma_init: float = 0.7):
        self.adaptive = adaptive
        self.preset = preset
        self.gamma = float(gamma_init)  # 0..1, higher => adaptive more weight

    def compute(self, context_vec: np.ndarray, meta: Dict) -> Dict:
        a = self.adaptive.compute(context_vec, meta)
        p = self.preset.compute(context_vec, meta)
        total = float(self.gamma * a["total"] + (1.0 - self.gamma) * p["total"])
        return {"total": total, "adaptive": a, "preset": p, "gamma": float(self.gamma)}

    def update_gamma(self, user_acceptance: float, lr: float = 0.01):
        # increase gamma when acceptance high, otherwise decay to preset
        self.gamma = max(0.0, min(1.0, self.gamma + lr * (user_acceptance - 0.5)))

# -------------------------
# Example integration usage
# -------------------------
if __name__ == "__main__":
    # toy flavor funcs
    def humor(context, meta): return max(0.0, 0.8 * (np.tanh(np.mean(context)) + 0.1))
    def play(context, meta): return max(0.0, 0.6 * (1.0 - abs(np.tanh(np.mean(context)))))
    def care(context, meta): return 0.5 + 0.2 * meta.get("safety", 0.0)

    flavor_funcs = {"humor": lambda c,m: humor(c,m), "play": lambda c,m: play(c,m), "care": lambda c,m: care(c,m)}
    adaptive = ExpandedFlavor(flavor_funcs, {"humor":1.0,"play":0.5,"care":0.3})
    preset = ExpandedFlavor(flavor_funcs, {"humor":0.2,"play":0.2,"care":1.0})

    hybrid = HybridFlavorController(adaptive, preset, gamma_init=0.7)
    pid = AdaptivePIDTampon(kp=0.5, ki=0.2, kd=0.05, dt=1.0)
    vsub = VSubsetHealth({"light": ["humor","play"], "soft": ["care"]},
                         flavor_resonance_fn=lambda fname, c, m: adaptive.compute(c,m)["contributions"].get(fname,0.0))

    ctx = np.random.randn(128)
    meta = {"safety":0.9}
    print("Hybrid flavor before:", hybrid.compute(ctx, meta))
    # simulate a silence tampon application
    res = hybrid.compute(ctx, meta)
    desired = 1.0
    error = (pid.step(desired - res["total"]))
    print("PID tampon output:", error)
    # update weights on fake reward
    reward = 0.2
    adaptive.adapt_weights({"humor": 0.1, "play": 0.05}, lr=0.1)
    hybrid.update_gamma(user_acceptance=0.8, lr=0.02)
    print("Hybrid flavor after:", hybrid.compute(ctx, meta))

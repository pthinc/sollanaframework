# reflex_module.py
import time, random
import math
import numpy as np

class MicroRLReflex:
    def __init__(self, arms=None, lr=0.05, eps=0.1, max_updates_per_window=5):
        # arms: candidate epsilon values (speedups) e.g. [0.0, 0.05, 0.1, 0.15]
        self.arms = arms or [0.0, 0.05, 0.1, 0.15]
        self.n = len(self.arms)
        self.values = np.ones(self.n, dtype=float)
        self.counts = np.zeros(self.n, dtype=int)
        self.lr = float(lr)
        self.eps = float(eps)
        self.updates_done = 0
        self.max_updates = int(max_updates_per_window)

    def choose_arm(self):
        if random.random() < self.eps:
            return random.randrange(self.n)
        return int(np.argmax(self.values))

    def update(self, arm_idx, reward):
        if self.updates_done >= self.max_updates:
            return
        self.counts[arm_idx] += 1
        self.values[arm_idx] += self.lr * (reward - self.values[arm_idx])
        self.updates_done += 1

    def reset_window(self):
        self.updates_done = 0

class ReflexController:
    def __init__(self, base_tau=0.6, tau_min=0.05, tau_max=2.0, rl_agent=None):
        self.base_tau = float(base_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.rl = rl_agent or MicroRLReflex()
    def detect_stress(self, behavior_meta):
        # örnek: decay_level yüksek veya context_error yüksek
        decay = float(behavior_meta.get("decay_level", 0.0))
        ctx_err = float(behavior_meta.get("context_error", 0.0))
        stress = (decay * 0.7 + ctx_err * 0.3)  # ağırlıklı skor
        return stress > 0.2, stress
    def apply_reflex(self, behavior_meta):
        stressed, stress_score = self.detect_stress(behavior_meta)
        if not stressed:
            return self.base_tau, {"epsilon": 0.0, "stressed": False}
        arm = self.rl.choose_arm()
        eps = float(self.rl.arms[arm])
        new_tau = max(self.tau_min, min(self.tau_max, self.base_tau * (1.0 - eps)))
        return new_tau, {"epsilon": eps, "arm": arm, "stressed": True, "stress_score": stress_score}
    def learn(self, arm_idx, reward):
        self.rl.update(arm_idx, reward)

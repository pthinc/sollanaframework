# obsession_control.py
import time, math, numpy as np

EPS = 1e-12

def compute_delta_D(kappa, Omega, eta, E, G, S_n):
    return float(kappa * (1.0 - Omega) * eta * E - G * S_n)

class ObsessionController:
    def __init__(self, kappa=0.5, Omega=0.2, G=1.0, theta_soft=0.02, theta_hard=0.1):
        self.kappa = float(kappa)
        self.Omega = float(Omega)
        self.G = float(G)
        self.theta_soft = float(theta_soft)
        self.theta_hard = float(theta_hard)
        self.snapshot_rate = 1.0     # snapshots per minute baseline
        self.last_snapshot_ts = {}

    def evaluate_trace(self, trace):
        eta = float(trace.get("eta", 1.0))
        E = float(trace.get("ethical", 1.0))
        S_n = float(trace.get("snapshot_count", 0.0))
        d = compute_delta_D(self.kappa, self.Omega, eta, E, self.G, S_n)
        return d

    def adapt_params(self, trace, delta_D):
        # small-step adaptive policy
        if delta_D < -self.theta_hard:
            self.Omega = max(0.0, self.Omega * 0.95)   # decrease obsession slowly
            self.kappa = min(2.0, self.kappa * 1.05)   # boost exploration
            self.G = max(0.1, self.G * 0.95)           # reduce gravity
            action = "hard_remediate"
        elif delta_D < -self.theta_soft:
            self.kappa *= 1.02
            action = "soft_sandbox"
        else:
            action = "ok"
        return action

    def snapshot_allowed(self, user_id, now=None, cooldown_sec=60.0):
        now = now or time.time()
        last = self.last_snapshot_ts.get(user_id, 0)
        if now - last < cooldown_sec * (1.0 / max(0.1, self.snapshot_rate)):
            return False
        self.last_snapshot_ts[user_id] = now
        return True

    def throttle_snapshot_rate(self, factor):
        # reduce snapshot_rate multiplicatively
        self.snapshot_rate = max(0.01, self.snapshot_rate * factor)

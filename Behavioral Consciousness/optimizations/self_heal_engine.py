# self_heal_engine.py
"""
Monoton Code Self-Healing Reward Engine - prototype
Dependencies: numpy
"""

import time, math, json, heapq
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

EPS = 1e-12

# -------------------------
# Helpers
# -------------------------
def now_ts() -> float:
    return time.time()

def mean_std(arr: List[float]) -> Tuple[float,float]:
    if not arr:
        return 0.0, 0.0
    a = np.array(arr, dtype=float)
    return float(a.mean()), float(a.std())

# -------------------------
# Sliding window anomaly detector
# -------------------------
class SlidingAnomalyDetector:
    def __init__(self, window_size: int = 100, lambda_thresh: float = 2.0):
        self.window_size = int(window_size)
        self.lambda_thresh = float(lambda_thresh)
        self.buffer: List[float] = []

    def push(self, value: float):
        self.buffer.append(float(value))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_anomaly(self, value: float) -> bool:
        if len(self.buffer) < max(4, int(self.window_size*0.2)):
            return False
        mu, sigma = mean_std(self.buffer)
        return value < (mu - self.lambda_thresh * sigma)

# -------------------------
# PID Controller for adaptive thresholds and gains
# -------------------------
class PID:
    def __init__(self, kp=0.1, ki=0.01, kd=0.01, dt=1.0, integrator_max=10.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.dt = float(dt)
        self.integrator = 0.0; self.prev_err = 0.0
        self.integrator_max = integrator_max

    def step(self, error: float) -> float:
        self.integrator += error * self.dt
        # anti-windup
        self.integrator = max(-self.integrator_max, min(self.integrator, self.integrator_max))
        deriv = (error - self.prev_err) / (self.dt + 1e-12)
        out = self.kp*error + self.ki*self.integrator + self.kd*deriv
        self.prev_err = error
        return out

    def tune_by_reward(self, reward: float, lr: float = 0.01):
        # small adaptive update
        self.kp += lr * reward
        self.ki += lr * 0.1 * reward
        self.kd += lr * 0.01 * reward
        # safe clamps
        self.kp = max(0.0, min(5.0, self.kp))
        self.ki = max(0.0, min(1.0, self.ki))
        self.kd = max(0.0, min(1.0, self.kd))

# -------------------------
# Blueprint store and selector
# -------------------------
class BlueprintStore:
    def __init__(self):
        self.blueprints: Dict[str, Dict[str,Any]] = {}  # id -> metadata

    def register(self, id: str, metadata: Dict[str,Any]):
        self.blueprints[id] = metadata

    def select_best_by_resonance(self) -> Optional[str]:
        best = None; best_score = -1e9
        for id, meta in self.blueprints.items():
            r = float(meta.get("resonance", 0.0))
            if r > best_score:
                best_score = r; best = id
        return best

# -------------------------
# Flavor and tag prior updater
# -------------------------
class PriorsManager:
    def __init__(self, priors: Optional[Dict[str,float]] = None):
        self.priors = priors or {}

    def update(self, delta_map: Dict[str,float], normalize: bool = True):
        for k,v in delta_map.items():
            self.priors[k] = max(0.0, self.priors.get(k, 0.0) + float(v))
        if normalize:
            s = sum(self.priors.values()) + EPS
            for k in list(self.priors.keys()):
                self.priors[k] = self.priors[k] / s

# -------------------------
# Self-Heal Engine
# -------------------------
class SelfHealEngine:
    def __init__(self,
                 anomaly_window: int = 120,
                 lambda_thresh: float = 2.0,
                 pid: Optional[PID] = None,
                 blueprint_store: Optional[BlueprintStore] = None,
                 priors: Optional[PriorsManager] = None):
        self.detector = SlidingAnomalyDetector(window_size=anomaly_window, lambda_thresh=lambda_thresh)
        self.pid = pid or PID()
        self.blueprints = blueprint_store or BlueprintStore()
        self.priors = priors or PriorsManager()
        self.history: List[Dict[str,Any]] = []  # time series of signals
        self.last_checkpoint = None

    def observe(self, psi_opt: float, net_affect: float, context_drift: float, error_rate: float):
        ts = now_ts()
        sample = {"ts":ts, "psi":psi_opt, "affect":net_affect, "drift":context_drift, "errors":error_rate}
        self.history.append(sample)
        # push a combined health score to anomaly detector
        health = psi_opt - (0.5*context_drift + 0.5*error_rate)
        self.detector.push(health)
        return sample

    def check_and_heal(self):
        if not self.history:
            return {"action":"noop"}
        last = self.history[-1]
        health = last["psi"] - (0.5*last["drift"] + 0.5*last["errors"])
        if self.detector.is_anomaly(health):
            # anomaly path
            # checkpoint rollback: select last good checkpoint if exists
            action_log = {"action":"anomaly_detected", "health":health}
            # select alternative blueprint
            candidate_bp = self.blueprints.select_best_by_resonance()
            action_log["selected_blueprint"] = candidate_bp
            # compute pre/post Psi snapshot: simulate apply blueprint (external hook)
            # user must provide apply_blueprint hook integration
            return action_log
        return {"action":"healthy", "health":health}

    def apply_reward_and_update(self, psi_before: float, psi_after: float, affect_before: float, affect_after: float, delta_priors_map: Dict[str,float]):
        dpsi = psi_after - psi_before
        daff = affect_after - affect_before
        # reward
        gamma1 = 1.0; gamma2 = 0.7
        R = gamma1 * dpsi + gamma2 * daff
        # update PID gains modestly
        self.pid.tune_by_reward(R, lr=0.005)
        # update priors
        delta_map = {k: (0.01 * R * v) for k,v in delta_priors_map.items()}
        self.priors.update(delta_map)
        return {"R":R, "pid": {"kp":self.pid.kp, "ki":self.pid.ki, "kd":self.pid.kd}, "priors": self.priors.priors}

    # grid search for alpha,beta (loss function expects history of R_t)
    def grid_search_alpha_beta(self, alpha_grid: List[float], beta_grid: List[float], target_recovery_series: List[float], observed_R_series: List[float]):
        best = None; best_loss = float("inf")
        for a in alpha_grid:
            for b in beta_grid:
                # simple param effect simulation: predicted = a*observed + b*other (placeholder)
                pred = [a*r + b*0.0 for r in observed_R_series]
                loss = sum((t - p)**2 for t,p in zip(target_recovery_series, pred))
                if loss < best_loss:
                    best_loss = loss; best = (a,b,loss)
        return {"best_alpha":best[0], "best_beta":best[1], "loss":best[2]}

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    engine = SelfHealEngine()
    engine.blueprints.register("bp1", {"resonance":0.2})
    engine.blueprints.register("bp2", {"resonance":0.8})
    # simulate observations
    for i in range(200):
        psi = 0.6 + 0.01*math.sin(i/10.0)
        affect = 0.5 + 0.02*math.cos(i/15.0)
        drift = 0.05*(1.0 if i%50==0 else 0.0)
        errors = 0.01*(1.0 if i%70==0 else 0.0)
        engine.observe(psi, affect, drift, errors)
        if i % 30 == 0:
            print(engine.check_and_heal())
    # reward update example
    res = engine.apply_reward_and_update(psi_before=0.6, psi_after=0.7, affect_before=0.5, affect_after=0.6, delta_priors_map={"p1":1.0, "p2":0.5})
    print(res)

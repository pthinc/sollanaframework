# behavior_health_optimizer.py
import math, time, random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from scipy.fft import rfft, rfftfreq

EPS = 1e-12

class BehaviorCandidate:
    def __init__(self, id: str, eta: float, E: float, S_n: float, time_series: Optional[np.ndarray]=None):
        self.id = id
        self.eta = float(eta)
        self.E = float(E)
        self.S_n = float(S_n)
        self.time_series = np.asarray(time_series) if time_series is not None else None

class BehaviorHealthOptimizer:
    def __init__(self,
                 kappa_fn=lambda t: 0.5,
                 omega_fn=lambda t: 0.3,
                 G: float = 1.0,
                 xi_scale: float = 0.01):
        self.kappa_fn = kappa_fn
        self.omega_fn = omega_fn
        self.G = float(G)
        self.xi_scale = float(xi_scale)

    def psi_score(self, cand: BehaviorCandidate, t: float) -> float:
        k = float(self.kappa_fn(t))
        o = float(self.omega_fn(t))
        xi = random.gauss(0.0, self.xi_scale)
        return k * cand.eta * cand.E - o * self.G * cand.S_n + xi

    def monte_carlo_select(self, cands: List[BehaviorCandidate], n_iters: int = 1024, seed: Optional[int]=None) -> Tuple[BehaviorCandidate, float]:
        rng = random.Random(seed)
        best = None; best_score = -1e9
        t = time.time()
        for _ in range(n_iters):
            cand = rng.choice(cands)
            s = self.psi_score(cand, t)
            if s > best_score:
                best_score = s; best = cand
        return best, best_score

    def fft_peak(self, series: np.ndarray, sample_rate: float = 1.0) -> Tuple[float, float]:
        # returns (peak_freq, peak_amplitude)
        if series is None or len(series) < 8:
            return 0.0, 0.0
        N = len(series)
        yf = np.abs(rfft(series - np.mean(series)))
        xf = rfftfreq(N, 1.0 / sample_rate)
        idx = np.argmax(yf[1:]) + 1
        return float(xf[idx]), float(yf[idx])

    def ziegler_nichols_pid(self, Ku: float, Tu: float) -> Tuple[float,float,float]:
        # classic PID tuning
        Kp = 0.6 * Ku
        Ki = 2.0 * Kp / Tu
        Kd = Kp * Tu / 8.0
        return Kp, Ki, Kd

    def discover_Ku_Tu_sim(self, sim_fn, kp_grid: List[float], obs_time: float = 20.0, dt: float = 0.1) -> Tuple[Optional[float],Optional[float]]:
        # sim_fn(kp, dt, obs_time) -> list of y samples
        for kp in kp_grid:
            ys = sim_fn(kp, dt, obs_time)
            y = np.asarray(ys)
            mean = y.mean()
            # count zero-crossings around mean to detect oscillation
            zc = ((y[:-1]-mean)*(y[1:]-mean) < 0).sum()
            if zc > 6:
                # estimate period from FFT
                xf = np.fft.rfftfreq(len(y), dt)
                yf = np.abs(np.fft.rfft(y - mean))
                idx = np.argmax(yf[1:]) + 1
                Tu = 1.0 / xf[idx] if xf[idx] > 0 else obs_time/10.0
                return float(kp), float(Tu)
        return None, None

    def superego_priority(self, context_importance: float, drift_risk: float, alpha: float=0.7, beta: float=0.3) -> float:
        return alpha * float(context_importance) + beta * float(drift_risk)

    def decay_score(self, buffer_level: float, error_rate: float, gamma1: float=0.6, gamma2: float=0.4) -> float:
        return gamma1*buffer_level + gamma2*error_rate

    def decide_rollback(self, buffer_level: float, error_rate: float, delta_thresh: float=0.6) -> bool:
        return self.decay_score(buffer_level, error_rate) > delta_thresh

# Example sim_fn for Ku/Tu detection
def example_sim_fn(kp, dt, obs_time):
    # toy closed-loop simulated y(t) = sin(0.5*t) scaled by (1 + kp*0.1) with added dynamics to provoke oscillation
    steps = int(obs_time/dt)
    y = []
    val = 0.0
    for i in range(steps):
        t = i*dt
        # simple second-order pseudo-system
        val = 0.9*val + 0.1*math.sin(0.5*t)*(1 + 0.1*kp) + 0.01*random.gauss(0,1)
        y.append(val)
    return y

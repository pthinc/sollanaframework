# pleasure_pain_controller.py
import time, math
import numpy as np

EPS = 1e-12

class MicroPID:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, dt=1.0, integ_max=5.0, deriv_tau=0.2):
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.dt = float(dt)
        self.integ = 0.0
        self.prev_e = 0.0
        self.integ_max = float(integ_max)
        self.deriv = 0.0
        self.tau = float(deriv_tau)

    def step(self, e):
        # P
        P = self.kp * e
        # I with anti-windup
        self.integ += e * self.dt
        self.integ = max(-self.integ_max, min(self.integ, self.integ_max))
        I = self.ki * self.integ
        # D with first-order low-pass filter
        raw_d = (e - self.prev_e) / (self.dt + EPS)
        alpha = self.dt / (self.tau + self.dt)
        self.deriv = (1 - alpha) * self.deriv + alpha * raw_d
        D = self.kd * self.deriv
        self.prev_e = e
        u = P + I + D
        return u, {"P":P, "I":I, "D":D, "u":u}

    def adapt_gains(self, reward, lr=0.005, kp_bounds=(0.0,5.0), ki_bounds=(0.0,1.0), kd_bounds=(0.0,1.0)):
        # small safe updates proportional to reward
        self.kp = min(kp_bounds[1], max(kp_bounds[0], self.kp + lr * reward))
        self.ki = min(ki_bounds[1], max(ki_bounds[0], self.ki + lr * 0.1 * reward))
        self.kd = min(kd_bounds[1], max(kd_bounds[0], self.kd + lr * 0.01 * reward))

# Pleasure–Pain orchestrator
class PleasurePainController:
    def __init__(self, pid=None, s_opt=0.0):
        self.pid = pid or MicroPID()
        self.s_opt = float(s_opt)
        self.last_ts = time.time()

    def measure_S(self, desire, pleasure, pain):
        return float(desire - pleasure + pain)

    def control_step(self, desire, pleasure, pain, feedback_mu=0.0, context_meta=None):
        now = time.time()
        dt = max(1e-3, now - self.last_ts)
        self.pid.dt = dt
        S = self.measure_S(desire, pleasure, pain)
        e = S - self.s_opt
        u, comps = self.pid.step(e)   # control output acts as tampon delta
        # apply minimal saturation and dosing
        tampon_delta = float(max(-1.0, min(1.0, u)))
        # internal reward computed after environment response; placeholder here
        # return control action and diagnostics
        self.last_ts = now
        return {"S":S, "error":e, "tampon_delta":tampon_delta, "pid":comps}

    def apply_reward(self, delta_psi, delta_affect):
        gamma1, gamma2 = 1.0, 0.7
        R = gamma1 * delta_psi + gamma2 * delta_affect
        self.pid.adapt_gains(R)
        return R

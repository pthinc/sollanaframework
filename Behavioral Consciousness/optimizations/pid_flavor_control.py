# pid_flavor_control.py
import time, math
import numpy as np

PHI = 1.6180339887498948
EPS = 1e-12

def cos_sim(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS; nb = np.linalg.norm(b) + EPS
    return float(np.dot(a,b)/(na*nb))

def amplitude(psi_vec, norm_vec, n):
    sim = max(0.0, min(1.0, (cos_sim(psi_vec, norm_vec)+1.0)/2.0))
    return (PHI ** float(n)) * sim

def frequency(R_ij):
    return math.pi * PHI * max(0.0, min(1.0, 1.0 - float(R_ij)))

def phase_from_gaps(norm_gaps):
    return float(sum(norm_gaps))

def balance(approval, critical, flavor, approval_weight):
    a_term = approval_weight * float(approval) * (1.0 - float(critical))
    f_term = (1.0 - approval_weight) * float(flavor)
    return max(0.0, a_term + f_term)

def salinum(t, psi_vec, norm_vec, n, R_ij, norm_gaps, approval, critical, flavor, approval_weight):
    A = amplitude(psi_vec, norm_vec, n)
    B = frequency(R_ij)
    C = phase_from_gaps(norm_gaps)
    D = balance(approval, critical, flavor, approval_weight)
    return float(A * math.sin(B * float(t) + C) + D)

class PIDController:
    def __init__(self, Kp=0.72, Ki=0.32, Kd=0.405, dt=0.1, integrator_max=10.0, deriv_filter_tau=0.01):
        self.Kp = float(Kp); self.Ki = float(Ki); self.Kd = float(Kd)
        self.dt = float(dt)
        self.integrator = 0.0
        self.prev_error = 0.0
        self.integrator_max = float(integrator_max)
        # derivative low-pass filter state (TD)
        self.deriv_filtered = 0.0
        self.tau = float(deriv_filter_tau)

    def reset(self):
        self.integrator = 0.0
        self.prev_error = 0.0
        self.deriv_filtered = 0.0

    def step(self, error):
        # P
        P = self.Kp * error
        # I with anti-windup clamping
        self.integrator += error * self.dt
        self.integrator = float(max(-self.integrator_max, min(self.integrator, self.integrator_max)))
        I = self.Ki * self.integrator
        # raw derivative
        raw_d = (error - self.prev_error) / (self.dt + 1e-12)
        # derivative low-pass filter (first order)
        alpha = self.dt / (self.tau + self.dt)
        self.deriv_filtered = (1 - alpha) * self.deriv_filtered + alpha * raw_d
        D = self.Kd * self.deriv_filtered
        self.prev_error = float(error)
        # control output
        u = P + I + D
        return u, {"P":P, "I":I, "D":D, "u":u}

# Example closed-loop wrapper that uses PID to adjust approval_weight (0..1) to track target sal
class FlavorPIDLoop:
    def __init__(self, pid: PIDController, clamp=(0.0,1.0)):
        self.pid = pid
        self.clamp_min, self.clamp_max = clamp

    def run_episode(self, duration_s, dt,
                    psi_vec, norm_vec, n, R_ij, norm_gaps,
                    initial_approval, critical, flavor,
                    target_func):
        t = 0.0
        approval = float(initial_approval)
        history = []
        self.pid.reset()
        steps = int(max(1, math.ceil(duration_s / dt)))
        for i in range(steps):
            y = salinum(t, psi_vec, norm_vec, n, R_ij, norm_gaps, approval, critical, flavor, approval)
            y_target = float(target_func(t))
            error = y_target - y
            u, comps = self.pid.step(error)
            # map control u to delta approval (small step), then clamp
            approval += 0.01 * u  # scaling factor controls actuator sensitivity
            approval = max(self.clamp_min, min(self.clamp_max, approval))
            history.append({"t":t, "y":y, "y_target":y_target, "error":error, "approval":approval, "pid":comps})
            t += dt
        return history

def discover_Ku_Tu(sim_loop_fn, Kp_values, observe_window=60.0, dt=0.1):
    """
    sim_loop_fn(Kp) must run a short closed-loop simulation with given Kp and return y(t) list.
    This function increases Kp over Kp_values and detects sustained oscillation:
    find first Kp that produces persistent oscillation -> Ku, measure period Tu.
    """
    for Kp in Kp_values:
        ys = sim_loop_fn(Kp)  # list of (t,y) pairs or just y values sampled at dt
        y_vals = np.array([v for (_,v) in ys])
        # detect zero-crossing count per window -> oscillation approximate
        mean = y_vals.mean()
        sign_changes = ((y_vals[:-1]-mean)*(y_vals[1:]-mean) < 0).sum()
        if sign_changes > 6:  # heuristic: sustained oscillation threshold
            # estimate period Tu from FFT peak or zero-crossing intervals
            times = np.array([t for (t,_) in ys])
            crossings = []
            for i in range(len(y_vals)-1):
                if (y_vals[i]-mean)*(y_vals[i+1]-mean) < 0:
                    crossings.append(times[i])
            if len(crossings) >= 2:
                intervals = np.diff(crossings)
                Tu = float(np.median(intervals)*2)  # step to period estimate
            else:
                Tu = observe_window / 10.0
            return float(Kp), float(Tu)
    return None, None

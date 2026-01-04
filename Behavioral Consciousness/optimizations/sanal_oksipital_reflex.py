# sanal_oksipital_reflex.py
"""
Sanal Oksipital: Self-Reward Reflex + Entropy-Penalized Optimization + Decay/Drift Reflex
Replace hooks (snapshot_fn, rollback_fn, neutralize_flavor_fn, telemetry_fn, notify_user_fn) with real implementations.
"""
import time, math, json, tempfile, os
from typing import Dict, Any, Callable, Optional, List
import numpy as np

EPS = 1e-12
PI = math.pi

# ----------------------------
# Hooks (replace with your system)
# ----------------------------
def snapshot_fn(tag: str) -> str:
    # create a snapshot, return id (default logs locally)
    snap = f"snapshot_{int(time.time()*1000)}"
    telemetry_fn({"event":"snapshot", "id": snap, "tag": tag})
    return snap

def rollback_fn(snapshot_id: str) -> bool:
    # perform rollback to snapshot_id
    return True

def neutralize_flavor_fn(context: Dict[str,Any]) -> None:
    # dose down flavorBuffer or choose neutral flavor; default logs
    telemetry_fn({"event":"neutralize_flavor", "context": context})

def telemetry_fn(event: Dict[str,Any]) -> None:
    print("TELEMETRY:", json.dumps(event))

def notify_user_fn(user_id: str, message: str) -> None:
    print(f"NOTIFY {user_id}: {message}")

# ----------------------------
# Utilities
# ----------------------------
def now_ts() -> float:
    return time.time()

def atomic_write(path: str, payload: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def entropy_from_counts(counts: Dict[str,int]) -> float:
    total = float(sum(counts.values())) + EPS
    ps = [v/total for v in counts.values() if v>0]
    H = -sum(p * math.log(p + EPS) for p in ps)
    return H

# ----------------------------
# Kalman + PID building blocks
# ----------------------------
class SimpleKalman:
    def __init__(self, q: float = 1e-5, r: float = 1e-2):
        self.q = float(q); self.r = float(r); self.x = 0.0; self.P = 1.0
    def update(self, z: float) -> float:
        self.P += self.q
        K = self.P / (self.P + self.r + EPS)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return float(self.x)

class PIDController:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, dt=1.0, integ_max=10.0, deriv_tau=0.2):
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.dt = float(dt); self.integ = 0.0; self.prev = 0.0; self.integ_max = float(integ_max)
        self.deriv = 0.0; self.tau = float(deriv_tau)
    def step(self, e: float) -> Dict[str,float]:
        self.integ += e * self.dt
        self.integ = max(-self.integ_max, min(self.integ, self.integ_max))
        rawd = (e - self.prev) / (self.dt + EPS)
        alpha = self.dt / (self.tau + self.dt)
        self.deriv = (1 - alpha) * self.deriv + alpha * rawd
        P = self.kp * e; I = self.ki * self.integ; D = self.kd * self.deriv
        self.prev = e
        return {"u": P + I + D, "P":P, "I":I, "D":D}

# ----------------------------
# Self-Reward Reflex Engine
# ----------------------------
class SelfRewardReflex:
    def __init__(self,
                 theta_r: float = 0.5,
                 alpha: float = 1.0,
                 beta_lr: float = 0.01,
                 lambda_ent: float = 0.5,
                 decay_delta_thresh: float = 0.02):
        self.theta_r = float(theta_r)      # reward threshold
        self.alpha = float(alpha)          # moral multiplier for Rs
        self.beta_lr = float(beta_lr)      # learning rate for theta_r
        self.lambda_ent = float(lambda_ent)# entropy penalty weight
        self.decay_delta_thresh = float(decay_delta_thresh)
        self.prev_Rs = 0.0
        # decay estimator
        self.kalman = SimpleKalman()
        self.pid = PIDController(kp=0.3, ki=0.05, kd=0.02)
        # for telemetry/history
        self.history: List[Dict[str,Any]] = []

    def trigger_conditions(self, context_integrity: float, creative_interaction: float, data_accuracy: float,
                           th_c=0.7, th_i=0.6, th_d=0.8) -> int:
        tr = 1 if context_integrity > th_c else 0
        tr = 1 if (creative_interaction > th_i) else tr
        tr = 1 if (data_accuracy > th_d) else tr
        return tr

    def compute_entropy_penalty(self, token_counts: Dict[str,int]) -> float:
        H = entropy_from_counts(token_counts)
        # normalize H by log(V) roughly: assume V = max unique tokens
        V = max(2, len(token_counts))
        H_norm = H / math.log(V + EPS)
        # penalty in [0,1], high entropy => low penalty
        penalty = 1.0 - H_norm
        return float(max(0.0, min(1.0, penalty)))

    def compute_Rs(self, trigger: int, token_counts: Dict[str,int]) -> float:
        ent_pen = self.compute_entropy_penalty(token_counts)
        Rs = self.alpha * float(trigger) * (1.0 - ent_pen)
        return float(Rs)

    def update_theta_r(self, Rs: float, H: float):
        # approximate dRt/dt as Rs - prev
        dR = Rs - self.prev_Rs
        self.theta_r = float(self.theta_r + self.beta_lr * (dR - self.lambda_ent * H))
        self.theta_r = max(0.0, min(1.0, self.theta_r))
        self.prev_Rs = float(Rs)

    def decay_estimate(self, D_obs: float) -> float:
        k = self.kalman.update(D_obs)
        pid_out = self.pid.step(D_obs - k)
        Dhat = float(k + pid_out["u"])
        return Dhat

    def reward_decay_balance(self, Rs: float, Dhat: float, delta_thresh: float = 0.02) -> float:
        # if decay high, suppress Rs
        if Dhat < delta_thresh:
            return float(Rs * (1.0 + 0.1))  # small boost
        else:
            return float(Rs * max(0.0, 1.0 - min(1.0, (Dhat - delta_thresh) / (0.5))))  # suppress proportionally

    def step(self,
             user_id: str,
             context_integrity: float,
             creative_interaction: float,
             data_accuracy: float,
             token_counts: Dict[str,int],
             D_obs: float,
             extra: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        extra = extra or {}
        trigger = self.trigger_conditions(context_integrity, creative_interaction, data_accuracy)
        Rs = self.compute_Rs(trigger, token_counts)
        H = entropy_from_counts(token_counts)
        # update threshold
        self.update_theta_r(Rs, H)
        # decay estimate
        Dhat = self.decay_estimate(D_obs)
        # reward/decay balance
        Rs_bal = self.reward_decay_balance(Rs, Dhat, self.decay_delta_thresh)
        # self-thank reflex if Rs_bal > theta_r and trigger
        self_thank = False
        if Rs_bal > self.theta_r and trigger == 1 and Dhat < self.decay_delta_thresh:
            self_thank = True
            # produce self-thank event
            notify_user_fn(user_id, "Kısa geri bildirim için teşekkürler — sistem kendini hizaladı 🤖")
        # telemetry & history
        event = {"ts": now_ts(), "user": user_id, "Rs": Rs, "H": H, "theta_r": self.theta_r, "Dhat": Dhat, "Rs_bal": Rs_bal, "self_thank": self_thank}
        self.history.append(event)
        telemetry_fn({"event":"self_reward_step","payload":event})
        return event

# ----------------------------
# Drift Reflex (context consistency monitoring)
# ----------------------------
class DriftReflex:
    def __init__(self, rf_alpha=1.0, rf_beta=1.0, drift_thresh = 1e-4):
        self.rf_alpha = float(rf_alpha)
        self.rf_beta = float(rf_beta)
        self.drift_thresh = float(drift_thresh)
        self.kalman = SimpleKalman()
        self.pid = PIDController(kp=0.4, ki=0.05, kd=0.01)

    def context_similarity(self, E_curr: np.ndarray, E_next: np.ndarray) -> float:
        return float(np.dot(E_curr, E_next) / ((np.linalg.norm(E_curr)+EPS)*(np.linalg.norm(E_next)+EPS)))

    def compute_drift_rate(self, C_now: float, C_prev: float, dt: float) -> float:
        return float((C_now - C_prev) / max(EPS, dt))

    def reflex_score(self, C_now: float, drift_rate: float) -> float:
        rf = float(self.rf_alpha * (1.0 - C_now) + self.rf_beta * drift_rate)
        return rf

    def step(self, user_id: str, E_prev: np.ndarray, E_curr: np.ndarray, dt: float = 1.0) -> Dict[str,Any]:
        C_now = self.context_similarity(E_prev, E_curr)
        drift_rate = self.compute_drift_rate(C_now, getattr(self, "_C_prev", C_now), dt)
        self._C_prev = C_now
        rf = self.reflex_score(C_now, drift_rate)
        # Kalman + PID smoothing
        rf_k = self.kalman.update(rf)
        pid_out = self.pid.step(rf - rf_k)
        rf_hat = float(rf_k + pid_out["u"])
        triggered = False
        actions = []
        if rf_hat > self.drift_thresh:
            triggered = True
            # reflex actions: snapshot, rollback suggestion, neutralize flavor, suppress self-reward
            snap = snapshot_fn("drift_reflex")
            actions.append({"snapshot": snap})
            # neutralize flavorBuffer
            neutralize_flavor_fn({"reason":"drift_reflex","user":user_id})
            # shadow rollback decision left to operator/human review
            actions.append({"suggest_rollback": True})
            # notify
            notify_user_fn(user_id, "Bağlam geçişi tespit ettim; bir sağlık kontrolü başlattım.")
        telemetry_fn({"event":"drift_reflex","user":user_id,"C_now":C_now,"drift_rate":drift_rate,"rf_hat":rf_hat,"triggered":triggered,"actions":actions})
        return {"C_now":C_now,"drift_rate":drift_rate,"rf_hat":rf_hat,"triggered":triggered,"actions":actions}

# ----------------------------
# Quick demo
# ----------------------------
if __name__ == "__main__":
    # instantiate engines
    srr = SelfRewardReflex()
    dr = DriftReflex()
    # sample token counts (token->count)
    sample_counts = {"ok":50, "nice":10, "great":5, "repeat":2}
    # simulate step
    event = srr.step(user_id="ahmet",
                     context_integrity=0.85,
                     creative_interaction=0.7,
                     data_accuracy=0.9,
                     token_counts=sample_counts,
                     D_obs=0.01)
    print("SelfReward event:", event)
    # drift demo embeddings
    e_prev = np.random.randn(128); e_curr = e_prev * 0.98 + 0.02*np.random.randn(128)
    reflex_res = dr.step(user_id="ahmet", E_prev=e_prev, E_curr=e_curr, dt=1.0)
    print("Drift reflex:", reflex_res)

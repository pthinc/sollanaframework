# bce_self_reflex.py
import time, math
from typing import List, Dict, Optional, Callable
import numpy as np

EPS = 1e-12

def now_ts() -> float:
    return time.time()

# ---- düşük seviye yardımcılar ----
def trapezoid_integral(samples: List[Dict], key: str) -> float:
    if len(samples) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(samples)):
        t0, t1 = samples[i-1]["ts"], samples[i]["ts"]
        v0, v1 = float(samples[i-1].get(key, 0.0)), float(samples[i].get(key,0.0))
        total += 0.5 * (v0 + v1) * max(0.0, t1 - t0)
    return float(total)

def l2_norm(v):
    a = np.asarray(v, dtype=float)
    return float(np.linalg.norm(a))

# ---- temel hesaplar ----
def compute_reflex(delta_decay: float, norm: float, grad_char: float) -> float:
    """Reflex = ∇D * N * (1 - ∇C) ; clamp 0..1"""
    val = float(delta_decay) * float(norm) * max(0.0, 1.0 - float(grad_char))
    return max(0.0, min(1.0, val))

def compute_self_vector(samples: List[Dict]) -> float:
    """S = sum_t C*(1-D)*N*R over samples"""
    if not samples:
        return 0.0
    total = 0.0
    for s in samples:
        C = float(s.get("C", 0.0))
        D = float(s.get("D", 0.0))
        N = float(s.get("N", 0.0))
        R = float(s.get("R", 0.0))
        total += C * (1.0 - D) * N * R
    return float(total)

def compute_self_continuity(samples: List[Dict]) -> float:
    """Integral over time of ∇S ; approximated by total variation of S over time"""
    if len(samples) < 2:
        return 0.0
    # compute per-sample instantaneous S_i and integrate its derivative
    S_vals = []
    for s in samples:
        S_vals.append(s.get("_S_point", compute_self_vector([s])))
    total = 0.0
    for i in range(1, len(samples)):
        dt = max(1e-6, samples[i]["ts"] - samples[i-1]["ts"])
        dS = S_vals[i] - S_vals[i-1]
        total += dS * dt
    return float(total)

def compute_self_drift(S_now: float, S_prev: float) -> float:
    return float(abs(S_now - S_prev))

# ---- Benlik Denetleyici Sınıfı ----
class SelfController:
    def __init__(self,
                 reflex_threshold: float = 0.2,
                 drift_threshold: float = 0.15,
                 protect_duration_s: float = 3600.0,
                 remediate_callback: Optional[Callable[[Dict], None]] = None,
                 protect_callback: Optional[Callable[[str, float], None]] = None):
        """
        remediate_callback(entry) çağrılırsa sistem otomatik müdahaleyi uygular (ör: decay_boost)
        protect_callback(trace_id, duration_s) çağrısı ile trace korumaya alınır.
        """
        self.reflex_threshold = float(reflex_threshold)
        self.drift_threshold = float(drift_threshold)
        self.protect_duration_s = float(protect_duration_s)
        self.remediate_callback = remediate_callback
        self.protect_callback = protect_callback
        self.history_by_user = {}  # user_id -> list of samples (ts,C,D,N,R,...)
        self.self_memory = {}      # user_id -> list of self snapshots (ts, S)
        self.last_S = {}           # user_id -> last S

    def register_sample(self, user_id: str, sample: Dict):
        """
        sample must include: ts (optional), C, D, N, R, optional trace_id
        stores sample and evaluates reflex/drift
        """
        ts = float(sample.get("ts", now_ts()))
        s = {"ts": ts, "C": float(sample.get("C",0.0)), "D": float(sample.get("D",0.0)),
             "N": float(sample.get("N",0.0)), "R": float(sample.get("R",0.0)),
             "trace_id": sample.get("trace_id")}
        hist = self.history_by_user.setdefault(user_id, [])
        hist.append(s)
        # keep last window
        if len(hist) > 1024:
            hist.pop(0)
        # compute derivative approximations
        delta_decay = 0.0
        grad_char = 0.0
        if len(hist) >= 2:
            delta_decay = (hist[-1]["D"] - hist[-2]["D"]) / max(1e-6, hist[-1]["ts"] - hist[-2]["ts"])
            grad_char = (hist[-1]["C"] - hist[-2]["C"]) / max(1e-6, hist[-1]["ts"] - hist[-2]["ts"])
            # clamp small values
            delta_decay = max(-1.0, min(1.0, delta_decay))
            grad_char = max(-1.0, min(1.0, grad_char))
        reflex = compute_reflex(delta_decay, s["N"], grad_char)
        # compute S and drift
        S_now = compute_self_vector(hist)
        S_prev = self.last_S.get(user_id, S_now)
        drift = compute_self_drift(S_now, S_prev)
        self.last_S[user_id] = S_now
        # store S snapshot
        mem = self.self_memory.setdefault(user_id, [])
        mem.append({"ts": ts, "S": S_now, "drift": drift})
        if len(mem) > 4096:
            mem.pop(0)
        # actions
        actions = {"reflex": reflex, "drift": drift, "S": S_now}
        if reflex >= self.reflex_threshold:
            actions["reflex_action"] = "remediate"
            if self.remediate_callback:
                try:
                    self.remediate_callback({"user_id": user_id, "sample": s, "reflex": reflex})
                except Exception:
                    pass
        if drift >= self.drift_threshold:
            actions["drift_action"] = "protect"
            if s.get("trace_id") and self.protect_callback:
                try:
                    self.protect_callback(s["trace_id"], self.protect_duration_s)
                except Exception:
                    pass
        return actions

    def get_self_time_series(self, user_id: str, last_n: int = 200):
        mem = self.self_memory.get(user_id, [])
        return mem[-last_n:]

    def compute_continuity(self, user_id: str) -> float:
        mem = self.get_self_time_series(user_id)
        if not mem:
            return 0.0
        # continuity approximated by inverse of mean drift
        drifts = [m["drift"] for m in mem if "drift" in m]
        if not drifts:
            return 1.0
        mean_d = float(sum(drifts)/len(drifts))
        return float(max(0.0, 1.0 - mean_d))

# ---- küçük demo ----
if __name__ == "__main__":
    sc = SelfController(
        reflex_threshold=0.15,
        drift_threshold=0.1,
        remediate_callback=lambda e: print("REMEDIATE:", e),
        protect_callback=lambda tid,dur: print("PROTECT:", tid, dur)
    )
    # simulate user samples
    user = "alice"
    base_ts = now_ts()
    for i in range(6):
        sample = {
            "ts": base_ts + i*10,
            "C": 0.5 + 0.05*i,
            "D": 0.02 + (0.05 if i==3 else 0.0),  # spike at i==3
            "N": 0.8,
            "R": 0.6,
            "trace_id": f"t{i}"
        }
        actions = sc.register_sample(user, sample)
        print(i, actions)
    print("continuity:", sc.compute_continuity(user))

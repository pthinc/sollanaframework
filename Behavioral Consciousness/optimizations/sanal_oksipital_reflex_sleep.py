# sanal_oksipital_reflex_sleep.py
import time, math, json
from typing import Dict, Any, Callable, List, Tuple
import numpy as np

EPS = 1e-12

# ------------------------
# Hooks (replace these)
# ------------------------
def snapshot_fn(reason: str) -> str:
    return f"snapshot_{int(time.time()*1000)}"

def quarantine_fn(scope: Dict[str,Any]) -> None:
    print("QUARANTINE:", scope)

def telemetry_fn(event: Dict[str,Any]) -> None:
    print("TELEMETRY:", json.dumps(event))

def notify_fn(user: str, msg: str) -> None:
    print(f"NOTIFY {user}: {msg}")

# ------------------------
# Fuzzy membership helpers
# ------------------------
def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a + EPS)
    return (c - x) / (c - b + EPS)

def gaussian_membership(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / (sigma + EPS))**2)

# ------------------------
# FuzzySet definitions (default thresholds)
# ------------------------
class FuzzyMetric:
    def __init__(self, low_thr: float, high_thr: float, kind: str = "tri"):
        self.low = float(low_thr); self.high = float(high_thr); self.kind = kind
    def membership(self, x: float) -> Dict[str,float]:
        if self.kind == "tri":
            low = triangular_membership(x, -1.0, 0.0, self.low)
            med = triangular_membership(x, self.low, (self.low + self.high)/2.0, self.high)
            high = triangular_membership(x, self.high, (self.high + 1.0)/2.0, 1.0)
        else:
            low = gaussian_membership(x, 0.0, max(0.05, self.low/2))
            med = gaussian_membership(x, (self.low+self.high)/2.0, max(0.05,(self.high-self.low)/2))
            high = gaussian_membership(x, 1.0, max(0.05,(1.0-self.high)/2))
        # normalize to 0..1
        s = max(EPS, low + med + high)
        return {"Low": low/s, "Medium": med/s, "High": high/s}

# ------------------------
# Reflex Sleep Controller
# ------------------------
class ReflexSleepController:
    def __init__(self,
                 fuzzy_params: Dict[str, Tuple[float,float]] = None,
                 window_tokens: int = 4000,
                 telemetry_interval_tokens: int = 2000,
                 lambda_penalty: float = 0.1,
                 grid_search_budget: int = 25):
        # fuzzy thresholds per metric (low_thr, high_thr)
        defaults = {"D":(0.02,0.08), "H":(0.15,0.5), "A":(0.01,0.05), "C":(0.6,0.85), "M":(0.1,0.4)}
        fps = fuzzy_params or defaults
        self.metrics = {k: FuzzyMetric(*fps[k]) for k in fps}
        self.window_tokens = int(window_tokens)
        self.telemetry_tokens = int(telemetry_interval_tokens)
        self.lambda_penalty = float(lambda_penalty)
        self.grid_budget = int(grid_search_budget)
        # running windows: store recent values per metric (sampled per event)
        self.buffers: Dict[str, List[float]] = {k:[] for k in self.metrics.keys()}
        self.token_counter = 0
        # adaptive thresholds baseline
        self.theta_base = {k: (fps[k][1]) for k in fps}  # use high_thr as base
        self.theta = dict(self.theta_base)
        self.last_telemetry_ts = time.time()
    # observe one snapshot (sample)
    def observe(self, token_count: int, D: float, H: float, A: float, C: float, M: float, context_fit: float = 0.0, user: str = "system"):
        self.token_counter += token_count
        for k,v in [("D",D),("H",H),("A",A),("C",C),("M",M)]:
            buf = self.buffers[k]; buf.append(float(v))
            if len(buf) > 1024: buf.pop(0)
        # compute fuzzy memberships averaged over sliding window T
        wi = self._compute_window_weights()
        mu = self._compute_memberships(wi)
        # trigger rule
        trigger = self._apply_trigger_rule(mu)
        # if trigger then schedule/rest action
        if trigger:
            # compute kümülatif tolerans
            tcum = sum(wi[k] * mu[k]["High"] for k in wi.keys())
            # threshold total adaptive
            theta_total = 0.5 + 0.5 * (1.0 - context_fit)
            if tcum > theta_total:
                # execute rest reflex actions
                snap_id = snapshot_fn("reflex_sleep")
                quarantine_fn({"reason":"reflex_sleep","user":user,"tcum":tcum})
                notify_fn(user, "Kısa bir dinlenme kalibrasyonu başlatıldı; hemen döneceğim 🤖")
                telemetry_fn({"event":"reflex_sleep_trigger","user":user,"tcum":tcum,"snap":snap_id,"mu":mu})
        # telemetry periodic
        if self.token_counter >= self.telemetry_tokens:
            tel = {"event":"periodic_telemetry","counts":{k:len(self.buffers[k]) for k in self.buffers},"avg":{k:np.mean(self.buffers[k]) if self.buffers[k] else None for k in self.buffers}}
            telemetry_fn(tel)
            self.token_counter = 0
            # adapt thresholds
            self._adaptive_threshold_update()
        return {"mu": mu, "trigger": bool(trigger)}
    def _compute_window_weights(self) -> Dict[str,float]:
        # integrate membership mu_i over last T tokens; simple avg weight per metric
        weights = {}
        for k,b in self.buffers.items():
            if not b: weights[k] = 0.0
            else: weights[k] = float(np.mean(np.clip(b,0.0,1.0)))
        # normalize
        s = sum(abs(v) for v in weights.values()) + EPS
        return {k:weights[k]/s for k in weights}
    def _compute_memberships(self, weights: Dict[str,float]) -> Dict[str,Dict[str,float]]:
        mu = {}
        for k, metric in self.metrics.items():
            vals = self.buffers[k][-16:] if self.buffers[k] else [0.0]
            # average membership across recent samples
            ms = [metric.membership(v) for v in vals]
            agg = {"Low":0.0,"Medium":0.0,"High":0.0}
            for m in ms:
                agg["Low"] += m["Low"]; agg["Medium"] += m["Medium"]; agg["High"] += m["High"]
            n = len(ms) + EPS
            mu[k] = {kk: agg[kk]/n for kk in agg}
        return mu
    def _apply_trigger_rule(self, mu: Dict[str,Dict[str,float]]) -> int:
        high_counts = sum(1 for k in mu.keys() if mu[k]["High"] > 0.5)
        med_counts = sum(1 for k in mu.keys() if mu[k]["Medium"] > 0.5)
        if high_counts >= 2: return 1
        if high_counts == 1 and med_counts >= 1: return 1
        return 0
    def _adaptive_threshold_update(self):
        # simple learning: increase high thresholds when frequent triggers, else decay
        # compute trigger frequency proxy: number of High in last windows
        freq = sum(1 for k in self.buffers if self.buffers[k] and np.mean(self.buffers[k]) > self.theta_base[k])
        for k in self.theta.keys():
            self.theta[k] = self.theta[k] + 0.01 * (freq - 1)  # small nudges
            # clamp sensible
            self.theta[k] = max(0.01, min(0.99, self.theta[k]))
    # small grid search to tune fuzzy membership weights (Wi)
    def grid_optimize_weights(self, objective_fn: Callable[[Dict[str,float]],float], grid_steps: int = 5):
        # search small grid around current weights
        cur_w = self._compute_window_weights()
        keys = list(cur_w.keys())
        best = (cur_w, objective_fn(cur_w))
        for _ in range(self.grid_budget):
            cand = {k: max(0.0, min(1.0, cur_w[k] + np.random.randn()*0.1)) for k in keys}
            s = sum(cand.values()) + EPS
            cand = {k: cand[k]/s for k in keys}
            val = objective_fn(cand)
            if val > best[1]:
                best = (cand, val)
        return best

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    controller = ReflexSleepController()
    # simulate stream of observations (per sample)
    for i in range(120):
        # synthetic metrics in [0,1]
        D = max(0.0, min(1.0, 0.02 + 0.01*math.sin(i/6.0)))
        H = max(0.0, min(1.0, 0.1 + 0.2*((i%30)>25)))
        A = max(0.0, min(1.0, 0.02 if (i%20)>18 else 0.005))
        C = max(0.0, min(1.0, 0.8 - 0.05*math.sin(i/10.0)))
        M = max(0.0, min(1.0, 0.2 + 0.1*math.cos(i/7.0)))
        res = controller.observe(token_count=100, D=D, H=H, A=A, C=C, M=M, context_fit=0.9, user="ahmet")
        if res["trigger"]:
            print("REST TRIGGERED at step", i, "mu:", res["mu"])

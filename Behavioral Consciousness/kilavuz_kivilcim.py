# kilavuz_kivilcim.py
import math, time, numpy as np
from typing import Dict, Any, Callable

EPS = 1e-12
PI = math.pi

# Hook placeholders
audit_fn: Callable[[Dict], None] = lambda e: print("AUDIT", e)
notify_fn: Callable[[str, Dict], None] = lambda level, p: print("NOTIFY", level, p)
shadow_action: Callable[[Dict], None] = lambda p: print("SHADOW_ACTION", p)
hitl_request: Callable[[Dict], None] = lambda p: print("HITL_REQUEST", p)

# EMA helper
def ema(prev, value, alpha=0.15):
    return alpha*value + (1-alpha)*(prev if prev is not None else value)

class KivilcimEngine:
    def __init__(self, rho=0.8, eps_min=1e-3):
        self.rho = float(rho)
        self.eps_min = float(eps_min)
        self.state = {}  # hold EMAs for smoothing

    # Estimators (replace these with real instrumentation)
    def estimate_phi(self, signal_window: np.ndarray, ref=0.5):
        rms = float(np.sqrt(np.mean(np.square(signal_window)))) if len(signal_window)>0 else 0.0
        return min(10.0, rms / max(ref, EPS))  # allow >1 scale

    def estimate_lambda(self, soul_tag_count: int):
        return math.log(1 + max(0, soul_tag_count))

    def estimate_mu(self, intent_confidence: float):
        return float(max(0.0, min(1.0, intent_confidence)))

    def estimate_delta(self, context_embs: np.ndarray):
        # context_embs shape (k, d) last k embeddings; compute mean pairwise distance
        if context_embs is None or len(context_embs) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(context_embs)):
            diffs.append(np.linalg.norm(context_embs[i] - context_embs[i-1]))
        return float(np.mean(diffs))

    def estimate_epsilon(self, anon_score: float):
        # anon_score in [0,1] where 1 = fully anonymous -> large epsilon
        return max(self.eps_min, float(0.001 + anon_score))

    def compute_C(self, phi, lam, mu, delta, eps):
        denom = delta + max(eps, self.eps_min)
        base = (phi * lam + mu) / denom
        C = base * (PI ** self.rho)
        return float(C)

    def step(self, telemetry: Dict[str, Any]):
        # telemetry expected keys: signal_window (np.array), soul_tag_count (int), intent_conf (float),
        # context_embs (np.array kxd), anon_score (0..1)
        phi = self.estimate_phi(np.asarray(telemetry.get("signal_window", [])))
        lam = self.estimate_lambda(int(telemetry.get("soul_tag_count", 0)))
        mu = self.estimate_mu(float(telemetry.get("intent_conf", 0.0)))
        delta = self.estimate_delta(telemetry.get("context_embs", None))
        eps = self.estimate_epsilon(float(telemetry.get("anon_score", 0.0)))

        # smoothing
        self.state['phi'] = ema(self.state.get('phi'), phi)
        self.state['lam'] = ema(self.state.get('lam'), lam)
        self.state['mu'] = ema(self.state.get('mu'), mu)
        self.state['delta'] = ema(self.state.get('delta'), delta)
        self.state['eps'] = ema(self.state.get('eps'), eps)

        C = self.compute_C(self.state['phi'], self.state['lam'], self.state['mu'], self.state['delta'], self.state['eps'])

        record = {"ts": time.time(), "C": C, "components": {
            "phi": self.state['phi'], "lambda": self.state['lam'], "mu": self.state['mu'],
            "delta": self.state['delta'], "eps": self.state['eps'], "rho": self.rho
        }}
        audit_fn({"event":"kivilcim_eval","payload": record})

        # decision logic with cooldown and safety
        if C > 7.0:
            audit_fn({"event":"kivilcim_evolution","C":C})
            hitl_request({"level":"evolved","C":C, "record": record})
            return {"level":"evolved","C":C}
        if C > math.pi:
            audit_fn({"event":"kivilcim_ready_canary","C":C})
            notify_fn("canary", {"C":C})
            shadow_action({"action":"canary_suggest","C":C})
            return {"level":"canary_recommended","C":C}
        if C > 1.0:
            audit_fn({"event":"kivilcim_character","C":C})
            shadow_action({"action":"character_suggest","C":C})
            return {"level":"character_threshold","C":C}
        return {"level":"below_threshold","C":C}

import math
import time
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

EPS = 1e-12


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmax = max(vals)
    exps = [math.exp(v - vmax) for v in vals]
    s = sum(exps) + EPS
    return [e / s for e in exps]


def _safe_log(x: float) -> float:
    return math.log(max(x, EPS))


def _cos_theta(theta: Optional[float]) -> float:
    if theta is None:
        return 1.0
    return math.cos(theta)


def _norm(arr: np.ndarray) -> float:
    return float(np.linalg.norm(arr))


def _normalize(v: List[float]) -> List[float]:
    s = sum(abs(x) for x in v) + EPS
    return [x / s for x in v]


class MasterLawScorer:
    """
    Implements the "Master Law - Token Pathting" scorer from BCE docs.

    Key pieces:
      - Subtoken depth SD_i = sum_j delta_j * log(DataDensity_ij)
      - ContextFit_i = B(U) * Gamma(U) * K_i(U)
      - TamponValue_i = rho * e(t) * K_i(U)
      - S_i = lambda1 * SD_i + K_i(U) * (lambda2 * B(U) * Gamma(U) + lambda3 * rho * e(t))
      - P(i|U) = softmax(S_i)

    Usage:
        scorer = MasterLawScorer()
        scores = scorer.score(token_candidates, context)

    token_candidates: List[{
        "data_density": List[float],  # per-subtoken density values
        "delta": List[float],         # per-subtoken ceremonial weights (same length as data_density)
        "context_fit": Optional[float],
        "tampon_value": Optional[float],
        "theta": Optional[float]      # cultural/intent projection angle
    }]

    context:
        {
          "bayes_posteriors": List[float],    # P(C_k | U)
          "kp": float, "ki": float, "kd": float,
          "rho": float,                       # tampon scaling
          "lambda": {"sd":, "ctx":, "tampon":},
          "dt": float,                        # time delta for PID
          "prior_error": float,               # optional seed
        }
    """

    def __init__(self,
                 lambda_sd: float = 1.0,
                 lambda_ctx: float = 1.0,
                 lambda_tampon: float = 1.0,
                 rho: float = 1.0,
                 kp: float = 1.0,
                 ki: float = 0.0,
                 kd: float = 0.0):
        self.lambda_sd = float(lambda_sd)
        self.lambda_ctx = float(lambda_ctx)
        self.lambda_tampon = float(lambda_tampon)
        self.rho = float(rho)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self._last_error: float = 0.0
        self._integral: float = 0.0
        self._last_ts: Optional[float] = None

    def reset_state(self) -> None:
        self._last_error = 0.0
        self._integral = 0.0
        self._last_ts = None

    @staticmethod
    def _subtoken_depth(data_density: List[float], delta: List[float]) -> float:
        if not data_density:
            return 0.0
        dd = np.asarray(data_density, dtype=float)
        de = np.asarray(delta if delta else np.ones_like(dd), dtype=float)
        if de.shape != dd.shape:
            de = np.ones_like(dd)
        return float(np.sum(de * np.log(np.clip(dd, EPS, None))))

    def _pid_temper(self, e_t: float, dt: Optional[float]) -> float:
        now = time.time()
        if dt is None:
            if self._last_ts is None:
                dt = 1.0
            else:
                dt = max(1e-3, now - self._last_ts)
        self._last_ts = now
        self._integral += e_t * dt
        derivative = (e_t - self._last_error) / max(dt, 1e-3)
        self._last_error = e_t
        u_t = self.kp * e_t + self.ki * self._integral + self.kd * derivative
        gamma = 1.0 - _sigmoid(u_t)
        return float(max(0.0, min(1.0, gamma)))

    def score(self,
              token_candidates: List[Dict[str, Any]],
              context: Dict[str, Any]) -> Dict[str, Any]:
        if not token_candidates:
            return {"scores": [], "probs": [], "meta": {"B": 0.0, "e_t": 0.0, "gamma": 0.0}}

        bayes_post = context.get("bayes_posteriors") or []
        B = float(max(bayes_post) if bayes_post else context.get("B", 0.6))
        e_t = 1.0 - B
        self.kp = float(context.get("kp", self.kp))
        self.ki = float(context.get("ki", self.ki))
        self.kd = float(context.get("kd", self.kd))
        self.rho = float(context.get("rho", self.rho))
        lambdas = context.get("lambda") or {}
        self.lambda_sd = float(lambdas.get("sd", self.lambda_sd))
        self.lambda_ctx = float(lambdas.get("ctx", self.lambda_ctx))
        self.lambda_tampon = float(lambdas.get("tampon", self.lambda_tampon))
        dt = context.get("dt")

        gamma = self._pid_temper(e_t, dt)

        scores = []
        details: List[Dict[str, float]] = []
        for cand in token_candidates:
            dd = cand.get("data_density", [])
            delta = cand.get("delta", [])
            theta = cand.get("theta")
            K_i = _cos_theta(theta)
            sd_i = self._subtoken_depth(dd, delta)
            tampon_val = cand.get("tampon_value")
            if tampon_val is None:
                tampon_val = self.rho * e_t * K_i
            ctx_fit = cand.get("context_fit")
            if ctx_fit is None:
                ctx_fit = B * gamma * K_i
            S_i = self.lambda_sd * sd_i + K_i * (self.lambda_ctx * ctx_fit + self.lambda_tampon * tampon_val)
            scores.append(S_i)
            details.append({
                "sd": sd_i,
                "ctx_fit": float(ctx_fit),
                "tampon": float(tampon_val),
                "K_i": float(K_i),
                "S_i": float(S_i)
            })

        probs = _softmax(scores)
        # normalize scores for readability
        norm_scores = _normalize(scores)
        return {
            "scores": norm_scores,
            "probs": probs,
            "meta": {"B": B, "e_t": e_t, "gamma": gamma},
            "details": details
        }


def example_usage():
    scorer = MasterLawScorer(lambda_sd=1.0, lambda_ctx=0.7, lambda_tampon=0.5, rho=0.8, kp=1.2, ki=0.0, kd=0.0)
    token_candidates = [
        {"data_density": [0.4, 0.8, 0.6], "delta": [1.0, 0.8, 1.2], "theta": 0.2},
        {"data_density": [0.6, 0.3, 0.9], "delta": [1.0, 1.0, 1.0], "theta": 0.1},
    ]
    context = {
        "bayes_posteriors": [0.55, 0.25, 0.2],
        "kp": 1.0, "ki": 0.1, "kd": 0.05,
        "rho": 0.9,
        "lambda": {"sd": 1.0, "ctx": 0.7, "tampon": 0.6}
    }
    result = scorer.score(token_candidates, context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    example_usage()

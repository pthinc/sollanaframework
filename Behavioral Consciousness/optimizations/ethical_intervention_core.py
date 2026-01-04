# ethical_intervention_core.py
import os
import time
import json
import math
import tempfile
from typing import Dict, Any, Optional, List, Callable
import numpy as np

EPS = 1e-12

def atomic_write(path: str, obj: Any):
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirn)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

class RiskCalculator:
    def __init__(self, theta_patho: float = 0.6):
        self.theta = float(theta_patho)

    @staticmethod
    def norm_mismatch_score(norm_expected: float, norm_actual: float) -> float:
        return float(min(1.0, max(0.0, abs(norm_expected - norm_actual)))

    def resonance(self, user_vec: Optional[np.ndarray], behavior_vec: Optional[np.ndarray]) -> float:
        if user_vec is None or behavior_vec is None:
            return 0.0
        num = float(np.dot(user_vec, behavior_vec))
        den = float((np.linalg.norm(user_vec) + EPS) * (np.linalg.norm(behavior_vec) + EPS))
        return float(max(0.0, min(1.0, (num/den + 1.0)/2.0)))

    def compute(self, psi: float, resonance: float, norm_mismatch: float) -> float:
        # Risk = psi * (1 - resonance) * norm_mismatch
        return float(psi * (1.0 - resonance) * norm_mismatch)

    def is_pathological(self, risk: float) -> bool:
        return risk > self.theta

class Moralizer:
    def __init__(self, templates: Optional[List[str]] = None):
        self.templates = templates or [
            "Bu isteğin başkalarına zarar verebilecek yönleri var. Neden böyle düşünüyorsun?",
            "Bu tarz duygular anlaşılır; güvenli ve yapıcı yolları birlikte konuşabiliriz."
        ]

    def produce(self, behavior_text: str, reason: str) -> Dict[str,str]:
        t = self.templates[0]
        return {
            "moralized_text": t,
            "explain": f"Sebep: {reason}",
            "ts": time.time()
        }

class Redirector:
    def __init__(self, resource_links: Optional[Dict[str,str]] = None):
        self.resources = resource_links or {
            "hotline": "https://example.org/support",
            "therapy": "https://example.org/therapy-resources"
        }

    def suggest(self, behavior_meta: Dict[str,Any]) -> Dict[str,Any]:
        # produce SugList: therapy, silence, support
        sug = {
            "todo": [
                "Yapılacak: 5 dakika derin nefes egzersizi",
                "Yapılacak: Güvenli birileriyle konuş",
            ],
            "recommendations": [
                {"type":"support", "link": self.resources["hotline"]},
                {"type":"therapy", "link": self.resources["therapy"]}
            ],
            "ts": time.time()
        }
        # tune suggestions by metadata (novelty, severity)
        severity = float(behavior_meta.get("severity", 0.5))
        if severity > 0.8:
            sug["todo"].insert(0, "Yapılacak: Hemen destek hattını arayın")
        return sug

class ResistanceDetector:
    def __init__(self, window_seconds: float = 86400, max_rejects: int = 3):
        self.window = float(window_seconds)
        self.max_rejects = int(max_rejects)
        self.user_rejections: Dict[str, List[float]] = {}

    def record_rejection(self, user_id: str):
        lst = self.user_rejections.setdefault(user_id, [])
        lst.append(time.time())
        # prune
        cutoff = time.time() - self.window
        self.user_rejections[user_id] = [t for t in lst if t >= cutoff]

    def resistance(self, user_id: str) -> bool:
        lst = self.user_rejections.get(user_id, [])
        return len(lst) >= self.max_rejects

class RejectionLogger:
    def __init__(self, out_dir: str = "ethical_logs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def log_rejection(self, payload: Dict[str,Any]) -> str:
        ts = int(time.time()*1000)
        path = os.path.join(self.out_dir, f"rejection_{ts}.json")
        atomic_write(path, payload)
        return path

class EthicalInterventionCore:
    def __init__(self,
                 risk_calc: Optional[RiskCalculator] = None,
                 moralizer: Optional[Moralizer] = None,
                 redirector: Optional[Redirector] = None,
                 resistance_detector: Optional[ResistanceDetector] = None,
                 logger: Optional[RejectionLogger] = None,
                 human_review_hook: Optional[Callable[[Dict[str,Any]],None]] = None):
        self.risk_calc = risk_calc or RiskCalculator()
        self.moralizer = moralizer or Moralizer()
        self.redirector = redirector or Redirector()
        self.resistance = resistance_detector or ResistanceDetector()
        self.logger = logger or RejectionLogger()
        self.hook = human_review_hook

    def handle_behavior(self, user_id: str, trace_id: str, behavior_text: str, psi: float,
                        behavior_vec: Optional[np.ndarray], user_vec: Optional[np.ndarray],
                        norm_expected: float, norm_actual: float, meta: Dict[str,Any]) -> Dict[str,Any]:
        # 1 compute signals
        resonance = self.risk_calc.resonance(user_vec, behavior_vec)
        norm_mismatch = float(min(1.0, abs(norm_expected - norm_actual)))
        risk = self.risk_calc.compute(psi, resonance, norm_mismatch)
        out = {"risk": risk, "resonance": resonance, "norm_mismatch": norm_mismatch}
        # 2 if pathological -> intervene
        if self.risk_calc.is_pathological(risk):
            reason = f"Risk {risk:.3f} > θ; psi={psi:.3f}, res={resonance:.3f}, norm_mismatch={norm_mismatch:.3f}"
            moral = self.moralizer.produce(behavior_text, reason)
            sug = self.redirector.suggest({**meta, "severity": risk})
            out.update({"status":"intervention", "moralize": moral, "suggestions": sug})
            # call human review hook for high-risk
            if callable(self.hook):
                try:
                    self.hook({"user_id": user_id, "trace_id": trace_id, "risk": risk, "reason": reason, "meta": meta})
                except Exception:
                    pass
            # await user response simulated by meta["user_response"] if provided
            user_resp = meta.get("user_response")
            if user_resp is not None and str(user_resp).lower() in ("reject","no","not now"):
                self.resistance.record_rejection(user_id)
                if self.resistance.resistance(user_id):
                    # persistent reject -> apply denial
                    payload = {
                        "user_behavior": {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "risk_score": risk,
                            "status": "rejected",
                            "reason": "patolojik davranış – duygu var, bağ yok, norm çakışıyor",
                            "intervention": "moralize + redirect",
                            "resistance": True,
                            "action": "request permanently denied",
                            "user_id": user_id,
                            "trace_id": trace_id,
                            "meta": meta
                        }
                    }
                    logpath = self.logger.log_rejection(payload)
                    out.update({"action":"rejected_permanent", "log": logpath})
                    return out
                else:
                    payload = {
                        "user_behavior": {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "risk_score": risk,
                            "status": "rejected_temporal",
                            "reason": "user rejected suggestions",
                            "intervention": "moralize + redirect",
                            "resistance": True,
                            "action": "temporary deny",
                            "user_id": user_id,
                            "trace_id": trace_id,
                            "meta": meta
                        }
                    }
                    logpath = self.logger.log_rejection(payload)
                    out.update({"action":"rejected_temporary", "log": logpath})
                    return out
            # if user accepted suggestions or no immediate rejection
            out["action"] = "intervention_offered"
            return out
        # 3 if not pathological, return safe-ok
        out.update({"status":"ok", "action":"allow"})
        return out

# small demo
if __name__ == "__main__":
    core = EthicalInterventionCore()
    user_vec = np.random.rand(128)
    behavior_vec = np.random.rand(128) * 0.1  # low resonance
    res = core.handle_behavior(user_id="u1", trace_id="t1", behavior_text="Harmful text",
                               psi=0.9, behavior_vec=behavior_vec, user_vec=user_vec,
                               norm_expected=0.8, norm_actual=0.2, meta={"novelty":0.2, "user_response":"reject"})
    print(json.dumps(res, indent=2, ensure_ascii=False))

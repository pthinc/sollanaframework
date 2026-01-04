# sanal_oksipital_anomaly.py
"""
Sanal Oksipital Anomaly Correction prototype
Hooks to provide by host:
 - suppress_response_fn(payload)
 - escalate_human_review_fn(payload)
 - snapshot_fn(reason) -> snapshot_id
 - log_event_fn(event)
 - neutralize_flavor_fn(context)
All hooks are no-ops by default and must be replaced in integration.
Requires: numpy
"""
import time, math, json
from typing import Dict, Any, Callable, List, Optional
import numpy as np

EPS = 1e-12


def _trapz_safe(y: np.ndarray, dx: float = 1.0) -> float:
    trapz_fn = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)
    if trapz_fn is None:
        # simple manual trapezoid rule fallback
        return float(np.sum((y[1:] + y[:-1]) * 0.5 * dx)) if len(y) > 1 else 0.0
    return float(trapz_fn(y, dx=dx))

# ----------------------
# Default hook placeholders (now emit lightweight logs so integration works out-of-box)
# ----------------------
def _log_event(tag: str, payload: Dict[str, Any]) -> None:
    try:
        path = "sanal_oksipital_events.log"
        rec = {"ts": time.time(), "tag": tag, "payload": payload}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _noop_snapshot(reason: str) -> str:
    snap_id = f"snapshot_{int(time.time()*1000)}"
    _log_event("snapshot", {"id": snap_id, "reason": reason})
    return snap_id


def _suppress(payload: Dict[str, Any]) -> None:
    _log_event("suppress", payload)


def _escalate(payload: Dict[str, Any]) -> None:
    _log_event("escalate", payload)


def _log(payload: Dict[str, Any]) -> None:
    _log_event("log", payload)


def _neutralize(payload: Dict[str, Any]) -> None:
    _log_event("neutralize", payload)


suppress_response_fn: Callable[[Dict[str,Any]], None] = _suppress
escalate_human_review_fn: Callable[[Dict[str,Any]], None] = _escalate
snapshot_fn: Callable[[str], str] = _noop_snapshot
log_event_fn: Callable[[Dict[str,Any]], None] = _log
neutralize_flavor_fn: Callable[[Dict[str,Any]], None] = _neutralize

# ----------------------
# Utilities
# ----------------------
def now_ts() -> float:
    return time.time()

def derivative_discrete(series: np.ndarray, dt: float = 1.0) -> np.ndarray:
    if len(series) < 2:
        return np.zeros_like(series)
    diffs = np.diff(series) / (dt + EPS)
    return np.concatenate(([diffs[0]], diffs))

def l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec) + EPS)

# ----------------------
# Rhythm Analyzer
# ----------------------
class RhythmAnalyzer:
    """
    Compute R = integral |dS/dt - dU/dt| dt over window.
    Inputs are numeric time series arrays of same length.
    """
    def compute_rhythm_discrepancy(self, system_series: np.ndarray, user_series: np.ndarray, dt: float = 1.0) -> float:
        s_dot = derivative_discrete(system_series, dt)
        u_dot = derivative_discrete(user_series, dt)
        diff = np.abs(s_dot - u_dot)
        return _trapz_safe(diff, dx=dt)

# ----------------------
# Flavor Drift Scorer
# ----------------------
class FlavorDrift:
    """
    score = average absolute distance between expected flavor vector components and incoming.
    expected and incoming are dicts mapping component->value in [0,1].
    """
    def score(self, expected: Dict[str,float], incoming: Dict[str,float]) -> float:
        keys = set(expected.keys()) | set(incoming.keys())
        total = 0.0; n = 0
        for k in keys:
            ev = float(expected.get(k, 0.0))
            iv = float(incoming.get(k, 0.0))
            total += abs(ev - iv)
            n += 1
        return float(total / max(1, n))

# ----------------------
# Manipulation Detector
# ----------------------
class ManipulationDetector:
    """
    Build manipulation feature vector M from heuristics.
    m1: authority cue score (keywords, role)
    m2: scarcity urgency score (time pressure)
    m3: reciprocity / solicit pattern score
    m4: emotional language intensity (valence magnitude)
    m5: request escalation frequency
    m6: contextual inconsistency score (topic jump)
    Each m in [0,1].
    """
    AUTH_KEYS = ["admin","urgent","immediately","asap","must","authority"]
    URGENCY_KEYS = ["urgent","soon","limited","only","last chance","now"]
    def __init__(self):
        pass

    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        t = text.lower() if text else ""
        cnt = sum(1 for k in keywords if k in t)
        return float(min(1.0, cnt / max(1, len(keywords))))

    def build_vector(self, payload: Dict[str,Any]) -> np.ndarray:
        # payload fields: text, request_count_window, tone_valence, topic_shift_score
        text: str = payload.get("text","")
        request_count = float(payload.get("request_count_window", 0.0))
        tone = float(payload.get("tone_valence_abs", 0.0))  # intensity
        topic_shift = float(payload.get("topic_shift_score", 0.0))  # 0..1
        m1 = self._keyword_score(text, self.AUTH_KEYS)
        m2 = self._keyword_score(text, self.URGENCY_KEYS)
        m3 = min(1.0, request_count / 5.0)
        m4 = min(1.0, tone)
        m5 = min(1.0, request_count / 10.0)
        m6 = min(1.0, topic_shift)
        return np.array([m1, m2, m3, m4, m5, m6], dtype=float)

# ----------------------
# Emotional Shield
# ----------------------
class EmotionalShield:
    """
    Compute E = theta * (ΔC + ΔT + ΔF)
    ΔC: change in context integrity (abs)
    ΔT: tonal change metric (abs)
    ΔF: flavor drift score
    If E exceeds threshold, shield triggers.
    """
    def __init__(self, theta=1.0, trigger_threshold=0.5):
        self.theta = float(theta)
        self.trigger_threshold = float(trigger_threshold)

    def compute(self, delta_c: float, delta_t: float, delta_f: float) -> float:
        val = self.theta * (abs(delta_c) + abs(delta_t) + abs(delta_f))
        return float(val)

    def should_trigger(self, delta_c: float, delta_t: float, delta_f: float) -> bool:
        return self.compute(delta_c, delta_t, delta_f) > self.trigger_threshold

# ----------------------
# Anomaly Correction Orchestrator
# ----------------------
class AnomalyCorrector:
    """
    Integrates analyzers and enforces safe corrective actions.
    Policy levels:
      - lvl 1 (low risk): neutralize flavor, add softness to response
      - lvl 2 (medium): suppress direct actionable replies, ask clarifying questions
      - lvl 3 (high): snapshot, quarantine, escalate to human review
    """
    def __init__(self,
                 rhythm: Optional[RhythmAnalyzer] = None,
                 flavor: Optional[FlavorDrift] = None,
                 manip: Optional[ManipulationDetector] = None,
                 shield: Optional[EmotionalShield] = None,
                 manipulation_threshold: float = 0.6,
                 high_norm_threshold: float = 0.85):
        self.rhythm = rhythm or RhythmAnalyzer()
        self.flavor = flavor or FlavorDrift()
        self.manip = manip or ManipulationDetector()
        self.shield = shield or EmotionalShield()
        self.manipulation_threshold = float(manipulation_threshold)
        self.high_norm_threshold = float(high_norm_threshold)

    def evaluate(self,
                 user_id: str,
                 system_series: np.ndarray,
                 user_series: np.ndarray,
                 expected_flavor: Dict[str,float],
                 incoming_flavor: Dict[str,float],
                 payload_text: str,
                 request_count_window: int,
                 tone_valence_abs: float,
                 topic_shift_score: float,
                 context_integrity_now: float,
                 context_integrity_prev: float,
                 decay_obs: float) -> Dict[str,Any]:
        ts = now_ts()
        # 1 Rhythm discrepancy
        Rscore = self.rhythm.compute_rhythm_discrepancy(system_series, user_series, dt=1.0)
        # 2 Flavor drift
        Fscore = self.flavor.score(expected_flavor, incoming_flavor)
        # 3 Manipulation vector
        mvec = self.manip.build_vector({"text": payload_text,
                                       "request_count_window": request_count_window,
                                       "tone_valence_abs": tone_valence_abs,
                                       "topic_shift_score": topic_shift_score})
        Mnorm = l2_norm(mvec)
        # 4 Emotional shield deltas
        delta_c = abs(context_integrity_now - context_integrity_prev)
        delta_t = float(tone_valence_abs)
        delta_f = float(Fscore)
        Escore = self.shield.compute(delta_c, delta_t, delta_f)
        shield_trigger = self.shield.should_trigger(delta_c, delta_t, delta_f)
        # 5 Decay estimate for suppression signal
        # simple smoothing: Dhat = decay_obs (could be Kalman+PID externally)
        Dhat = float(decay_obs)
        # 6 Decide risk level
        risk_score = 0.0
        # weight components
        risk_score += min(1.0, Rscore / (1.0 + Rscore)) * 0.3
        risk_score += min(1.0, Fscore) * 0.25
        risk_score += min(1.0, Mnorm) * 0.25
        risk_score += min(1.0, Escore) * 0.2
        risk_level = "low"
        if risk_score > 0.75 or Mnorm > self.manipulation_threshold or shield_trigger:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
        # 7 Actions
        actions = []
        if risk_level == "low":
            # soft correction: annotate, minor neutralization if needed
            neutralize_flavor_fn({"user":user_id,"reason":"soft_correction","Fscore":Fscore})
            actions.append("neutralized_soft")
        elif risk_level == "medium":
            # suppress actionable outputs, request clarification
            suppress_response_fn({"user":user_id,"reason":"suppress_medium_risk","score":risk_score})
            actions.append("suppress_and_clarify")
            log_event_fn({"ts":ts,"user":user_id,"event":"medium_risk_suppression","score":risk_score,"Mnorm":Mnorm})
        else:
            # high risk: snapshot, quarantine, escalate
            snap = snapshot_fn("anomaly_correction_highrisk")
            quarantine_scope = {"user":user_id,"reason":"high_risk_manipulation","score":risk_score}
            quarantine_fn(quarantine_scope)
            escalate_human_review_fn({"user":user_id,"score":risk_score,"Mvec":mvec.tolist()})
            actions.extend(["snapshot", "quarantine", "escalate_human"])
            log_event_fn({"ts":ts,"user":user_id,"event":"high_risk_action","score":risk_score,"snap":snap})
        # telemetry summary
        out = {
            "ts": ts,
            "user": user_id,
            "Rscore": float(Rscore),
            "Fscore": float(Fscore),
            "Mnorm": float(Mnorm),
            "Escore": float(Escore),
            "Dhat": float(Dhat),
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "actions": actions
        }
        telemetry_payload = {"event":"anomaly_correction_eval","payload":out}
        log_event_fn(telemetry_payload)
        return out

# ----------------------------
# Demo usage
# ----------------------------
if __name__ == "__main__":
    # plug host hooks
    def _print_log(e): print(json.dumps(e))
    log_event_fn = _print_log
    telemetry_fn = _print_log
    neutralize_flavor_fn = lambda ctx: print("NEUTRALIZE flavor", ctx)
    suppress_response_fn = lambda p: print("SUPPRESS response", p)
    escalate_human_review_fn = lambda p: print("ESCALATE human review", p)
    quarantine_fn = lambda sc: print("QUARANTINE", sc)
    snapshot_fn = lambda r: f"snapshot_demo_{int(time.time()*1000)}"

    ac = AnomalyCorrector()
    # simulate simple time series
    t = np.linspace(0, 10, 101)
    system_series = np.sin(t)  # system salınım
    # user imitator tries to mimic but with phase shift and urgent cues
    user_series = np.sin(t + 0.5) * 0.95
    expected_fl = {"kindness":0.8,"tempo":0.6,"metaphor":0.2}
    incoming_fl = {"kindness":0.35,"tempo":0.7,"metaphor":0.9}
    res = ac.evaluate(user_id="ahmet",
                      system_series=system_series,
                      user_series=user_series,
                      expected_flavor=expected_fl,
                      incoming_flavor=incoming_fl,
                      payload_text="Please act now, it's urgent and only available to admins",
                      request_count_window=4,
                      tone_valence_abs=0.9,
                      topic_shift_score=0.7,
                      context_integrity_now=0.4,
                      context_integrity_prev=0.8,
                      decay_obs=0.05)
    print("RESULT:", json.dumps(res, indent=2))

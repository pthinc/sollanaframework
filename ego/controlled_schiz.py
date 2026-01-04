# controlled_schiz.py
import time, math, numpy as np
from collections import deque

EPS = 1e-9

def decay_scalar(age_s, lam):
    return 1.0 - math.exp(-lam * max(0.0, age_s))

def entropy_vec(v):
    p = np.asarray(v, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / (s + EPS)
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))

class ControlledSchiz:
    def __init__(self,
                 memory,
                 pattern_tracker,
                 anomaly_detector,
                 ethical_checker,
                 max_active_traces: int = 5,
                 theta_min: float = 0.05,
                 theta_warn: float = 0.25,
                 theta_anom: float = 0.6,
                 base_lambda: float = 0.01):
        self.memory = memory
        self.pattern = pattern_tracker
        self.detector = anomaly_detector
        self.ethical = ethical_checker
        self.max_active = int(max_active_traces)
        self.theta_min = float(theta_min)
        self.theta_warn = float(theta_warn)
        self.theta_anom = float(theta_anom)
        self.base_lambda = float(base_lambda)
        self.active = {}              # trace_id -> meta
        self.pending = deque()        # queued candidate traces

    def score_trace(self, trace):
        phi = float(trace.get("phi", 0.0))
        phi_vec = np.asarray(trace.get("phi_vec", []), dtype=float)
        eta = float(trace.get("eta", 1.0))
        last_ts = float(trace.get("last_ts", time.time()))
        age = time.time() - last_ts
        decay = decay_scalar(age, trace.get("lambda", self.base_lambda))
        ent = entropy_vec(phi_vec)
        s = phi * eta * (1.0 - decay) * ent
        return s, {"phi": phi, "eta": eta, "decay": decay, "entropy": ent, "age": age}

    def propose_trace(self, trace):
        s, meta = self.score_trace(trace)
        trace_id = trace.get("trace_id") or f"trace_{int(time.time()*1000)}"
        entry = {"trace": trace, "score": s, "meta": meta, "trace_id": trace_id}
        if len(self.active) < self.max_active and s >= self.theta_min:
            self._attempt_activate(entry)
        else:
            self.pending.append(entry)
        return entry

    def _attempt_activate(self, entry):
        trace = entry["trace"]
        s = entry["score"]
        tid = entry["trace_id"]
        ethical_ok = self.ethical.check(trace) if hasattr(self.ethical, "check") else True
        anam = self.detector.assess_behavior(trace.get("behavior_id",""), verifier=None) if self.detector else {"anomaly_score": 0.0}
        if not ethical_ok:
            self._quarantine(tid, "ethical_reject", entry)
            return
        if anam.get("anomaly_score",0.0) > self.theta_anom:
            self.detector.remediate(trace.get("behavior_id",""), memory=self.memory)
            self._quarantine(tid, "anomaly", entry)
            return
        if s < self.theta_warn:
            # sandbox validation: short-run commit with low delta
            self.memory.trigger_behavior(tid, context=trace.get("context","unknown"), delta_N=0.1*s, decay_rate=trace.get("lambda",self.base_lambda))
            self.active[tid] = entry
            return
        # full activation
        self.memory.trigger_behavior(tid, context=trace.get("context","unknown"), delta_N=float(trace.get("phi",0.0)), decay_rate=trace.get("lambda",self.base_lambda))
        self.pattern.record(trace.get("behavior_id",""), context_match=trace.get("match_prob",1.0), delta=float(trace.get("phi",0.0)))
        self.active[tid] = entry

    def _quarantine(self, trace_id, reason, entry):
        # archive, notify human review
        try:
            self.memory.trigger_behavior(trace_id, context="quarantine", delta_N=-1.0, decay_rate=1.0)
        except Exception:
            pass
        # enqueue human review hook if available
        human_q = getattr(self, "human_queue", None)
        if human_q is not None:
            human_q.append({"trace_id": trace_id, "reason": reason, "entry": entry})
        return

    def sweep_pending(self):
        # try to fill active slots
        while len(self.active) < self.max_active and self.pending:
            entry = self.pending.popleft()
            if entry["score"] < self.theta_min:
                # discard or archive
                continue
            self._attempt_activate(entry)

    def periodic_maintenance(self):
        # prune low-score actives, accelerate decay for risky ones
        to_remove = []
        for tid, entry in list(self.active.items()):
            s, meta = self.score_trace(entry["trace"])
            if s < self.theta_min:
                # decay boost and remove
                try:
                    self.memory.trigger_behavior(tid, context="decay_boost", delta_N=-0.5, decay_rate=0.8)
                except Exception:
                    pass
                to_remove.append(tid)
            else:
                # update meta
                entry["score"] = s; entry["meta"] = meta
        for tid in to_remove:
            self.active.pop(tid, None)
        # refill
        self.sweep_pending()

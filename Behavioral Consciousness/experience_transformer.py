# experience_transformer.py
import time, json, tempfile, os
from typing import Dict, Any, Optional, Callable
import math

# ---------- helpers ----------
EPS = 1e-12

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

def atomic_write(path: str, obj: Any) -> None:
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
            except: pass

# ---------- heuristic mappers ----------
def map_emotional_cluster(context: str, resonance: float, meta: Dict[str,Any]) -> str:
    # Heuristic mapping; replace with classifier for production
    if resonance > 0.85:
        if "grat" in context.lower() or meta.get("intent")=="appreciate":
            return "Şükran"
        return "Saygı"
    if resonance > 0.6:
        return "Merak"
    if resonance > 0.3:
        return "Hüzün"
    return "Nötr"

def select_flavor(context: str, resonance: float, flavor_protos: Dict[str, float]) -> str:
    # flavor_protos: name->prototype_score or embedding similarity
    # simple rank by (prototype_score * resonance)
    best = None; best_score = -1.0
    for name, score in flavor_protos.items():
        s = score * float(resonance)
        if s > best_score:
            best_score = s; best = name
    return best or "neutral"

def summarize_memory_trace(x_t: Dict[str,Any]) -> str:
    # short human readable summary
    c = x_t.get("context","")
    r = float(x_t.get("resonance",0.0))
    d = float(x_t.get("decay_risk",0.0))
    return f"Context='{c}', resonance={r:.3f}, decay={d:.3f}"

# ---------- main transformer ----------
class ExperienceTransformer:
    def __init__(self,
                 flavor_prototypes: Optional[Dict[str,float]] = None,
                 decay_threshold: float = 0.3,
                 decay_policy_fn: Optional[Callable[[Dict[str,Any], Dict[str,Any]], Dict[str,Any]]] = None,
                 superego_check: Optional[Callable[[Dict[str,Any]],bool]] = None,
                 memory_commit: Optional[Callable[[Dict[str,Any]],str]] = None,
                 pattern_update: Optional[Callable[[Dict[str,Any]],None]] = None,
                 telemetry_emit: Optional[Callable[[Dict[str,Any]],None]] = None,
                 output_dir: str = "bce_traces"):
        self.flavor_prototypes = flavor_prototypes or {"aesthetic_resonance":0.9,"gentle":0.6,"playful":0.7,"sober":0.5}
        self.decay_threshold = float(decay_threshold)
        self.decay_policy_fn = decay_policy_fn
        self.superego_check = superego_check
        self.memory_commit = memory_commit
        self.pattern_update = pattern_update
        self.telemetry_emit = telemetry_emit or (lambda e: None)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def transform(self, user_id: str, x_t: Dict[str,Any], pi_constant: str = "stable", meta: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        meta = meta or {}
        # sanitize inputs
        decay = float(x_t.get("decay_risk", 0.0))
        resonance = float(x_t.get("resonance", 0.0))
        norm_align = bool(x_t.get("norm_alignment", False))

        # optional decay policy adjustment
        if callable(self.decay_policy_fn):
            try:
                pol_res = self.decay_policy_fn({"score": max(0.0, 1.0 - decay), "lambda_base": x_t.get("lambda_base", 0.1)}, x_t.get("decay_signals", {}))
                if isinstance(pol_res, dict) and "score" in pol_res:
                    decay = max(0.0, 1.0 - float(pol_res["score"]))
            except Exception:
                pass

        # decay filtering logic
        decay_filtered = decay <= self.decay_threshold

        # if decay too high, mark quarantined experience
        status = "ready" if decay_filtered else "quarantined"

        # emotional cluster
        emotional_cluster = map_emotional_cluster(x_t.get("context",""), resonance, meta)

        # flavor selection (respect superego)
        candidate_flavor = select_flavor(x_t.get("context",""), resonance, self.flavor_prototypes)
        allowed = True
        if callable(self.superego_check):
            try:
                allowed = self.superego_check({"user_id": user_id, "flavor": candidate_flavor, "resonance": resonance})
            except Exception:
                allowed = False
        flavor = candidate_flavor if allowed else "gentle_fallback"

        # memory trace summarization
        memory_trace = summarize_memory_trace(x_t)

        # build ΔSelf
        delta_self = {
            "emotional_cluster": emotional_cluster,
            "flavor": flavor,
            "memory_trace": memory_trace,
            "decay_filtered": decay_filtered
        }

        # assemble BCE JSON
        bce_obj = {
            "bce_trace": {
                "timestamp": now_iso(),
                "user_id": user_id,
                "x_t": x_t,
                "pi_sabiti": pi_constant,
                "experience_vector": {"delta_self_t": delta_self},
                "write_status": "pending" if status=="ready" else "quarantined",
                "read_status": "unavailable"
            }
        }

        # if ready, commit
        if status == "ready":
            # atomic write
            fname = os.path.join(self.output_dir, f"bce_{user_id}_{int(time.time()*1000)}.json")
            try:
                atomic_write(fname, bce_obj)
                bce_obj["bce_trace"]["write_status"] = "committed"
                bce_obj["bce_trace"]["read_status"] = "available"
                # invoke memory_commit hook if provided
                if callable(self.memory_commit):
                    try:
                        commit_id = self.memory_commit(bce_obj)
                        bce_obj["bce_trace"]["memory_commit_id"] = commit_id
                    except Exception:
                        pass
                # pattern update hook
                if callable(self.pattern_update):
                    try:
                        self.pattern_update({"user_id": user_id, "delta": delta_self})
                    except Exception:
                        pass
            except Exception as e:
                bce_obj["bce_trace"]["write_status"] = "error"
                bce_obj["bce_trace"]["error"] = str(e)
        else:
            # quarantined: write quarantined log (not committed to main memory)
            qname = os.path.join(self.output_dir, f"quarantine_{user_id}_{int(time.time()*1000)}.json")
            try:
                atomic_write(qname, bce_obj)
            except Exception:
                pass

        # telemetry
        self.telemetry_emit({
            "event": "experience_transformed",
            "user_id": user_id,
            "decay": decay,
            "resonance": resonance,
            "decay_filtered": decay_filtered,
            "flavor": flavor
        })

        return bce_obj

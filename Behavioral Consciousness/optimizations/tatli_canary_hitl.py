# tatli_canary_hitl.py
# Single-file reference: HITL approval ledger + canary automation + audit
# Requires: Python 3.8+, standard lib only for core; replace hooks with infra endpoints.

import time, json, os, threading, hashlib
from typing import Dict, List, Any, Callable, Optional

# ---------- Config ----------
AUDIT_DIR = "audit_ledger"
os.makedirs(AUDIT_DIR, exist_ok=True)
CANARY_METRIC_THRESHOLDS = {"latency_p95":0.5, "error_rate":0.01, "throughput":0.0}  # example
APPROVAL_TIMEOUT = 48*3600  # seconds
CANARY_SEGMENT = "canary_group_1"

# ---------- Utilities ----------
def now_ts(): return time.time()
def iso_ts(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def audit_write(event: Dict[str,Any]):
    fname = os.path.join(AUDIT_DIR, f"audit_{int(now_ts()*1000)}.json")
    with open(fname,"w",encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)

def hash_payload(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

# ---------- Approval ledger (2-person) ----------
class TwoPersonApproval:
    def __init__(self, audit_fn: Callable[[Dict],None]=audit_write, timeout: int = APPROVAL_TIMEOUT):
        self.store: Dict[str, Dict] = {}  # recon_id -> record
        self.lock = threading.Lock()
        self.audit = audit_fn
        self.timeout = int(timeout)

    def create_request(self, recon_id: str, recon_type: str, payload: Dict[str,Any], required: List[str]):
        rec = {
            "recon_id": recon_id,
            "recon_type": recon_type,
            "payload_hash": hash_payload(payload),
            "required": list(required),
            "approvals": {},
            "created_ts": now_ts(),
            "deadline_ts": now_ts()+self.timeout,
            "status": "pending"
        }
        with self.lock:
            self.store[recon_id] = rec
        self.audit({"event":"approval_created","recon_id":recon_id,"required":required,"recon_type":recon_type,"ts":iso_ts()})
        return rec

    def approve(self, recon_id: str, user: str, decision: bool, reason: Optional[str]=None):
        with self.lock:
            rec = self.store.get(recon_id)
            if not rec:
                return {"status":"error","reason":"not_found"}
            if user not in rec["required"]:
                return {"status":"error","reason":"not_authorized"}
            rec["approvals"][user] = {"decision":bool(decision),"reason":reason,"ts":iso_ts()}
            self.audit({"event":"approval_recorded","recon_id":recon_id,"user":user,"decision":decision,"reason":reason,"ts":iso_ts()})
            # evaluate
            all_present = all(u in rec["approvals"] for u in rec["required"])
            if all_present and all(rec["approvals"][u]["decision"] for u in rec["required"]):
                rec["status"] = "approved"
                self.audit({"event":"approval_fulfilled","recon_id":recon_id,"ts":iso_ts()})
                return {"status":"approved"}
            # if any denial -> rejected
            if any(u in rec["approvals"] and not rec["approvals"][u]["decision"] for u in rec["approvals"]):
                rec["status"] = "rejected"
                self.audit({"event":"approval_rejected","recon_id":recon_id,"ts":iso_ts()})
                return {"status":"rejected"}
            return {"status":"pending","approvals":rec["approvals"]}

    def check_timeout(self, recon_id: str):
        with self.lock:
            rec = self.store.get(recon_id)
            if not rec: return {"status":"not_found"}
            if now_ts() > rec["deadline_ts"]:
                rec["status"] = "timeout"
                self.audit({"event":"approval_timeout","recon_id":recon_id,"ts":iso_ts()})
                return {"status":"timeout"}
            return {"status":"waiting","deadline_ts": rec["deadline_ts"]}

# ---------- Canary monitor ----------
class CanaryMonitor:
    def __init__(self, thresholds: Dict[str,float], on_rollback: Callable[[str,Dict],None], audit_fn: Callable[[Dict],None]=audit_write):
        self.thresholds = dict(thresholds)
        self.on_rollback = on_rollback
        self.audit = audit_fn

    def evaluate(self, canary_id: str, metrics: Dict[str,float]):
        violations = {k:v for k,v in metrics.items() if k in self.thresholds and v > self.thresholds[k]}
        rec = {"event":"canary_eval","canary_id":canary_id,"metrics":metrics,"violations":violations,"ts":iso_ts()}
        self.audit(rec)
        if violations:
            # log and trigger rollback
            self.audit({"event":"canary_violation","canary_id":canary_id,"violations":violations,"ts":iso_ts()})
            self.on_rollback(canary_id, {"violations":violations,"metrics":metrics,"ts":iso_ts()})
            return {"action":"rollback","violations":violations}
        return {"action":"ok"}

# ---------- Promotion pipeline ----------
class PromotionPipeline:
    def __init__(self, deploy_fn: Callable[[str,Dict],None], monitor_fn: Callable[[str],Dict], rollback_fn: Callable[[str],None], audit_fn: Callable[[Dict],None]=audit_write):
        self.deploy_fn = deploy_fn
        self.monitor_fn = monitor_fn
        self.rollback_fn = rollback_fn
        self.audit = audit_fn

    def promote_sequence(self, config_id: str, config_spec: Dict, stages: List[str], thresholds: Dict[str,float]):
        prev_stage = None
        for stage in stages:
            self.audit({"event":"deploy_stage_start","config":config_id,"stage":stage,"ts":iso_ts()})
            self.deploy_fn(stage, config_spec)
            metrics = self.monitor_fn(stage)
            # evaluate
            bad = any(metrics.get(m,0) > thresholds.get(m, float("inf")) for m in thresholds)
            self.audit({"event":"deploy_stage_metrics","config":config_id,"stage":stage,"metrics":metrics,"bad":bad,"ts":iso_ts()})
            if bad:
                # rollback to prev
                if prev_stage:
                    self.rollback_fn(prev_stage)
                    self.audit({"event":"auto_rollback","from_stage":stage,"to_stage":prev_stage,"config":config_id,"ts":iso_ts()})
                else:
                    self.rollback_fn(stage)
                    self.audit({"event":"auto_rollback","from_stage":stage,"to_stage":"none","config":config_id,"ts":iso_ts()})
                return {"status":"blocked","stage":stage,"metrics":metrics}
            prev_stage = stage
        self.audit({"event":"promote_success","config":config_id,"stages":stages,"ts":iso_ts()})
        return {"status":"promoted","config":config_id}

# ---------- System hooks (stubs: replace with infra) ----------
def deploy_stub(stage: str, config_spec: Dict):
    # implement actual deployment for stage: alpha,beta,shadow,canary,prod
    print(f"[deploy_stub] deploy {stage}")

def monitor_stub(stage: str) -> Dict[str,float]:
    # return metric snapshot for stage; replace with metrics query
    # example mock: random below thresholds
    import random
    return {"latency_p95": random.random()*0.4, "error_rate": random.random()*0.005, "throughput": random.random()*100}

def rollback_stub(stage: str):
    print(f"[rollback_stub] rollback to {stage}")

def on_rollback_action(canary_id: str, info: Dict):
    # provoke test mode
    audit_write({"event":"provoke_test_start","canary":canary_id,"info":info,"ts":iso_ts()})
    # sample synthetic tests here; escalate human alert
    audit_write({"event":"escalate_human","canary":canary_id,"info":info,"ts":iso_ts()})

# ---------- Example wiring ----------
if __name__ == "__main__":
    # approval usage
    apr = TwoPersonApproval()
    req = apr.create_request("recon_123", "critical_recon", {"change":"restore_rare_items"}, required=["alice","bob"])
    print("created", req)
    print("alice approves:", apr.approve("recon_123","alice", True, "ok"))
    print("bob approves:", apr.approve("recon_123","bob", True, "ok"))
    # canary monitor usage
    cm = CanaryMonitor(CANARY_METRIC_THRESHOLDS, on_rollback_action)
    print("canary eval:", cm.evaluate(CANARY_SEGMENT, monitor_stub("canary")))
    # promotion pipeline usage
    pipeline = PromotionPipeline(deploy_stub, monitor_stub, rollback_stub)
    stages = ["alpha","beta","shadow","canary","prod"]
    res = pipeline.promote_sequence("confA", {"spec":"x"}, stages, CANARY_METRIC_THRESHOLDS)
    print("promote result", res)

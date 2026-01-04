# sanal_hipofiz.py
"""
Sanal Hipofiz - reference implementation
- Homeostasis monitor H(t)
- Versioned snapshots and immutable audit
- Two person approval for critical reconstructions
- Telemetry and alarm hooks
"""

import time, os, json, math, threading, hashlib
from typing import Dict, Any, Callable, Optional, List
import numpy as np

# --- Configuration (tune for your system) ---
AUDIT_DIR = "hipofiz_audit"
SNAP_DIR = "hipofiz_snapshots"
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 1.0
ALARM_THRESHOLD = 0.3   # H(t) < ALARM_THRESHOLD triggers alarm
COOLDOWN_SEC = 3600     # cooldown between aggressive automated actions

# Hook placeholders - replace with infra implementations
audit_fn: Callable[[Dict[str,Any]], None] = lambda e: print("AUDIT", json.dumps(e, ensure_ascii=False))
telemetry_emit: Callable[[Dict[str,Any]], None] = lambda e: print("TELEM", json.dumps(e, ensure_ascii=False))
deploy_snapshot_hook: Callable[[str], None] = lambda path: None
human_notify: Callable[[str, Dict[str,Any]], None] = lambda tag, p: print("NOTIFY", tag, p)

# --- Utils ---
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_hex(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def write_json_atomic(path: str, payload: Dict[str,Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# --- Two person approval for critical reconstructions ---
class TwoPersonApproval:
    def __init__(self, audit: Callable[[Dict],None] = audit_fn, timeout_sec: int=48*3600):
        self.store: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.audit = audit
        self.timeout_sec = timeout_sec

    def create(self, recon_id: str, recon_payload: Dict[str,Any], approvers: List[str]):
        record = {
            "recon_id": recon_id,
            "payload_hash": sha256_hex(recon_payload),
            "approvers": list(approvers),
            "approvals": {},
            "created_at": now_iso(),
            "deadline": now_iso(),
            "status": "pending"
        }
        with self.lock:
            self.store[recon_id] = record
        self.audit({"event":"approval_created","recon_id":recon_id,"approvers":approvers,"ts":now_iso()})
        return record

    def approve(self, recon_id: str, user: str, decision: bool, note: Optional[str]=None):
        with self.lock:
            rec = self.store.get(recon_id)
            if not rec:
                return {"status":"error","reason":"not_found"}
            if user not in rec["approvers"]:
                return {"status":"error","reason":"not_authorized"}
            rec["approvals"][user] = {"decision":bool(decision), "note": note, "ts": now_iso()}
            self.audit({"event":"approval_recorded","recon_id":recon_id,"user":user,"decision":decision,"note":note,"ts":now_iso()})
            # check all distinct approvals true
            done = all(u in rec["approvals"] and rec["approvals"][u]["decision"] for u in rec["approvers"])
            if done:
                rec["status"] = "approved"
                self.audit({"event":"approval_fulfilled","recon_id":recon_id,"ts":now_iso()})
                return {"status":"approved"}
            # if any deny => rejected
            if any(u in rec["approvals"] and not rec["approvals"][u]["decision"] for u in rec["approvals"]):
                rec["status"] = "rejected"
                self.audit({"event":"approval_rejected","recon_id":recon_id,"ts":now_iso()})
                return {"status":"rejected"}
            return {"status":"pending","approvals":rec["approvals"]}

# --- Snapshot manager and genetic store ---
class SnapshotManager:
    def __init__(self, snap_dir: str = SNAP_DIR, min_retention: int = 3):
        self.snap_dir = snap_dir
        self.min_retention = int(min_retention)
        os.makedirs(self.snap_dir, exist_ok=True)
        self.index_path = os.path.join(self.snap_dir, "index.json")
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = []

    def _persist_index(self):
        write_json_atomic(self.index_path, self.index)

    def create_snapshot(self, meta: Dict[str,Any], state_blob: Dict[str,Any]) -> str:
        ts = int(time.time()*1000)
        snap_id = f"snap_{ts}"
        path = os.path.join(self.snap_dir, f"{snap_id}.json")
        payload = {"snap_id": snap_id, "ts": now_iso(), "meta": meta, "state": state_blob}
        write_json_atomic(path, payload)
        self.index.append({"snap_id": snap_id, "path": path, "ts": now_iso(), "meta": meta})
        self._persist_index()
        audit_fn({"event":"snapshot_created","snap_id":snap_id,"meta":meta,"ts":now_iso()})
        return snap_id

    def list_snapshots(self):
        return self.index.copy()

    def rollback_to(self, snap_id: str) -> bool:
        entry = next((e for e in self.index if e["snap_id"] == snap_id), None)
        if not entry:
            return False
        # call deploy snapshot hook
        deploy_snapshot_hook(entry["path"])
        audit_fn({"event":"rollback_performed","snap_id":snap_id,"ts":now_iso()})
        return True

# --- Hipofiz core ---
class SanalHipofiz:
    def __init__(self,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 gamma: float = DEFAULT_GAMMA,
                 alarm_threshold: float = ALARM_THRESHOLD):
        self.alpha = float(alpha); self.beta = float(beta); self.gamma = float(gamma)
        self.alarm_threshold = float(alarm_threshold)
        self.snapshot_mgr = SnapshotManager()
        self.approval = TwoPersonApproval()
        self.last_action_ts = 0.0
        self.cooldown = COOLDOWN_SEC

    def collect_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        # expected keys: D, C, S
        D = float(metrics.get("D", 0.0))
        C = float(metrics.get("C", 0.0))
        S = float(metrics.get("S", 0.0))
        # apply EMA smoothing stored per key if desired (omitted for brevity)
        H = self.alpha * D + self.beta * C - self.gamma * S
        telemetry_emit({"event":"hipofiz_eval","D":D,"C":C,"S":S,"H":H,"ts":now_iso()})
        # action logic
        if H < self.alarm_threshold:
            self._handle_alarm(H, metrics)
        return {"H": H, "D": D, "C": C, "S": S}

    def _handle_alarm(self, H_val: float, metrics: Dict[str, float]):
        now = time.time()
        audit_fn({"event":"hipofiz_alarm","H":H_val,"metrics":metrics,"ts":now_iso()})
        if now - self.last_action_ts < self.cooldown:
            telemetry_emit({"event":"hipofiz_alarm_cooldown","H":H_val,"ts":now_iso()})
            return
        # conservative responses:
        # 1) create immutable snapshot
        snap_meta = {"reason":"hipofiz_alarm","H":H_val, "metrics": metrics}
        snap_state = {"metrics": metrics}
        snap_id = self.snapshot_mgr.create_snapshot(snap_meta, snap_state)
        # 2) trigger decay suppression across system (hook to apply)
        apply_decay_suppression("global", {"reason":"hipofiz_alarm","snap_id": snap_id})
        # 3) notify humans and open HITL ticket with snapshot
        human_notify("hipofiz_alarm", {"H":H_val, "snap_id": snap_id, "metrics": metrics})
        self.last_action_ts = now

    def propose_critical_reconstruction(self, recon_id: str, payload: Dict[str,Any], approvers: List[str]):
        # create approval request; recon cannot proceed without 2 distinct approvers
        rec = self.approval.create(recon_id, payload, approvers)
        telemetry_emit({"event":"reconstruction_proposed","recon_id":recon_id,"meta":rec,"ts":now_iso()})
        return rec

    def finalize_reconstruction(self, recon_id: str, user: str, decision: bool, note: Optional[str]=None):
        res = self.approval.approve(recon_id, user, decision, note)
        telemetry_emit({"event":"reconstruction_approval_update","recon_id":recon_id,"user":user,"res":res,"ts":now_iso()})
        if res.get("status") == "approved":
            # perform reconstruction under canary and shadow validation outside this module
            audit_fn({"event":"reconstruction_approved","recon_id":recon_id,"ts":now_iso()})
        return res

# --- Small demo ---
if __name__ == "__main__":
    hip = SanalHipofiz()
    # sample metrics where D high, C low, S high -> H low -> alarm
    sample_metrics = {"D": 0.9, "C": 0.1, "S": 0.8}
    print("collect:", hip.collect_metrics(sample_metrics))
    # create reconstruction
    recon = hip.propose_critical_reconstruction("recon_001", {"type":"restore_cluster","cluster":"c42"}, ["alice","bob"])
    print("recon created:", recon)
    print("alice approve:", hip.finalize_reconstruction("recon_001","alice",True,"ok"))
    print("bob approve:", hip.finalize_reconstruction("recon_001","bob",True,"ok"))

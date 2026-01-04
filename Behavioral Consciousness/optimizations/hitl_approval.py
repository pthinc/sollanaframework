# hitl_approval.py
import time, json, threading
from typing import Dict, List, Callable, Optional

class TwoPersonApproval:
    def __init__(self, auditors: List[str], timeout_seconds: int = 48*3600, audit_log: Callable[[Dict],None]=print):
        self.auditors = set(auditors)
        self.timeout = int(timeout_seconds)
        self.audit_log = audit_log
        self.lock = threading.Lock()
        self.store: Dict[str, Dict] = {}  # recon_id -> {"approvals":{user:bool}, "ts":, "deadline":}
    def create_request(self, recon_id: str, required: List[str]):
        with self.lock:
            self.store[recon_id] = {"approvals": {}, "required": list(required), "ts": time.time(), "deadline": time.time()+self.timeout}
            self.audit_log({"event":"approval_created","recon_id":recon_id,"required":required,"ts":time.time()})
    def approve(self, recon_id: str, user: str, decision: bool, reason: Optional[str]=None) -> Dict:
        with self.lock:
            r = self.store.get(recon_id)
            if not r:
                return {"status":"error","reason":"not_found"}
            if user not in r["required"]:
                return {"status":"error","reason":"not_authorized"}
            r["approvals"][user] = {"decision":bool(decision),"reason":reason,"ts":time.time()}
            self.audit_log({"event":"approval_recorded","recon_id":recon_id,"user":user,"decision":decision,"reason":reason,"ts":time.time()})
            # check if all distinct and all True
            if all(u in r["approvals"] and r["approvals"][u]["decision"] for u in r["required"]):
                return {"status":"approved"}
            return {"status":"pending","approvals":r["approvals"]}
    def check_timeout(self, recon_id: str) -> Dict:
        with self.lock:
            r = self.store.get(recon_id)
            if not r: return {"status":"not_found"}
            if time.time() > r["deadline"]:
                self.audit_log({"event":"approval_timeout","recon_id":recon_id,"ts":time.time()})
                return {"status":"timeout"}
            return {"status":"waiting","deadline":r["deadline"]}

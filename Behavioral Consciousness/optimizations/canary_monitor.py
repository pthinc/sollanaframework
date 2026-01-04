# canary_monitor.py
import time
from typing import Dict, Callable, List

class CanaryMonitor:
    def __init__(self, metrics_thresholds: Dict[str,float], on_rollback: Callable[[str,Dict],None], telemetry: Callable[[Dict],None]=print):
        self.thresholds = metrics_thresholds
        self.on_rollback = on_rollback
        self.telemetry = telemetry
    def evaluate(self, canary_id: str, metrics: Dict[str,float]):
        # metrics: {m1:val, m2:val, ...}
        violations = {k:v for k,v in metrics.items() if k in self.thresholds and v > self.thresholds[k]}
        self.telemetry({"event":"canary_evaluate","canary_id":canary_id,"metrics":metrics,"violations":violations,"ts":time.time()})
        if violations:
            self.on_rollback(canary_id, {"violations":violations,"metrics":metrics,"ts":time.time()})
            return {"action":"rollback","violations":violations}
        return {"action":"ok"}

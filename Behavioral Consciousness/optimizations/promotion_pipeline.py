# promotion_pipeline.py
import time
from typing import Dict, Callable, Optional, List

class PromotionPipeline:
    def __init__(self, deploy_fn: Callable[[str,Dict],None], monitor_fn: Callable[[str],Dict], rollback_fn: Callable[[str],None], telemetry: Callable[[Dict],None]=print):
        self.deploy = deploy_fn
        self.monitor = monitor_fn
        self.rollback = rollback_fn
        self.telemetry = telemetry
        self.history = []
    def promote(self, config_id: str, config_spec: Dict, stages: List[str], thresholds: Dict[str,float]):
        prev = None
        for stage in stages:
            self.telemetry({"event":"deploy_stage","config":config_id,"stage":stage,"ts":time.time()})
            self.deploy(stage, config_spec)
            metrics = self.monitor(stage)
            # check
            bad = any(metrics.get(k,0) > thresholds.get(k, float('inf')) for k in thresholds.keys())
            if bad:
                self.telemetry({"event":"promote_blocked","config":config_id,"stage":stage,"metrics":metrics,"ts":time.time()})
                # rollback to prev
                if prev is not None:
                    self.rollback(prev)
                return {"status":"blocked","stage":stage,"metrics":metrics}
            prev = stage
        self.telemetry({"event":"promote_success","config":config_id,"ts":time.time()})
        return {"status":"promoted","config":config_id}

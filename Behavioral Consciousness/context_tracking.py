# context_tracking.py
import time, functools
from typing import Optional, Dict, Any
import json
import os

class ContextTracker:
    def __init__(self, out_dir: str = "context_logs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_context: Dict[str, str] = {}  # channel -> context_id

    def log_transition(self, channel_from: str, channel_to: str):
        now = time.time()
        record = {"ts": now, "from": channel_from, "to": channel_to}
        fname = f"{self.out_dir}/transition_{int(now*1000)}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        # update last_context
        self.last_context[channel_to] = channel_from

    def match(self, context_id: str, context_map: Dict[str, str]) -> bool:
        # δ(ContextID_i, ContextMap_i) -> exact match or prefix rules
        expected = context_map.get(context_id)
        return expected == context_id  # simple δ; replace with richer logic

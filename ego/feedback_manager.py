# feedback_manager.py
import time
from typing import Dict, Any

class FeedbackManager:
    def __init__(self, memory, golden_multiplier: float = 1.0):
        self.memory = memory
        self.golden_multiplier = golden_multiplier
        self.store = {}  # behavior_id -> list of (label, user, ts)

    def submit_feedback(self, behavior_id: str, label: str, user: Optional[str] = None):
        ts = time.time()
        self.store.setdefault(behavior_id, []).append((label, user, ts))
        # adjust memory traces if present
        traces = [tr for tr in self.memory.traces.values() if tr.meta.get("behavior_id") == behavior_id]
        for tr in traces:
            if label in ("true", "approved"):
                tr.touch(delta_N=0.5)
                tr.decay_rate = max(0.0, tr.decay_rate * 0.9)
            else:
                tr.touch(delta_N=-0.5)
                tr.decay_rate = min(1.0, tr.decay_rate * 1.5)
            tr.compute_golden()

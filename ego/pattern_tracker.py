# pattern_tracker.py
"""Pattern tracking with decay windows; backend-agnostic (no torch dependency)."""

import time
import json
from collections import defaultdict, deque, Counter
from typing import Dict, Deque, Tuple, List, Optional

DEFAULT_WINDOW = 3600.0
PHI = (1.0 + 5**0.5) / 2.0

class PatternTracker:
    def __init__(self, window_seconds: float = DEFAULT_WINDOW, ema_alpha: float = 0.1):
        self.window_seconds = window_seconds
        self.ema_alpha = ema_alpha
        self.log: Deque[Tuple[float, str]] = deque()
        self.counts: Dict[str, int] = Counter()
        self.ema_strength: Dict[str, float] = defaultdict(float)
        self.decayed: Dict[str, float] = {}
        self.active: Dict[str, float] = {}

    def record(self, behavior_id: str, delta: float = 1.0, now: Optional[float] = None):
        now = now or time.time()
        self.log.append((now, behavior_id))
        self.counts[behavior_id] += 1
        prev = self.ema_strength.get(behavior_id, 0.0)
        self.ema_strength[behavior_id] = prev * (1.0 - self.ema_alpha) + delta * self.ema_alpha
        self._prune_old(now)

    def _prune_old(self, now: float):
        cutoff = now - self.window_seconds
        while self.log and self.log[0][0] < cutoff:
            ts, bid = self.log.popleft()
            self.counts[bid] -= 1
            if self.counts[bid] <= 0:
                del self.counts[bid]

    def pattern_strength(self, behavior_id: str, usage_count_weight: float = 1.0, decay_rate: float = 0.01) -> float:
        usage = self.counts.get(behavior_id, 0)
        ema = self.ema_strength.get(behavior_id, 0.0)
        golden = (usage * (1.0 - decay_rate)) / PHI
        score = 0.6 * ema + 0.4 * golden
        return float(score)

    def get_patterns(self, threshold: float = 1.0) -> Dict[str, float]:
        out = {}
        for bid in list(self.counts.keys()):
            score = self.pattern_strength(bid)
            if score >= threshold:
                out[bid] = score
                self.active[bid] = score
        return out

    def mark_decayed(self, threshold: float = 0.1):
        to_move = []
        for bid, score in list(self.active.items()):
            if score < threshold:
                to_move.append(bid)
        for bid in to_move:
            self.decayed[bid] = self.active.pop(bid, 0.0)

    def persist(self, active_path: str, decayed_path: str):
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump(self.active, f, ensure_ascii=False, indent=2)
        with open(decayed_path, "w", encoding="utf-8") as f:
            json.dump(self.decayed, f, ensure_ascii=False, indent=2)

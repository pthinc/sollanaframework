# pattern_genetics.py
import time
import json
import os
import tempfile
from collections import defaultdict, deque, Counter
from typing import Dict, Deque, Tuple, Optional

PHI = (1.0 + 5**0.5) / 2.0
PATTERN_DIR = "patterns"
os.makedirs(PATTERN_DIR, exist_ok=True)

class BehaviorPatternTracker:
    def __init__(self, window_seconds: float = 3600.0, ema_alpha: float = 0.12, persist_threshold: float = 1.0):
        self.window_seconds = window_seconds
        self.ema_alpha = ema_alpha
        self.persist_threshold = persist_threshold
        self.log: Deque[Tuple[float, str]] = deque()
        self.counts: Dict[str, int] = Counter()
        self.ema: Dict[str, float] = defaultdict(float)
        self.meta: Dict[str, Dict] = {}

    def _now(self) -> float:
        return time.time()

    def record(self, behavior_id: str, context_match: float = 1.0, delta: float = 1.0, now: Optional[float] = None):
        now = now or self._now()
        self.log.append((now, behavior_id))
        self.counts[behavior_id] += 1
        prev = self.ema.get(behavior_id, 0.0)
        self.ema[behavior_id] = prev * (1.0 - self.ema_alpha) + delta * self.ema_alpha * context_match
        self.meta.setdefault(behavior_id, {"first_seen": now, "last_seen": now})
        self.meta[behavior_id]["last_seen"] = now
        self._prune_old(now)
        score = self.pattern_strength(behavior_id)
        if score >= self.persist_threshold:
            self._persist_pattern(behavior_id, score)

    def _prune_old(self, now: float):
        cutoff = now - self.window_seconds
        while self.log and self.log[0][0] < cutoff:
            ts, bid = self.log.popleft()
            self.counts[bid] -= 1
            if self.counts[bid] <= 0:
                del self.counts[bid]

    def pattern_strength(self, behavior_id: str, decay_rate: float = 0.01) -> float:
        usage = float(self.counts.get(behavior_id, 0))
        ema = float(self.ema.get(behavior_id, 0.0))
        golden = (usage * (1.0 - decay_rate)) / PHI
        score = 0.6 * ema + 0.4 * golden
        return float(score)

    def get_patterns(self, threshold: float = 1.0) -> Dict[str, float]:
        out = {}
        for bid in list(self.counts.keys()):
            score = self.pattern_strength(bid)
            if score >= threshold:
                out[bid] = score
        return out

    def _pattern_path(self, behavior_id: str) -> str:
        return os.path.join(PATTERN_DIR, f"{behavior_id}.bce")

    def _atomic_write(self, path: str, data: str):
        dirn = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=dirn)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass

    def _persist_pattern(self, behavior_id: str, score: float):
        path = self._pattern_path(behavior_id)
        payload = {
            "behavior_id": behavior_id,
            "pattern_score": score,
            "usage_count": int(self.counts.get(behavior_id, 0)),
            "ema_strength": float(self.ema.get(behavior_id, 0.0)),
            "first_seen": self.meta[behavior_id]["first_seen"],
            "last_seen": self.meta[behavior_id]["last_seen"],
            "status": "active",
        }
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def mark_decayed_and_prune(self, golden_threshold: float = 1.0, prune_strength_threshold: float = 1e-3):
        to_delete = []
        now = self._now()
        for bid in list(self.ema.keys()):
            score = self.pattern_strength(bid)
            if score < (golden_threshold / PHI) or self.ema.get(bid, 0.0) < prune_strength_threshold:
                path = self._pattern_path(bid)
                if os.path.exists(path):
                    decayed_path = path + f".decayed.{int(now)}"
                    os.replace(path, decayed_path)
                to_delete.append(bid)
        for bid in to_delete:
            self.ema.pop(bid, None)
            self.counts.pop(bid, None)
            self.meta.pop(bid, None)

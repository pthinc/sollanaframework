"""
Recursive ritual / rehearsal runner for BCE processes.
Applies a ritual_fn recursively with depth/cooldown controls and keeps an audit trace.
"""
from typing import Callable, Dict, Any, List, Optional
import time


class RitualRunner:
    def __init__(self, depth_limit: int = 3, cooldown: float = 0.05):
        self.depth_limit = int(max(1, depth_limit))
        self.cooldown = float(max(0.0, cooldown))
        self.trace: List[Dict[str, Any]] = []
        self._last_ts: float = 0.0

    def run(self,
            seed_state: Dict[str, Any],
            ritual_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
            eval_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
            target: float = 0.0) -> Dict[str, Any]:
        self.trace = []
        final_state = self._step(seed_state, ritual_fn, eval_fn, target, depth=0)
        return {"state": final_state, "trace": self.trace}

    def _step(self,
              state: Dict[str, Any],
              ritual_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
              eval_fn: Optional[Callable[[Dict[str, Any]], float]],
              target: float,
              depth: int) -> Dict[str, Any]:
        now = time.time()
        cooldown_active = self.cooldown > 0 and (now - self._last_ts) < self.cooldown
        metric = eval_fn(state) if callable(eval_fn) else 0.0
        entry = {"depth": depth, "metric": metric, "cooldown": cooldown_active, "state": state}
        self.trace.append(entry)
        if cooldown_active or depth >= self.depth_limit or metric >= target:
            self._last_ts = now
            return state
        action = ritual_fn(state, depth) or {}
        next_state = action.get("state", state)
        stop = bool(action.get("stop", False))
        self._last_ts = now
        if stop:
            return next_state
        return self._step(next_state, ritual_fn, eval_fn, target, depth + 1)


def simple_ritual(state: Dict[str, Any], depth: int) -> Dict[str, Any]:
    # toy ritual: reduce drift and boost resonance a bit each depth
    next_state = dict(state)
    next_state["drift"] = max(0.0, state.get("drift", 1.0) - 0.1)
    next_state["resonance"] = min(1.0, state.get("resonance", 0.5) + 0.05)
    return {"state": next_state, "stop": next_state["drift"] <= 0.1}


def example_usage() -> Dict[str, Any]:
    runner = RitualRunner(depth_limit=4, cooldown=0.0)
    seed = {"drift": 0.6, "resonance": 0.4}
    res = runner.run(seed, simple_ritual, eval_fn=lambda s: 1.0 - s.get("drift", 0.0), target=0.8)
    return res


if __name__ == "__main__":
    print(example_usage())

"""
Brainlobsystem-inspired lightweight lobe message passing and gating.
"""
from typing import Dict, Any, List
import numpy as np

EPS = 1e-12


class Lobe:
    def __init__(self, name: str, gate: float = 1.0):
        self.name = name
        self.gate = gate
        self.state = np.zeros(4, dtype=float)

    def receive(self, msg: np.ndarray):
        self.state = self.state + self.gate * msg
        return self.state

    def emit(self) -> np.ndarray:
        return self.state


def route_message(lobes: Dict[str, "Lobe"], edges: Dict[str, List[str]], source: str, msg: np.ndarray) -> Dict[str, np.ndarray]:
    if source not in lobes:
        return {}
    lobes[source].receive(msg)
    visited = {source}
    queue = [source]
    outputs: Dict[str, np.ndarray] = {}
    while queue:
        cur = queue.pop(0)
        out = lobes[cur].emit()
        outputs[cur] = out
        for nbr in edges.get(cur, []):
            if nbr not in lobes:
                continue
            lobes[nbr].receive(out)
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return outputs

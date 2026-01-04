# character_dynamics.py
import time, math
import numpy as np
from typing import List, Dict

EPS = 1e-12

def integrate_character(samples: List[Dict], now: float = None) -> float:
    """
    samples: list of {"ts": timestamp, "resonance": R, "norm": N, "decay": D}
    returns C(t) approximated by trapezoid integration over samples sorted by ts
    """
    if not samples:
        return 0.0
    arr = sorted(samples, key=lambda s: s["ts"])
    total = 0.0
    for i in range(1, len(arr)):
        dt = arr[i]["ts"] - arr[i-1]["ts"]
        v1 = arr[i-1]["resonance"] * arr[i-1]["norm"] * (1.0 - arr[i-1]["decay"])
        v2 = arr[i]["resonance"] * arr[i]["norm"] * (1.0 - arr[i]["decay"])
        total += 0.5 * (v1 + v2) * max(0.0, dt)
    return float(total)

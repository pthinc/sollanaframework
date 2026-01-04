# selection.py
import numpy as np
from typing import List, Dict, Tuple

def behavior_score(behavior: Dict) -> float:
    """
    behavior must contain keys:
    resonance, char_sal, decay_level, norm_match
    """
    R = float(behavior.get("resonance", 0.0))
    C = float(behavior.get("char_sal", 0.0))
    D = float(behavior.get("decay_level", 0.0))
    N = float(behavior.get("norm_match", 1.0))
    return float(R * C * (1.0 - D) * N)

def select_best(behaviors: List[Dict], top_k: int = 1) -> List[Tuple[Dict, float]]:
    scored = []
    for b in behaviors:
        s = behavior_score(b)
        scored.append((b, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

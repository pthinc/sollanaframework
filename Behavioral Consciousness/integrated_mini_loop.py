# integrated_loop.py
import time
from typing import Any, Dict, List, Optional

import numpy as np

from character_dynamics import integrate_character
from decay_and_norms import dynamic_decay_threshold, norm_diagnostic
from selection import select_best


def run_cycle(behaviors: Optional[List[Dict[str, Any]]], history_samples_by_user: List[Dict[str, Any]], superego_S: float = 0.9) -> Dict[str, Any]:
    """Lightweight integrated loop that ties character, decay, and selection together."""
    behaviors = behaviors or []

    # compute per-user character
    C_val = integrate_character(history_samples_by_user)

    # gradient approximation for C
    grad_C = 0.0
    if len(history_samples_by_user) >= 2:
        last = history_samples_by_user[-1]["resonance"] * history_samples_by_user[-1]["norm"] * (1.0 - history_samples_by_user[-1]["decay"])
        prev = history_samples_by_user[-2]["resonance"] * history_samples_by_user[-2]["norm"] * (1.0 - history_samples_by_user[-2]["decay"])
        dt = history_samples_by_user[-1]["ts"] - history_samples_by_user[-2]["ts"]
        grad_C = (last - prev) / max(1e-3, dt)

    # update dynamic threshold
    N_t = float(np.mean([b.get("norm_match", 1.0) for b in behaviors])) if behaviors else 1.0
    R_t = float(np.mean([b.get("resonance", 0.5) for b in behaviors])) if behaviors else 0.5
    delta_threshold = dynamic_decay_threshold(N_t, abs(grad_C), R_t)

    # diagnose norms
    grad_N = float(np.std([b.get("norm_match", 1.0) for b in behaviors])) if behaviors else 0.0
    D_t = float(np.mean([b.get("decay_level", 0.0) for b in behaviors])) if behaviors else 0.0
    diag = norm_diagnostic(grad_N, D_t, R_t, superego_S)

    # selection
    selected = select_best(behaviors, top_k=1)
    selection_payload = {"behavior_id": selected[0][0].get("behavior_id") if selected else None, "score": selected[0][1] if selected else None}

    report = {
        "C": C_val,
        "grad_C": grad_C,
        "delta_threshold": delta_threshold,
        "norm_diag": diag,
        "selection": selection_payload,
    }
    return report


# Example usage
if __name__ == "__main__":
    now = time.time()
    history = [
        {"ts": now - 3600, "resonance": 0.6, "norm": 0.8, "decay": 0.02},
        {"ts": now - 60, "resonance": 0.7, "norm": 0.85, "decay": 0.01},
    ]
    behaviors = [
        {"behavior_id": "greet_001", "resonance": 0.8, "char_sal": 0.9, "decay_level": 0.02, "norm_match": 0.95},
        {"behavior_id": "help_002", "resonance": 0.6, "char_sal": 0.85, "decay_level": 0.1, "norm_match": 0.9},
    ]
    print(run_cycle(behaviors, history))

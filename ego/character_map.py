# character_map.py
from collections import defaultdict
import math
import time

PHI = (1.0 + 5**0.5) / 2.0

def build_character_map(memory_traces):
    """
    memory_traces: dict of trace_id -> {usage_count, decay_rate, last_used, N0}
    returns: dict behavior_id -> profile
    """
    profile = {}
    for tid, log in memory_traces.items():
        usage = log.get("usage_count", 0)
        decay = log.get("decay_rate", 0.01)
        golden = usage * (1.0 - decay) / PHI
        strength = log.get("N0", 0.0)
        profile[tid] = {"golden_score": golden, "usage_count": usage, "decay_rate": decay, "strength": strength}
    return profile

def rank_character(profile, key="golden_score", top_k=50):
    items = sorted(profile.items(), key=lambda kv: kv[1].get(key, 0.0), reverse=True)
    return items[:top_k]

def update_character_map(char_map, phi_vec, phi_scalar, eta, delta_M, ts=None):
    ts = ts or time.time()
    key = "global"
    char_map.setdefault(key, {"score":0.0, "history":[]})
    contribution = phi_scalar * eta * delta_M
    char_map[key]["score"] += contribution
    char_map[key]["history"].append({"ts":ts, "contrib":contribution})
    return char_map

# bce_peer_adapt.py
import math, numpy as np
from typing import List, Dict

class PeerAgentProfile:
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.history = []  # list of (ts, C, D, R, phi_vec)

    def report_stats(self):
        if not self.history:
            return {"C":0.0, "D":0.0, "R":0.0}
        arr = np.array([[h[1], h[2], h[3]] for h in self.history])
        return {"C":float(arr[:,0].mean()), "D":float(arr[:,1].mean()), "R":float(arr[:,2].mean())}

def peer_resonance(a: PeerAgentProfile, b: PeerAgentProfile) -> float:
    sa = a.report_stats(); sb = b.report_stats()
    # simple resonance metric
    return float((sa["C"] * sb["C"]) * (1.0 - sa["D"]*sb["D"]) * (sa["R"] + sb["R"]) / 2.0)

def adapt_profiles(a: PeerAgentProfile, b: PeerAgentProfile, phi_threshold: float = 0.5):
    r = peer_resonance(a, b)
    if r > phi_threshold:
        # transfer: mix centroid vectors (privacy-preserving: only low-dim stats)
        # here we simulate adaptation by blending mean Cs and Rs
        a_c = a.report_stats()["C"]; b_c = b.report_stats()["C"]
        a_r = a.report_stats()["R"]; b_r = b.report_stats()["R"]
        # small-step adaptation
        a_step = 0.05 * (b_c - a_c)
        b_step = 0.05 * (a_c - b_c)
        # apply as synthetic "nudge" record
        a.history.append((time.time(), a_c + a_step, a.report_stats()["D"], a_r + 0.02*(b_r - a_r), np.zeros(4)))
        b.history.append((time.time(), b_c + b_step, b.report_stats()["D"], b_r + 0.02*(a_r - b_r), np.zeros(4)))
        return True, r
    return False, r

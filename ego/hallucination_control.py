# hallucination_control.py
def anomaly_score(phi, M, ethical_acceptance):
    return float(phi * M * (1.0 - float(ethical_acceptance)))

def remediate_behavior(bid, score, memory, pattern_tracker, detector, human_queue):
    if score > 0.8:
        # quarantine
        memory.trigger_behavior(bid, delta_N=-1.0, decay_rate=1.0)
        human_queue.append({"behavior_id": bid, "reason":"high_anomaly", "score":score})
        detector.remediate(bid, memory)
    elif score > 0.4:
        memory.trigger_behavior(bid, delta_N=-0.5, decay_rate=0.5)

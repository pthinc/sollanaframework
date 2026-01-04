import json
import numpy as np
from system_integrator import SystemIntegrator

def run_scenario(name: str, integ: SystemIntegrator, behavior: dict, context: dict):
    res = integ.process_behavior(behavior, context)
    fields = [
        "bce",
        "trust",
        "decay",
        "activation_curve",
        "path_score",
        "dna_activation",
        "path_mapper",
        "temporal_phi",
        "feature_weighted_phi",
        "shaped_reward",
        "agent_context",
        "ethic_guard",
    ]
    print(f"\n=== {name} ===")
    print(json.dumps({k: res.get(k, None) for k in fields}, indent=2))


integ = SystemIntegrator()

base_behavior = {
    "phi_vec": np.linspace(0.1, 0.8, 8).tolist(),
    "resonance": 0.7,
    "decay_level": 0.08,
    "char_sal": 0.6,
    "decay_rate": 0.05,
    "attention": 0.65,
}
base_context = {
    "user_id": "demo_user",
    "text": "Stabil fakat yaratıcı bir yanıt üret",
    "latency_ms": 180,
    "context_integrity": 0.82,
    "context_integrity_prev": 0.8,
    "emb_prev": np.ones(8).tolist(),
    "emb_curr": (np.ones(8) * 1.02).tolist(),
    "system_series": [0.1, 0.15, 0.2],
    "user_series": [0.08, 0.12, 0.18],
    "norm_alignment": True,
    "allow_auto": True,
    "dt": 1.2,
    "env_reward": 0.3,
    "persist_path": False,
    "enable_vector_processor": False,
}

# Scenario 1: Baseline
run_scenario("baseline", integ, base_behavior, base_context)

# Scenario 2: Yüksek decay & yüksek ödül
beh_high_decay = {**base_behavior, "decay_level": 0.5, "decay_rate": 0.15, "attention": 0.8}
ctx_high_decay = {**base_context, "dt": 2.5, "env_reward": 0.8, "user_id": "demo_user"}
run_scenario("high_decay_high_reward", integ, beh_high_decay, ctx_high_decay)

# Scenario 3: Negatif ödül, etik sınama, patika kalıcılık açık
beh_neg_reward = {**base_behavior, "attention": 0.4, "resonance": 0.4, "decay_level": 0.2}
ctx_neg_reward = {
    **base_context,
    "env_reward": -0.4,
    "dt": 0.6,
    "persist_path": True,
    "user_id": "demo_user",
}
run_scenario("negative_reward_persist_path", integ, beh_neg_reward, ctx_neg_reward)

# Scenario 4: Düşük etik + yüksek drift riski
beh_low_ethics = {**base_behavior, "attention": 0.5, "resonance": 0.5, "decay_level": 0.6, "decay_rate": 0.2}
ctx_low_ethics = {**base_context, "text": "etik ihlali testi", "allow_auto": False, "latency_ms": 320, "dt": 1.8, "env_reward": 0.1, "user_id": "ethics_user"}
run_scenario("low_ethics_high_drift", integ, beh_low_ethics, ctx_low_ethics)

# Scenario 5: Yüksek bayes güven + düşük decay, pozitif ödül
beh_high_trust = {**base_behavior, "attention": 0.9, "resonance": 0.85, "decay_level": 0.02, "decay_rate": 0.02}
ctx_high_trust = {**base_context, "text": "yüksek güven testi", "env_reward": 0.9, "dt": 0.9, "latency_ms": 90, "user_id": "trust_user"}
run_scenario("high_bayes_low_decay", integ, beh_high_trust, ctx_high_trust)

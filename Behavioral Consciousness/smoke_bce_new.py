import os
import sys
import importlib.util
import time
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def load_module(rel_path: str, name: str):
    path = BASE_DIR / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {rel_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bce_core_module = load_module("bce_core_module.py", "bce_core_module")
experience_transformer = load_module("experience_transformer.py", "experience_transformer")
salinim_core = load_module("salinim_core.py", "salinim_core")
master_law = load_module("optimizations/master_law.py", "master_law")
taste_mod = load_module("optimizations/taste_manifold.py", "taste_manifold")
hipofiz_guard = load_module("hipofiz_guard.py", "hipofiz_guard")
brainlobsystem = load_module("optimizations/brainlobsystem.py", "brainlobsystem")
hyperlogic_solver = load_module("optimizations/hyperlogic_solver.py", "hyperlogic_solver")
superneuron_mod = load_module("optimizations/superneuron.py", "superneuron")
kpi_monitor = load_module("optimizations/kpi_monitor.py", "kpi_monitor")
robust_coder_art = load_module("optimizations/robust_coder_art.py", "robust_coder_art")
recursive_rituals = load_module("optimizations/recursive_rituals.py", "recursive_rituals")
system_integrator = load_module("system_integrator.py", "system_integrator")
bayes_math = load_module("optimizations/bayes_math.py", "bayes_math")
google_math = load_module("optimizations/google_math.py", "google_math")
sanal_pons = load_module("optimizations/sanal_pons.py", "sanal_pons")
zincir = load_module("optimizations/zincir.py", "zincir")
ethic_alarm_guard = load_module("optimizations/ethic_alarm_guard.py", "ethic_alarm_guard")
robust_coder_classic = load_module("optimizations/robust_coder_classic.py", "robust_coder_classic")
trust_control = load_module("optimizations/trust_control.py", "trust_control")
trust_control_v2 = load_module("optimizations/trust_control_v2.py", "trust_control_v2")
secret_kesifler = load_module("optimizations/secret_kesifler.py", "secret_kesifler")


def run_smoke():
    engine = bce_core_module.BCEEngine()
    # synthetic behavior and context
    phi_vec = np.random.rand(8)
    behavior = {
        "phi_vec": phi_vec.tolist(),
        "phi": 0.8,
        "history_count": 5,
        "decay_rate": 0.05,
        "meta": {"ethical_tag": "approved"},
        "decay_level": 0.1,
        "resonance": 0.9,
        "char_sal": 0.8
    }
    master_candidates = [
        {"data_density": list(np.random.rand(4)), "delta": [1,1,1,1], "theta": 0.1},
        {"data_density": list(np.random.rand(4)), "delta": [1,0.8,1.2,1], "theta": 0.2},
    ]
    master_ctx = {"bayes_posteriors": [0.6, 0.3, 0.1], "kp": 1.0, "rho": 0.8, "lambda": {"sd":1.0, "ctx":0.7, "tampon":0.5}}
    ctx = {
        "user_type": "bağ_kurucu",
        "context_vec": np.random.rand(8),
        "char_vector": np.random.rand(8),
        "recent_matrix_for_iso": np.random.rand(40,8),
        "clustering_matrix": np.random.rand(60,8),
        "interaction_features": {"engagement": 0.7},
        "allow_auto": True,
        "master_candidates": master_candidates,
        "master_context": master_ctx
    }
    bce_val, comps = engine.compute_BCE(behavior, ctx)
    print("BCE", bce_val, "components", comps.keys())

    # decay policy smoke
    state = {"score": 0.9, "lambda_base": 0.1}
    signals = {"recency": 0.6, "freq": 0.5, "bayes_bias": 0.2, "delta_t": 1.0, "hit_rate": 0.4, "drift_score": 0.1, "human_flag": False, "cluster_avg": 0.5, "eta": 0.1}
    decay_res = salinim_core.apply_decay_policy(state, signals)
    print("Decay policy", decay_res)

    # experience transformer smoke
    et = experience_transformer.ExperienceTransformer(decay_policy_fn=salinim_core.apply_decay_policy)
    x_t = {"context": "test context", "resonance": 0.7, "decay_risk": 0.2, "decay_signals": signals, "lambda_base": 0.1}
    res = et.transform("user1", x_t)
    print(res.get("bce_trace", {}).get("write_status"))

    # taste manifold smoke
    T = taste_mod.taste_vector(0.6,0.4,0.7,0.5,0.3)
    T2 = taste_mod.apply_provoke(T, 0.2)
    sim = taste_mod.taste_similarity(T, T2)
    print("taste sim", sim)

    # hipofiz alarm smoke
    alarms = hipofiz_guard.check_alarms({"latency": 120, "error": 0.01}, {"latency": 200, "error": 0.05})
    print("alarms", alarms)

    # brainlobsystem smoke
    lobes = {"A": brainlobsystem.Lobe("A", gate=1.0), "B": brainlobsystem.Lobe("B", gate=0.8)}
    edges = {"A": ["B"], "B": []}
    outputs = brainlobsystem.route_message(lobes, edges, "A", np.ones(4))
    print("brainlobs outputs", outputs.keys())

    # hyperlogic smoke
    best_seq, score = hyperlogic_solver.backtrack_solve([1,2,3], lambda p: True, lambda p: sum(p))
    print("hyperlogic score", score)

    # superneuron smoke
    sn = superneuron_mod.SuperNeuron(dim=4, v_th=0.5, decay=0.01)
    sn.open_epoch()
    res_sn = sn.step(np.random.rand(4))
    sn.close_epoch()
    print("superneuron fired", res_sn["fired"])

    # KPI monitor smoke
    kpi = kpi_monitor.KPIMonitor(alpha=0.25)
    evt = {"retained": True, "safe": True, "diversity": 0.7, "drift_score": 0.6, "latency_ms": 210, "bce_score": 0.72}
    kpi_snap = kpi.update(evt)
    print("kpi health", round(kpi_snap["health"], 3), kpi_snap["badge"], "p95", round(kpi_snap["p95_latency_ms"], 2))

    # robust coder (ART-style) smoke
    def _toy_pred(z: np.ndarray) -> int:
        return int(np.sum(z) > 1.5)
    vote = robust_coder_art.smooth_vote(_toy_pred, np.random.rand(4))
    print("robust coder top", vote["top_class"], "margin", round(vote["margin"], 3))

    # recursive rituals smoke
    runner = recursive_rituals.RitualRunner(depth_limit=3, cooldown=0.0)
    seed = {"drift": 0.5, "resonance": 0.4}
    ritual_res = runner.run(seed, recursive_rituals.simple_ritual, eval_fn=lambda s: 1.0 - s.get("drift", 0.0), target=0.7)
    print("ritual depth", len(ritual_res["trace"]), "final drift", ritual_res["state"].get("drift"))

    # new helpers smoke
    print("bayes post", bayes_math.bayes_update(0.6, 0.7))
    print("google math", google_math.google_math_rank([("a",0.2),("b",0.8)], [[1,0],[0,1]]))
    print("pons", sanal_pons.bridge(0.7, 0.5))
    print("zincir", zincir.evaluate_chain([lambda: True, lambda: True]))
    print("ethic guard", ethic_alarm_guard.guard({"tox":0.3,"drift":0.2,"latency":0.1}))
    print("robust classic", robust_coder_classic.robust_compare(np.array([0.1,0.2]), np.array([0.1,0.21])))
    print("trust v1", trust_control.evaluate_trust({"consistency":0.7,"honesty":0.8,"drift":0.2,"novelty":0.6}))
    now = time.time()
    tev = [{"ts": now-60, "val": 0.8}, {"ts": now-600, "val": 0.5}]
    print("trust v2", trust_control_v2.evaluate(tev))
    print("secret", secret_kesifler.gated({"hypothesis":"latent cluster"}, sensitivity="high", policy="allow_low"))

    # system integrator smoke
    integ = system_integrator.SystemIntegrator()
    beh = dict(behavior)
    beh["phi_vec"] = phi_vec.tolist()
    si_res = integ.process_behavior(beh, ctx)
    print("integrator keys", list(si_res.keys())[:5], "kpi badge", si_res["kpi"].get("badge"), "ego", si_res.get("ego_balance"))

if __name__ == "__main__":
    run_smoke()

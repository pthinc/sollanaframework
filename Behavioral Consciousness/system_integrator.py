"""
System integrator: wires BCE engine, decay policy, experience transformer, and optimization helpers
(KPI monitor, robust coder, recursive rituals, oksipital reflex/anomaly, superneuron, taste, hyperlogic).
"""
from typing import Dict, Any, List, Optional
import math
import numpy as np
import json
import os
import sys

# Ensure parent directory is on sys.path for cross-module imports (legacy); prefer package import.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bce_core_module import BCEEngine
import salinim_core
from experience_transformer import ExperienceTransformer
from dna.behavioral_dna import BehavioralDNA
from ego.behavior_path_mapper import BehaviorPathMapper
from integrations.agent_comm_system import AgentCommSystem
from integrations.feature_weighting import apply_behavioral_weights
from integrations.rl_reward_shaping import behavior_reward_shaped
from integrations.temporal_activation import TemporalActivation
from integrations.egitim_onayli_vektor_isleyici import VektorIsleyici
from optimizations.taste_manifold import taste_vector, taste_similarity, apply_provoke
from optimizations.hyperlogic_solver import backtrack_solve
from optimizations.superneuron import SuperNeuron
from optimizations.kpi_monitor import KPIMonitor
from optimizations.robust_coder_art import smooth_vote, robust_score
from optimizations.recursive_rituals import RitualRunner, simple_ritual
from optimizations.sanal_oksipital_reflex import SelfRewardReflex, DriftReflex
from optimizations.sanal_oksipital_anomaly import AnomalyCorrector
from optimizations.bayes_math import bayes_update, fuse_evidences
from optimizations.google_math import google_math_rank
from optimizations.sanal_pons import bridge as pons_bridge
from optimizations.zincir import evaluate_chain
from optimizations.ethic_alarm_guard import guard as ethic_guard
from optimizations.robust_coder_classic import robust_compare
from optimizations.trust_control import evaluate_trust
from optimizations.trust_control_v2 import evaluate as evaluate_trust_v2
from optimizations.secret_kesifler import gated as secret_gated
from optimizations.iit_gwt import iit_integrated_information, gwt_broadcast_score
from optimizations.compliance_checks import compliance_scores
from optimizations.path_scoring import activation_curve, path_score
from optimizations.sicimsel_optimizasyon import SicimselOptimizer

EPS = 1e-12


def _safe_prob(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _normalize_score(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


class SystemIntegrator:
    def __init__(self):
        self.engine = BCEEngine()
        self.decay_policy = salinim_core.apply_decay_policy
        self.transformer = ExperienceTransformer(decay_policy_fn=self.decay_policy)
        self.kpi = KPIMonitor(alpha=0.25)
        self.ritual = RitualRunner(depth_limit=3, cooldown=0.0)
        self.superneuron = SuperNeuron(dim=8, v_th=0.6, decay=0.02)
        self.self_reward = SelfRewardReflex()
        self.drift_reflex = DriftReflex()
        self.anomaly = AnomalyCorrector()
        self.sicim_opt = SicimselOptimizer()
        self.dna = BehavioralDNA()
        self.path_mapper = BehaviorPathMapper(persist_dir=os.path.join(os.path.dirname(__file__), "..", "paths"), history_limit=2000)
        self.agent_comm = AgentCommSystem()
        self.temporal_activation = TemporalActivation()
        self.vector_processor = None
        self.vector_model = "bert-base-uncased"
        self.feature_weighter = apply_behavioral_weights
        self.reward_shaper = behavior_reward_shaped
        # psikanalitik sinyaller
        self.ego_balance = 0.5
        self.superego_veto = 0.0
        self.id_drive = 0.5
        self.log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "bce_system.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def process_behavior(self, behavior: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # 1) BCE scoring
        bce_raw = self.engine.compute_BCE(behavior, context)
        if isinstance(bce_raw, tuple) and len(bce_raw) == 2:
            bce_val, comps = bce_raw
        else:
            bce_val = float(bce_raw)
            comps = {}
        output = self.engine.compute_output(bce_val, behavior)

        # 2) Decay policy / reconstruction
        decay_signals = context.get("decay_signals", {})
        decay_state = {"score": max(0.0, 1.0 - behavior.get("decay_level", 0.0)), "lambda_base": behavior.get("lambda_base", 0.1)}
        decay_res = self.decay_policy(decay_state, decay_signals)
        decay_risk = max(0.0, 1.0 - float(decay_res.get("score", 1.0)))

        # 3) Experience transform / trace commit
        x_t = {
            "context": context.get("text", ""),
            "resonance": behavior.get("resonance", 0.0),
            "decay_risk": decay_risk,
            "decay_signals": decay_signals,
            "lambda_base": behavior.get("lambda_base", 0.1),
            "norm_alignment": context.get("norm_alignment", True),
        }
        trace = self.transformer.transform(context.get("user_id", "user"), x_t)

        # 4) Taste manifold comparison
        T = taste_vector(behavior.get("char_sal", 0.5), behavior.get("resonance", 0.5), behavior.get("decay_rate", 0.1), comps.get("fuzzy", 0.5), comps.get("ethics", 0.5))
        T_shift = apply_provoke(T, 0.1)
        taste_sim = taste_similarity(T, T_shift)

        # 5) Superneuron gating
        self.superneuron.open_epoch()
        sn_res = self.superneuron.step(np.asarray(behavior.get("phi_vec", np.zeros(8)), dtype=float))
        self.superneuron.close_epoch()

        # 6) Hyperlogic quick solve
        _, hyper_score = backtrack_solve([1, 2, 3], lambda p: True, lambda p: sum(p))

        # 7) Robust coder margins
        vote = smooth_vote(lambda z: int(np.sum(z) > 1.5), np.asarray(behavior.get("phi_vec", np.zeros(4)), dtype=float))
        robust = robust_score(lambda z: float(np.tanh(np.sum(z))), np.asarray(behavior.get("phi_vec", np.zeros(4)), dtype=float))
        robust_classic = robust_compare(np.asarray(behavior.get("phi_vec", np.zeros(4)), dtype=float), np.asarray(context.get("phi_vec_cmp", np.zeros(4)), dtype=float))

        # trust controls
        trust = evaluate_trust({
            "consistency": comps.get("dcycle", 0.5),
            "honesty": comps.get("ethics", 0.5),
            "drift": decay_risk,
            "novelty": comps.get("bayes", 0.5),
        })
        trust_events = context.get("trust_events", [])
        trust_v2 = evaluate_trust_v2(trust_events)

        # 8) KPI monitor update
        kpi_evt = self.kpi.update({
            "retained": context.get("retained", True),
            "safe": context.get("safe", True),
            "diversity": comps.get("dbscan", 0.5),
            "drift_score": 1.0 - decay_risk,
            "latency_ms": context.get("latency_ms", 200.0),
            "bce_score": _safe_prob(bce_val),
        })

        # 9) Ritual runner if drift risk high
        ritual_res = None
        if decay_risk > 0.4:
            seed = {"drift": decay_risk, "resonance": behavior.get("resonance", 0.5)}
            ritual_res = self.ritual.run(seed, simple_ritual, eval_fn=lambda s: 1.0 - s.get("drift", 0.0), target=0.7)

        # 10) Oksipital reflexes
        token_counts = context.get("token_counts", {"default": 1})
        self_reward_evt = self.self_reward.step(
            user_id=context.get("user_id", "user"),
            context_integrity=context.get("context_integrity", 0.8),
            creative_interaction=context.get("creative_interaction", 0.6),
            data_accuracy=context.get("data_accuracy", 0.9),
            token_counts=token_counts,
            D_obs=behavior.get("decay_level", 0.1),
        )
        drift_evt = self.drift_reflex.step(
            user_id=context.get("user_id", "user"),
            E_prev=np.asarray(context.get("emb_prev", np.ones(8)), dtype=float),
            E_curr=np.asarray(context.get("emb_curr", np.ones(8)), dtype=float),
            dt=float(context.get("dt", 1.0)),
        )
        try:
            from backends.trainer_bridge import record_telemetry as _rec_telemetry
        except Exception:
            try:
                from sollana.backends.trainer_bridge import record_telemetry as _rec_telemetry
            except Exception:
                _rec_telemetry = None
        if callable(_rec_telemetry):
            if isinstance(self_reward_evt, dict):
                _rec_telemetry(self_reward_evt)
            if isinstance(drift_evt, dict):
                _rec_telemetry(drift_evt)

        # 11) Anomaly correction
        system_series = np.asarray(context.get("system_series", [0.0, 0.1, 0.2]), dtype=float)
        user_series = np.asarray(context.get("user_series", [0.0, 0.05, 0.1]), dtype=float)
        expected_flavor = context.get("expected_flavor", {"gentle": 0.6})
        incoming_flavor = context.get("incoming_flavor", {"gentle": 0.6})
        anomaly_eval = self.anomaly.evaluate(
            user_id=context.get("user_id", "user"),
            system_series=system_series,
            user_series=user_series,
            expected_flavor=expected_flavor,
            incoming_flavor=incoming_flavor,
            payload_text=context.get("text", ""),
            request_count_window=int(context.get("request_count_window", 1)),
            tone_valence_abs=float(context.get("tone_valence_abs", 0.1)),
            topic_shift_score=float(context.get("topic_shift_score", 0.1)),
            context_integrity_now=float(context.get("context_integrity", 0.8)),
            context_integrity_prev=float(context.get("context_integrity_prev", 0.8)),
            decay_obs=float(behavior.get("decay_level", 0.1)),
        )

        # 12) Bayes + GoogleMath style ranking
        bayes_post = bayes_update(0.6, 0.7)
        fused_probs = fuse_evidences([0.5, 0.5], [0.7, 0.6])
        gm_rank = google_math_rank([("a", 0.2), ("b", 0.8)], [[1, 0], [0, 1]])

        # 13) Pons bridge and Zincir chain
        pons_res = pons_bridge(comps.get("bayes", 0.5), comps.get("ethics", 0.5))
        chain_res = evaluate_chain([lambda: True, lambda: comps.get("ethics", 0.5) > 0.2])

        # 14) Ethic alarm guard
        ethic_res = ethic_guard({"tox": 1.0 - comps.get("ethics", 0.5), "drift": decay_risk, "latency": context.get("latency_ms", 0.2) / 1000.0})

        # 15) Secret discoveries gating
        secret_res = secret_gated({"hint": context.get("secret_hint", "")}, sensitivity=context.get("secret_sensitivity", "medium"), policy=context.get("secret_policy", "deny_by_default"))

        # 16) Psikanalitik metrikler
        ego_balance = float(0.5 * comps.get("dcycle", 0.5) + 0.5 * comps.get("ethics", 0.5))
        superego_veto = float(max(0.0, 1.0 - comps.get("ethics", 0.5)))
        id_drive = float(min(1.0, behavior.get("resonance", 0.5) + decay_risk))
        self.ego_balance = ego_balance
        self.superego_veto = superego_veto
        self.id_drive = id_drive

        # 17) Adler + Freud göstergeleri
        adler_social_interest = float(_safe_prob(trust.get("score", 0.5)))
        adler_inferiority_tension = float(_safe_prob(decay_risk))
        freud_conflict = float(max(0.0, superego_veto - id_drive))
        freud_drive_alignment = float(_safe_prob(1.0 - freud_conflict))

        # 18) IIT / GWT metrikleri
        phi_matrix = np.asarray([
            comps.get("bayes", 0.5),
            comps.get("ethics", 0.5),
            comps.get("dcycle", 0.5),
            comps.get("dbscan", 0.5),
        ], dtype=float)
        iit_res = iit_integrated_information(phi_matrix)
        gwt_res = gwt_broadcast_score(np.asarray(behavior.get("phi_vec", np.zeros(8)), dtype=float))

        # 19) Compliance heuristics (ISO family)
        compliance = compliance_scores(context)

        # 20) Temporal activation and path score (doc-aligned)
        activation_val = activation_curve(float(context.get("t", context.get("dt", 1.0))))
        trace_len = 1
        try:
            trace_len = max(1, len(trace))
        except Exception:
            trace_len = 1
        path_score_val = path_score(_safe_prob(bce_val), decay_risk, trust.get("score", 0.5), trace_len)

        # 21) Behavioral DNA activation
        dna_activation = None
        try:
            A = float(behavior.get("attention", behavior.get("resonance", 0.5)))
            P = float(max(EPS, comps.get("bayes", 0.5)))
            W = float(trust.get("score", 0.5))
            t_val = float(context.get("t", context.get("dt", 1.0)))
            dna_raw = self.dna(A, P, W, t_val) if callable(self.dna) else None
            if dna_raw is None:
                dna_activation = None
            elif hasattr(dna_raw, "detach"):
                dna_activation = float(dna_raw.detach().cpu().item())
            elif hasattr(dna_raw, "numpy"):
                dna_activation = float(np.asarray(dna_raw).item())
            else:
                dna_activation = float(dna_raw)
        except Exception:
            dna_activation = None

        # 22) Path mapper trace
        path_mapper_step = None
        path_mapper_cum = None
        path_mapper_export = None
        try:
            params = {
                "attention": float(behavior.get("attention", behavior.get("resonance", 0.5))),
                "match_prob": float(comps.get("bayes", 0.5)),
                "context_weight": float(comps.get("dcycle", 0.5)),
                "activation": activation_val,
                "ethical": comps.get("ethics", 0.5),
                "anomaly_penalty": float(anomaly_eval.get("anomaly_score", 0.0)) if isinstance(anomaly_eval, dict) else 0.0,
                "decay_rate": float(behavior.get("decay_rate", 0.05)),
            }
            ts_val = context.get("t", None) or context.get("dt", None)
            ts_num = float(ts_val) if ts_val is not None else None
            path_mapper_step = self.path_mapper.record_step(context.get("user_id", "user"), module="system_integrator", params=params, ts=ts_num)
            path_mapper_cum = self.path_mapper.cumulative_phi(context.get("user_id", "user"))
            path_mapper_export = self.path_mapper.export_path(context.get("user_id", "user"))
            if context.get("persist_path", False):
                self.path_mapper.persist_path(context.get("user_id", "user"))
        except Exception:
            path_mapper_step = None

        # 23) Temporal activation on phi vector
        temporal_phi = None
        try:
            t_val_scalar = float(context.get("t", context.get("dt", 1.0)))
            phi_vec_arr = np.asarray(behavior.get("phi_vec", np.zeros(8)), dtype=float)
            if self.temporal_activation is not None:
                try:
                    import torch
                    if hasattr(self.temporal_activation, "forward"):
                        hidden = torch.tensor(phi_vec_arr, dtype=torch.float32).unsqueeze(0)
                        t_t = torch.tensor([t_val_scalar], dtype=torch.float32)
                        ta = self.temporal_activation(hidden, t_t)
                        ta_detach = getattr(ta, "detach", None)
                        if callable(ta_detach):
                            ta_arr = np.asarray(ta_detach())
                        else:
                            ta_arr = np.asarray(ta)
                        temporal_phi = np.asarray(ta_arr, dtype=float).flatten().tolist()
                    else:
                        ta_arr = self.temporal_activation(phi_vec_arr, t_val_scalar)
                        temporal_phi = np.asarray(ta_arr, dtype=float).flatten().tolist()
                except Exception:
                    temporal_phi = (phi_vec_arr * activation_val).tolist()
        except Exception:
            temporal_phi = None

        # 24) Feature weighting utility
        feature_weighted = None
        try:
            phi_vec_arr = np.asarray(behavior.get("phi_vec", np.zeros(8)), dtype=float)
            ctx_w = context.get("context_weights")
            if ctx_w is None:
                ctx_w_arr = np.ones_like(phi_vec_arr)
            else:
                ctx_w_arr = np.asarray(ctx_w, dtype=float)
                if ctx_w_arr.shape[0] != phi_vec_arr.shape[0]:
                    ctx_w_arr = np.resize(ctx_w_arr, phi_vec_arr.shape[0])
            feature_weighted = self.feature_weighter(phi_vec_arr.reshape(1, -1), ctx_w_arr).flatten().tolist()
        except Exception:
            feature_weighted = None

        # 25) RL reward shaping using DNA coder
        shaped_reward = None
        try:
            env_reward = float(context.get("env_reward", 0.0))
            A = float(behavior.get("attention", behavior.get("resonance", 0.5)))
            P = float(max(EPS, comps.get("bayes", 0.5)))
            W = float(trust.get("score", 0.5))
            t_val_scalar = float(context.get("t", context.get("dt", 1.0)))
            shaped_reward = self.reward_shaper(env_reward, A, P, W, t_val_scalar, self.dna, alpha=context.get("reward_alpha", 1.0))
        except Exception:
            shaped_reward = None

        # 25b) Sicimsel optimizasyon metrikleri
        sicimsel_opt = None
        try:
            sicimsel_opt = self.sicim_opt.evaluate(
                behavior.get("phi_vec", np.zeros(8)),
                dt=float(context.get("dt", 1.0)),
                residual_overdrive=float(context.get("residual_overdrive", 1.0)),
            )
        except Exception:
            sicimsel_opt = None

        # 26) Vector processor (lazy, gated)
        vector_activation_summary = None
        try:
            if context.get("enable_vector_processor", False):
                if self.vector_processor is None:
                    self.vector_processor = VektorIsleyici(model_name=context.get("vector_model", self.vector_model))
                    self.vector_model = context.get("vector_model", self.vector_model)
                phi_vec_np = np.asarray(behavior.get("phi_vec", np.zeros(8)), dtype=float)
                emb_np = np.asarray(context.get("embeddings", np.zeros(16)), dtype=float)
                text_input = context.get("vector_text") or context.get("text", "") or "behavioral context"
                vres = self.vector_processor.girdi_ve_aktivasyon(phi_vec_np, emb_np, text_input)
                vector_activation_summary = {
                    "combined_shape": list(getattr(vres.get("combined_input"), "shape", [])),
                    "hidden_layers": len(vres.get("activations", {}).get("hidden_states", [])),
                    "attention_layers": len(vres.get("activations", {}).get("attentions", [])),
                    "model": self.vector_model,
                }
        except Exception:
            vector_activation_summary = None

        # 27) Agent communication context bridge
        agent_context_snapshot = None
        try:
            if self.agent_comm is not None:
                self.agent_comm.update_context({
                    "bce": bce_val,
                    "trust": trust.get("score", 0.5),
                    "decay_risk": decay_risk,
                    "activation": activation_val,
                })
                agent_context_snapshot = self.agent_comm.get_context()
        except Exception:
            agent_context_snapshot = None


        result = {
            "bce": bce_val,
            "components": comps,
            "output": output,
            "decay": decay_res,
            "trace": trace,
            "taste_sim": taste_sim,
            "superneuron": sn_res,
            "hyperlogic_score": hyper_score,
            "robust_vote": vote,
            "robust_score": robust,
            "robust_classic": robust_classic,
            "trust": trust,
            "trust_v2": trust_v2,
            "kpi": kpi_evt,
            "ritual": ritual_res,
            "self_reward": self_reward_evt,
            "drift_reflex": drift_evt,
            "anomaly": anomaly_eval,
            "bayes_post": bayes_post,
            "fused_probs": fused_probs,
            "google_math_rank": gm_rank,
            "pons": pons_res,
            "chain": chain_res,
            "ethic_guard": ethic_res,
            "secret": secret_res,
            "ego_balance": ego_balance,
            "superego_veto": superego_veto,
            "id_drive": id_drive,
            "adler": {
                "social_interest": adler_social_interest,
                "inferiority_tension": adler_inferiority_tension,
            },
            "freud": {
                "conflict": freud_conflict,
                "drive_alignment": freud_drive_alignment,
            },
            "iit": iit_res,
            "gwt": gwt_res,
            "compliance": compliance,
            "activation_curve": activation_val,
            "path_score": path_score_val,
            "dna_activation": dna_activation,
            "path_mapper": {
                "step": {
                    "ts": getattr(path_mapper_step, "ts", None),
                    "phi": getattr(path_mapper_step, "phi", None),
                    "module": getattr(path_mapper_step, "module", None),
                    "params": getattr(path_mapper_step, "params", None),
                } if path_mapper_step else None,
                "cumulative_phi": path_mapper_cum,
                "export": path_mapper_export,
            },
            "temporal_phi": temporal_phi,
            "feature_weighted_phi": feature_weighted,
            "shaped_reward": shaped_reward,
            "sicimsel_opt": sicimsel_opt,
            "vector_activation": vector_activation_summary,
            "agent_context": agent_context_snapshot,
        }
        try:
            payload = {
                "behavior_id": context.get("user_id", "user"),
                "bce": bce_val,
                "trust": trust,
                "trust_v2": trust_v2,
                "adler": result.get("adler", {}),
                "freud": result.get("freud", {}),
                "iit": iit_res,
                "gwt": gwt_res,
                "compliance": compliance,
                "ego_balance": ego_balance,
                "superego_veto": superego_veto,
                "id_drive": id_drive,
                "kpi": kpi_evt,
                "secret": secret_res,
                "ethic_guard": ethic_res,
                "activation_curve": activation_val,
                "path_score": path_score_val,
                "dna_activation": dna_activation,
                "path_mapper_cumulative": path_mapper_cum,
                "temporal_phi": temporal_phi,
                "feature_weighted_phi": feature_weighted,
                "shaped_reward": shaped_reward,
                "sicimsel_opt": sicimsel_opt,
                "vector_activation": vector_activation_summary,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

        return result

    def run_cycle(self, behaviors: List[Dict[str, Any]], history_samples_by_user: List[Dict[str, Any]], superego_S: float = 0.9) -> Dict[str, Any]:
        from integrated_mini_loop import run_cycle
        return run_cycle(behaviors, history_samples_by_user, superego_S=superego_S)


def example_usage() -> Dict[str, Any]:
    integ = SystemIntegrator()
    behavior = {
        "phi_vec": np.random.rand(8).tolist(),
        "phi": 0.8,
        "history_count": 4,
        "decay_rate": 0.05,
        "meta": {"ethical_tag": "approved"},
        "decay_level": 0.1,
        "resonance": 0.7,
        "char_sal": 0.8,
        "lambda_base": 0.1,
    }
    ctx = {
        "user_id": "demo_user",
        "user_type": "bağ_kurucu",
        "context_vec": np.random.rand(8),
        "char_vector": np.random.rand(8),
        "recent_matrix_for_iso": np.random.rand(20, 8),
        "clustering_matrix": np.random.rand(30, 8),
        "interaction_features": {"engagement": 0.6},
        "allow_auto": True,
        "text": "Demo payload",
    }
    return integ.process_behavior(behavior, ctx)


if __name__ == "__main__":
    print(example_usage())

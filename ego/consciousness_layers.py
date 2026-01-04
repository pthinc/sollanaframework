# consciousness_layers.py
"""Consciousness layers with backend-aware tensor handling (torch preferred)."""

import time
from typing import Any, Dict, Optional
from backends import ensure_backend

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

# Ön kabul: aşağıdaki modüller proje içinde mevcut
# from behavioral_dna import BehavioralDNA
# from temporal_memory import TemporalMemory
# from behavior_anomaly import BehaviorAnomalyDetector

class Behavior:
    def __init__(self, id: str, input_signal: Any, metadata: Dict[str, Any]):
        self.id = id
        self.input = input_signal
        self.metadata = metadata
        self.score = 0.0
        self.activation = 0.0
        self.trace_id: Optional[str] = None
        self.context_vec = metadata.get("context_vec")

class ConsciousnessLayer:
    def __init__(self, name: str):
        self.name = name

    def process(self, behavior: Behavior) -> Optional[Behavior]:
        raise NotImplementedError

class IdLayer(ConsciousnessLayer):
    def __init__(self, dna_model, mutator, max_variation: float = 0.05):
        super().__init__("id")
        self.dna = dna_model
        self.mutator = mutator
        self.max_variation = max_variation

    def process(self, behavior: Behavior, backend_name: Optional[str] = None) -> Behavior:
        backend = ensure_backend(backend_name)

        A_val = behavior.metadata.get("A", 0.5)
        P_val = behavior.metadata.get("P", 0.5)
        W_val = behavior.metadata.get("W", 0.0)
        t_val = behavior.metadata.get("t", 0.0)

        if backend.name.startswith("torch") and _TORCH_AVAILABLE:
            A = torch.tensor(A_val)
            P = torch.tensor(P_val)
            W = torch.tensor(W_val)
            t = torch.tensor(t_val)
            D = float(self.dna(A, P, W, t))
            behavior.activation = float(torch.tanh(torch.exp(t) - 3.141592653589793))
            variation = self.mutator.variation_range * (2.0 * (torch.rand(1).item() - 0.5))
        else:
            import numpy as np
            import math

            D = float(self.dna(np.array(A_val), np.array(P_val), np.array(W_val), np.array(t_val)))
            behavior.activation = float(np.tanh(np.exp(t_val) - math.pi))
            variation = self.mutator.variation_range * (2.0 * (np.random.rand() - 0.5))

        behavior.score = D + variation
        return behavior

class EgoLayer(ConsciousnessLayer):
    def __init__(self, temporal_memory, context_threshold: float = 0.6):
        super().__init__("ego")
        self.memory = temporal_memory
        self.context_threshold = context_threshold

    def process(self, behavior: Behavior) -> Optional[Behavior]:
        activation_vec = self.memory.activation_vector([behavior.id]) if hasattr(self.memory, "activation_vector") else None
        act = float(activation_vec[0].item()) if activation_vec is not None else behavior.activation
        behavior.activation = act
        context_match = behavior.metadata.get("context_match", 0.5)
        # bağlam eşiği kontrolü
        if context_match < self.context_threshold and act < 0.2:
            return None
        # temporal memory'e commit veya update
        if hasattr(self.memory, "trigger_behavior"):
            behavior.trace_id = f"trace_{int(time.time()*1000)}"
            self.memory.trigger_behavior(behavior.trace_id, context=behavior.metadata.get("context", "unknown"),
                                         delta_N=behavior.score, decay_rate=behavior.metadata.get("decay_rate", 0.01))
        return behavior

class SuperegoLayer(ConsciousnessLayer):
    def __init__(self, anomaly_detector, human_approval_required: bool = False, ethical_threshold: float = 0.5):
        super().__init__("superego")
        self.detector = anomaly_detector
        self.human_approval_required = human_approval_required
        self.ethical_threshold = ethical_threshold

    def process(self, behavior: Behavior) -> Optional[Behavior]:
        # anomali değerlendirmesi
        if hasattr(self.detector, "assess_behavior"):
            res = self.detector.assess_behavior(behavior.id, verifier=None)
            if res.get("anomaly_score", 0.0) > 0.6:
                self.detector.remediate(behavior.id, memory=self.detector)  # memory injection beklenir dışarıdan
                return None
        ethical_tag = behavior.metadata.get("ethical_tag", 1.0)
        if isinstance(ethical_tag, (int, float)):
            if ethical_tag < self.ethical_threshold:
                return None
        elif ethical_tag != "approved":
            return None
        if self.human_approval_required and behavior.metadata.get("human_ok", False) is not True:
            return None
        return behavior

class ConsciousnessController:
    def __init__(self, id_layer: IdLayer, ego_layer: EgoLayer, superego_layer: SuperegoLayer):
        self.id = id_layer
        self.ego = ego_layer
        self.superego = superego_layer

    def run(self, behavior: Behavior, backend_name: Optional[str] = None) -> Optional[Behavior]:
        b = self.id.process(behavior, backend_name=backend_name)
        if b is None:
            return None
        b = self.ego.process(b)
        if b is None:
            return None
        b = self.superego.process(b)
        return b

# example_pipeline.py

import time

from behavior_path_mapper import BehaviorPathMapper
from behavioral_dna import BehavioralDNA
from temporal_memory import TemporalMemory
from behavior_anomaly import BehaviorAnomalyDetector
from behavior_mutation import BehaviorMutator
from consciousness_layers import Behavior, IdLayer, EgoLayer, SuperegoLayer, ConsciousnessController

# sahte nesneler temporal memory ve pattern tracker yerine
class DummyMemory:
    def trigger_behavior(self, *a, **k): pass

class DummyPattern:
    def record(self, *a, **k): pass

memory = DummyMemory()
pattern = DummyPattern()

mapper = BehaviorPathMapper(memory=memory, pattern_tracker=pattern)

# Transformer inference stage 1 encode
bce = {
    "behavior_id": "greet_001",
    "attention": 0.82,
    "match_prob": 0.67,
    "context_weight": 0.91,
    "ethical": "approved",
    "timestamp": time.time()
}
# tokenization module
step1 = mapper.record_step("greet_001", module="tokenizer", params={"attention": bce["attention"], "match_prob": bce["match_prob"], "context_weight": bce["context_weight"], "activation": 0.0}, ts=bce["timestamp"])

# transformer attention bias injection
step2 = mapper.record_step("greet_001", module="transformer_attention", params={"attention": 0.82, "context_weight": 0.91, "activation": 0.1}, ts=time.time())

# response generation and ethical check
step3 = mapper.record_step("greet_001", module="ethics_check", params={"ethical": "approved", "anomaly_penalty": 0.0}, ts=time.time())

# persist path and read cumulative score
path = mapper.persist_path("greet_001")
cum = mapper.cumulative_phi("greet_001")
print("persisted to", path, "cumulative phi", cum)

# modüller
dna = BehavioralDNA(h=0.7, k=0.5, F=1.2, learnable=False)
mem = TemporalMemory(prune_threshold=1e-5)
anom = BehaviorAnomalyDetector()
mut = BehaviorMutator(variation_range=0.02, seed=42)

# katmanlar ve controller
id_layer = IdLayer(dna_model=dna, mutator=mut)
ego_layer = EgoLayer(temporal_memory=mem, context_threshold=0.6)
superego = SuperegoLayer(anomaly_detector=anom, human_approval_required=False)
controller = ConsciousnessController(id_layer, ego_layer, superego)

# davranış örneği
beh = Behavior(id="greeting_v1", input_signal="hey", metadata={
    "A": 0.8, "P": 0.9, "W": 0.4, "t": 0.1, "context_match": 0.9, "ethical_tag": 1.0
})
result = controller.run(beh)
print("result", result and result.score)

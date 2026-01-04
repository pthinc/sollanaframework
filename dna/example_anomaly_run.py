# example_anomaly_run.py
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from behavior_anomaly import BehaviorAnomalyDetector, BehaviorEvent

if not TORCH_AVAILABLE:
    raise SystemExit("This example requires PyTorch. Install torch to run.")

detector = BehaviorAnomalyDetector()
# callback örneği
def alert_cb(payload):
    print("ALERT", payload)
detector.alert_callback = alert_cb

# uygun context embedding üretimi modeller hattından gelir
ctx1 = torch.randn(128)
ctx2 = ctx1 * 0.95
ctx3 = torch.randn(128)

# log olayları
detector.log_event(BehaviorEvent("greet_v1", score=0.9, context_vec=ctx1))
detector.log_event(BehaviorEvent("greet_v1", score=0.92, context_vec=ctx2))
detector.log_event(BehaviorEvent("greet_v1", score=0.91, context_vec=ctx3))

# verifier örneği basit doğruluk simülasyonu
def simple_verifier(ev):
    return 0.2
res = detector.assess_behavior("greet_v1", verifier=simple_verifier)
print("anomaly result", res)
# remediate örneği memory entegrasyonu ile
# detector.remediate("greet_v1", memory_instance)

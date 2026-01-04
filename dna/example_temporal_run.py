# example_temporal_run.py
import time

try:
	import torch
	TORCH_AVAILABLE = True
except Exception:
	TORCH_AVAILABLE = False

from temporal_memory import TemporalMemory, TemporalMemoryModule

if not TORCH_AVAILABLE:
	raise SystemExit("This example requires PyTorch. Install torch to run.")

mem = TemporalMemory(prune_threshold=1e-5)
mem.trigger_behavior('greeting', context='greeting', delta_N=1.0, decay_rate=0.05)
mem.trigger_behavior('warning', context='safety', delta_N=0.5, decay_rate=0.3)

module = TemporalMemoryModule(mem, scale=0.4)
hidden = torch.randn(1, 2)  # iki iz ile eşleşen gizli vektör
out = module(hidden, ['greeting', 'warning'])
print("out:", out)

# Simüle zaman akışı
time.sleep(1.0)
mem.trigger_behavior('greeting', context='greeting', delta_N=0.2)  # tekrar kullanıldı
mem.sweep_prune()
print(mem.list_traces())

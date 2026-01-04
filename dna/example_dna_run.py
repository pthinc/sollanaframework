# example_dna_run.py
try:
	import torch
	import torch.nn.functional as F
	TORCH_AVAILABLE = True
except Exception:
	TORCH_AVAILABLE = False

from behavioral_dna import BehavioralDNA

# Örnek veri
if not TORCH_AVAILABLE:
	raise SystemExit("This example requires PyTorch. Install torch to run.")

A = torch.tensor([0.8, 0.3, 0.1])        # attention skorları
P = torch.tensor([0.9, 0.2, 0.01])       # bağlam olasılıkları
W = torch.tensor([0.5, 0.0, 1.2])        # bağlam ağırlıkları
t = torch.tensor([0.0, 1.0, 2.0])        # zaman farkları veya zaman indexleri

model = BehavioralDNA(h=0.7, k=0.5, F=1.2, learnable=True)
D = model(A, P, W, t)                    # D shape: (3,)
print("D:", D)

# Basit eğitim adımı örneği
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.tensor([0.6, 0.1, 0.05])
loss = F.mse_loss(D, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

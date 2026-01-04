"""Physical behavior coder example with torch guard."""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from physical_behavior_coder import PhysicalBehaviorCoder

if not TORCH_AVAILABLE:
    raise SystemExit("This example requires PyTorch. Install torch to run.")

# Örnek input
A = torch.tensor([0.8, 0.3])
P = torch.tensor([0.9, 0.01])
W = torch.tensor([0.5, 1.2])
t = torch.tensor([0.0, 1.0])

coder = PhysicalBehaviorCoder(learnable_scale=True)
D = coder(A, P, W, t)          # D shape: (2,)
print("D:", D)

# Basit eğitim döngüsü: hedef D_target elde etmeye çalış
D_target = torch.tensor([0.6, 0.05])
opt = torch.optim.Adam(coder.parameters(), lr=1e-3)
for step in range(200):
    opt.zero_grad()
    D_out = coder(A, P, W, t)
    loss = F.mse_loss(D_out, D_target)
    loss.backward()
    opt.step()

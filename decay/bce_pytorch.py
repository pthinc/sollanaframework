# bce_pytorch.py
from typing import Optional
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

from decay_core import BehavioralMemory

if _TORCH_AVAILABLE:

    class BehavioralDecayModule(nn.Module):
        def __init__(self, mem: BehavioralMemory, device: Optional[torch.device] = None, scale: float = 1.0):
            super().__init__()
            self.mem = mem
            self.device = device or torch.device("cpu")
            self.scale = scale

        def forward(self, hidden: torch.Tensor, trace_ids: list):
            now = time.time()
            strengths = []
            for tid in trace_ids:
                strengths.append(self.mem.get_strength(tid, now))
            if len(strengths) == 0:
                return hidden
            s = torch.tensor(strengths, dtype=hidden.dtype, device=self.device)
            s = s.view(1, -1)
            contribution = (s @ hidden) / (s.sum() + 1e-9)
            out = hidden * (1 - self.scale) + contribution * self.scale
            return out

else:

    class BehavioralDecayModule:
        """Placeholder when torch is not installed. Instantiation raises an error."""

        def __init__(self, *_, **__):
            raise RuntimeError(
                "BehavioralDecayModule requires PyTorch. Install torch or select a different backend."
            )

def decay_regularization_loss(mem: BehavioralMemory, target_sparsity: float = 0.1):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("decay_regularization_loss requires PyTorch. Install torch or skip this regularizer.")
    now = time.time()
    strengths = torch.tensor([tr.strength(now) for tr in mem.traces.values()], dtype=torch.float32)
    if strengths.numel() == 0:
        return torch.tensor(0.0)
    total = strengths.sum()
    # hedef daha az toplam iz gücü ise cezalandır
    loss = F.relu(total - target_sparsity)
    return loss

def lambda_from_half_life(T_half: float) -> float:
    return 0.693 / max(T_half, 1e-9)

class LearnableLambda(nn.Module):
    def __init__(self, init_half_life: float):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(lambda_from_half_life(init_half_life))))
    def forward(self):
        return torch.exp(self.log_lambda)


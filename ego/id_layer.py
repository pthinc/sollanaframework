# id_layer.py
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def x_t(t: torch.Tensor, clamp_max: float = 20.0) -> torch.Tensor:
    t_clamped = torch.clamp(t, max=clamp_max)
    return torch.tanh(torch.exp(t_clamped) - math.pi)

class IdNeuronLayer(nn.Module):
    """
    Vektörize İd katmanı.
    input: (batch, input_dim)
    returns: behaviors (batch, num_behaviors), activations (batch, num_behaviors)
    """
    def __init__(self, input_dim: int, num_behaviors: int,
                 init_h: float = 1.0, init_k: float = 1.0, init_F: float = 1.0,
                 variation_scale: float = 0.02, eps: float = 1e-9):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_behaviors, bias=True)
        self.variation_scale = variation_scale
        self.eps = eps
        self.log_h = nn.Parameter(torch.log(torch.tensor(float(init_h))))
        self.log_k = nn.Parameter(torch.log(torch.tensor(float(init_k))))
        self.log_F = nn.Parameter(torch.log(torch.tensor(float(init_F))))

    def effective_h(self) -> torch.Tensor:
        return torch.exp(self.log_h)

    def effective_k(self) -> torch.Tensor:
        return torch.exp(self.log_k)

    def effective_F(self) -> torch.Tensor:
        return torch.exp(self.log_F)

    def forward(self,
                x: torch.Tensor,
                attention: torch.Tensor,
                match_prob: torch.Tensor,
                context_weight: torch.Tensor,
                timestamp: torch.Tensor,
                seed_noise: Optional[torch.Generator] = None):
        base = torch.sigmoid(self.linear(x))                             # (batch, num_behaviors)
        t = timestamp if timestamp.ndim > 1 else timestamp.view(-1, 1)
        xt = x_t(t)                                                      # broadcastable
        A = attention if attention.shape == base.shape else attention.view(-1,1).expand_as(base)
        P = torch.clamp(match_prob, min=self.eps, max=1.0)
        P = P if P.shape == base.shape else P.view(-1,1).expand_as(base)
        W = context_weight if context_weight.shape == base.shape else context_weight.view(-1,1).expand_as(base)
        h = self.effective_h()
        k = self.effective_k()
        Fv = self.effective_F()
        D = xt * (h * A + k * torch.log(P) + Fv * W)
        if self.training and self.variation_scale > 0.0:
            noise = torch.randn_like(D, generator=seed_noise) * self.variation_scale
            D = D + noise
        behaviors = base * D
        activations = xt.expand_as(base)
        return behaviors, activations

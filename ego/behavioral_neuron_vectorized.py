# behavioral_neuron_vectorized.py
"""Vectorized behavioral neuron layer (torch-first with guard)."""

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

def x_t(t, clamp_max: float = 20.0):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("BehavioralNeuronLayer requires torch; install torch or use another backend.")
    t_clamped = torch.clamp(t, max=clamp_max)
    return torch.tanh(torch.exp(t_clamped) - math.pi)

class BehavioralNeuronLayer(nn.Module):
    """
    Çok nöronlu, vektörize davranışsal katman.
    Girdi shape: (batch, input_dim)
    Çıktı shape: (batch, num_neurons)
    attention, match_prob, context_weight, timestamp shapes broadcastable olmalıdır.
    """
    def __init__(self, input_dim: int, num_neurons: int,
                 init_h: float = 1.0, init_k: float = 1.0, init_F: float = 1.0,
                 learnable_constants: bool = True, eps: float = 1e-9):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("BehavioralNeuronLayer requires torch; install torch to use this layer.")
        super().__init__()
        self.linear = nn.Linear(input_dim, num_neurons)
        self.eps = eps
        if learnable_constants:
            self.log_h = nn.Parameter(torch.log(torch.tensor(float(init_h))))
            self.log_k = nn.Parameter(torch.log(torch.tensor(float(init_k))))
            self.log_F = nn.Parameter(torch.log(torch.tensor(float(init_F))))
        else:
            self.register_buffer('log_h', torch.log(torch.tensor(float(init_h))))
            self.register_buffer('log_k', torch.log(torch.tensor(float(init_k))))
            self.register_buffer('log_F', torch.log(torch.tensor(float(init_F))))

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
                timestamp: torch.Tensor) -> torch.Tensor:
        base = torch.sigmoid(self.linear(x))                # shape: (batch, num_neurons)
        t = timestamp
        if t.ndim == 1:
            t = t.view(-1, 1)
        xt = x_t(t)                                         # broadcastable to (batch, num_neurons)
        A = attention
        P = torch.clamp(match_prob, min=self.eps, max=1.0)
        W = context_weight
        # broadcast A, P, W to (batch, num_neurons)
        A = A if A.shape == base.shape else A.view(A.size(0), 1).expand_as(base)
        P = P if P.shape == base.shape else P.view(P.size(0), 1).expand_as(base)
        W = W if W.shape == base.shape else W.view(W.size(0), 1).expand_as(base)
        h = self.effective_h()
        k = self.effective_k()
        Fc = self.effective_F()
        D = xt * (h * A + k * torch.log(P) + Fc * W)
        return base * D

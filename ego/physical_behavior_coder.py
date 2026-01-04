# physical_behavior_coder.py
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Önerilen referans sabitler (SI); model içinde ölçeklendirilecek
SI_H = 6.62607015e-34
SI_K = 1.380649e-23
SI_F = 96485.33212

def x_t(t: "torch.Tensor") -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for x_t")
    return torch.tanh(torch.exp(t) - math.pi)

class PhysicalBehaviorCoder(nn.Module):
    """
    Fiziksel sabitlerle davranış skoru hesaplayıcısı.
    Gerçek SI sabitleri learning_scale ile çarpılır; learning_scale learnable olabilir.
    """
    def __init__(self,
                 base_h: float = SI_H,
                 base_k: float = SI_K,
                 base_F: float = SI_F,
                 learnable_scale: bool = True,
                 eps: float = 1e-9):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for PhysicalBehaviorCoder")
        super().__init__()
        self.eps = eps
        # Sabitler sabit olarak saklanır, ölçekler öğrenilebilir veya sabit olabilir
        self.register_buffer('base_h', torch.tensor(float(base_h)))
        self.register_buffer('base_k', torch.tensor(float(base_k)))
        self.register_buffer('base_F', torch.tensor(float(base_F)))
        if learnable_scale:
            # log-space parametrizasyon ile pozitiflik garantisi
            self.log_scale_h = nn.Parameter(torch.log(torch.tensor(1e20)))  # init ölçekleri makul aralığa çek
            self.log_scale_k = nn.Parameter(torch.log(torch.tensor(1e20)))
            self.log_scale_F = nn.Parameter(torch.log(torch.tensor(1.0)))
        else:
            # sabit ölçek 1.0
            self.register_buffer('log_scale_h', torch.log(torch.tensor(1.0)))
            self.register_buffer('log_scale_k', torch.log(torch.tensor(1.0)))
            self.register_buffer('log_scale_F', torch.log(torch.tensor(1.0)))

    def effective_h(self) -> torch.Tensor:
        return self.base_h * torch.exp(self.log_scale_h)

    def effective_k(self) -> torch.Tensor:
        return self.base_k * torch.exp(self.log_scale_k)

    def effective_F(self) -> torch.Tensor:
        return self.base_F * torch.exp(self.log_scale_F)

    def compute_energy(self, attention: torch.Tensor) -> torch.Tensor:
        h = self.effective_h()
        return h * attention

    def compute_entropy(self, match_prob: torch.Tensor) -> torch.Tensor:
        k = self.effective_k()
        p = torch.clamp(match_prob, min=self.eps, max=1.0)
        return k * torch.log(p)

    def compute_transfer(self, context_weight: torch.Tensor) -> torch.Tensor:
        Fc = self.effective_F()
        return Fc * context_weight

    def forward(self,
                A: torch.Tensor,
                P: torch.Tensor,
                W: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        A, P, W ve t tensörleri broadcast uyumlu olmalıdır.
        D(t) çıktısı aynı broadcast shape'i döner.
        """
        energy = self.compute_energy(A)
        entropy = self.compute_entropy(P)
        transfer = self.compute_transfer(W)
        term = energy + entropy + transfer
        return x_t(t) * term

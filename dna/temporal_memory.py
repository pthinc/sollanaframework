# temporal_memory.py
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

def x_t(t: "torch.Tensor") -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for temporal_memory.x_t")
    return torch.tanh(torch.exp(t) - math.pi)

@dataclass
class TemporalTrace:
    id: str
    context: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    decay_rate: float = 0.01
    N0: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def age(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return now - self.last_used

    def activation(self, now: Optional[float] = None) -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        t = torch.tensor(self.age(now), dtype=torch.float32)
        return float(x_t(t).item())

    def effective_strength(self, now: Optional[float] = None) -> float:
        # decayed strength based on exponential decay from last_used
        now = now if now is not None else time.time()
        elapsed = now - self.last_used
        return self.N0 * math.exp(-self.decay_rate * elapsed)

    def trigger(self, delta_N: float = 0.0, now: Optional[float] = None):
        now = now if now is not None else time.time()
        # update N0 by adding evidence, reset last_used to now
        current_strength = self.effective_strength(now)
        self.N0 = current_strength + delta_N
        self.last_used = now
        self.usage_count += 1

class TemporalMemory:
    def __init__(self, prune_threshold: float = 1e-4):
        self.traces: Dict[str, TemporalTrace] = {}
        self.prune_threshold = prune_threshold

    def ensure_trace(self, trace_id: str, context: str, decay_rate: float = 0.01, init_strength: float = 1.0, meta: dict = None):
        if trace_id not in self.traces:
            self.traces[trace_id] = TemporalTrace(
                id=trace_id, context=context, decay_rate=decay_rate, N0=init_strength, meta=meta or {}
            )

    def trigger_behavior(self, trace_id: str, context: str, delta_N: float = 0.0, decay_rate: Optional[float] = None, now: Optional[float] = None):
        now = now if now is not None else time.time()
        if trace_id not in self.traces:
            self.ensure_trace(trace_id, context, decay_rate=decay_rate or 0.01, init_strength=delta_N if delta_N>0 else 1.0)
        trace = self.traces[trace_id]
        if decay_rate is not None:
            trace.decay_rate = decay_rate
        trace.trigger(delta_N=delta_N, now=now)

    def activation_vector(self, ids: List[str], now: Optional[float] = None) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for activation_vector")
        now = now if now is not None else time.time()
        acts = [self.traces[tid].activation(now) if tid in self.traces else 0.0 for tid in ids]
        return torch.tensor(acts, dtype=torch.float32)

    def strength_vector(self, ids: List[str], now: Optional[float] = None) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for strength_vector")
        now = now if now is not None else time.time()
        strengths = [self.traces[tid].effective_strength(now) if tid in self.traces else 0.0 for tid in ids]
        return torch.tensor(strengths, dtype=torch.float32)

    def sweep_prune(self, now: Optional[float] = None):
        now = now if now is not None else time.time()
        to_delete = [tid for tid, tr in self.traces.items() if tr.effective_strength(now) < self.prune_threshold]
        for tid in to_delete:
            del self.traces[tid]

    def list_traces(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        return {
            tid: {
                "context": tr.context,
                "usage_count": tr.usage_count,
                "last_used": tr.last_used,
                "age": now - tr.last_used,
                "activation": tr.activation(now),
                "strength": tr.effective_strength(now),
                "decay_rate": tr.decay_rate,
            }
            for tid, tr in self.traces.items()
        }

# PyTorch module that injects activation-weighted contributions into hidden states
class TemporalMemoryModule(nn.Module):
    def __init__(self, mem: TemporalMemory, device: Optional[torch.device] = None, scale: float = 1.0):
        super().__init__()
        self.mem = mem
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for TemporalMemoryModule")
        self.device = device or torch.device('cpu')
        self.scale = scale

    def forward(self, hidden: torch.Tensor, trace_ids: List[str]):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for TemporalMemoryModule.forward")
        if len(trace_ids) == 0:
            return hidden
        if hidden.shape[-1] != len(trace_ids):
            return hidden
        strengths = self.mem.strength_vector(trace_ids).to(self.device)
        acts = self.mem.activation_vector(trace_ids).to(self.device)
        # combined weight = activation * normalized strength
        s = acts * (strengths / (strengths.sum() + 1e-9))
        s = s.view(1, -1)  # (1, n_traces)
        weights = s.expand_as(hidden)
        norm_w = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        contribution = norm_w * hidden
        out = hidden * (1 - self.scale) + contribution * self.scale
        return out

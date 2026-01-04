"""Behavioral network demo with torch guard."""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from behavioral_neuron_vectorized import BehavioralNeuronLayer

class BehavioralMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_behaviors: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for BehavioralMLP")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.behavior_layer = BehavioralNeuronLayer(hidden_dim, num_behaviors)
        self.head = nn.Linear(num_behaviors, 1)

    def forward(self, x, attention, match_prob, context_weight, timestamp):
        h = self.encoder(x)
        b = self.behavior_layer(h, attention, match_prob, context_weight, timestamp)
        out = self.head(b)
        return out, b

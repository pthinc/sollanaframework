import random, math, numpy as np

def compute_hallucination_prob(base_p, context_match, data_quality, novelty, beta=1.0):
    return base_p * max(0.0, 1.0 - context_match) * max(0.0, 1.0 - data_quality) * (1.0 + beta * novelty)

def sample_random_variation(scale=0.02, seed=None):
    rng = random.Random(seed)
    return rng.uniform(-scale, scale)

def apply_behavior_noise(base_value, context_match, data_quality, novelty, base_p=0.02, scale=0.02, seed=None):
    p = compute_hallucination_prob(base_p, context_match, data_quality, novelty)
    if random.random() < p:
        eps = sample_random_variation(scale=scale, seed=seed)
        return base_value + eps, True, p
    return base_value, False, p

import torch

def inject_logit_noise(logits: torch.Tensor, context_match: float, data_quality: float,
                       novelty: float, base_p: float = 0.02, scale: float = 0.02, seed: int = None):
    p = compute_hallucination_prob(base_p, context_match, data_quality, novelty)
    if torch.rand(1).item() < p:
        g = torch.Generator()
        if seed is not None: g.manual_seed(seed)
        noise = torch.randn_like(logits, generator=g) * scale
        return logits + noise, True, p
    return logits, False, p

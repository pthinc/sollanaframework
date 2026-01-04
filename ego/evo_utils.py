# evo_utils.py
import math, random

def mutation_rate_from_score(score: float, base: float = 0.05, min_rate: float = 0.005, max_rate: float = 0.5):
    s = float(max(0.0, score))
    rate = min_rate + (base - min_rate) * (1.0 / (1.0 + math.exp(5.0 * (s - 1.0))))
    return max(min_rate, min(max_rate, rate))

def mutate_weights(weights, rate, scale=0.01, seed=None):
    rng = random.Random(seed)
    return [w + rng.gauss(0, scale) * rate for w in weights]

# usage: rate = mutation_rate_from_score(pattern_score); new = mutate_weights(old, rate)

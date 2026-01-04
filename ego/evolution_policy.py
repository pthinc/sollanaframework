# evolution_policy.py
def mutation_rate_from_pattern_score(pattern_score: float, base_rate: float = 0.05, min_rate: float = 0.005, max_rate: float = 0.5):
    # yüksek score -> düşük mutation rate (inverse sigmoid)
    import math
    s = float(pattern_score)
    rate = min_rate + (base_rate - min_rate) * (1.0 / (1.0 + math.exp(5.0 * (s - 1.0))))
    return float(max(min_rate, min(max_rate, rate)))

def apply_evolutionary_policy(variant_population, tracker: BehaviorPatternTracker):
    # variant_population: list of dicts with behavior_id and mutation metadata
    for v in variant_population:
        bid = v.get("behavior_id")
        score = tracker.pattern_strength(bid) if bid else 0.0
        v["mutation_rate"] = mutation_rate_from_pattern_score(score)
    return variant_population

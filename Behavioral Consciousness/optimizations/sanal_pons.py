"""
Sanal Pons: bridge module to combine cortical (reasoning) and limbic (affect) signals.
Outputs a harmonized signal and imbalance flag.
"""
from typing import Dict, Any
import math

EPS = 1e-12


def bridge(cortical: float, limbic: float, weight_c: float = 0.6, weight_l: float = 0.4) -> Dict[str, Any]:
    c = float(max(0.0, min(1.0, cortical)))
    l = float(max(0.0, min(1.0, limbic)))
    w_sum = weight_c + weight_l + EPS
    h = (weight_c * c + weight_l * l) / w_sum
    imbalance = abs(c - l)
    tone = math.tanh(h * 2.0)
    return {"harmonized": h, "tone": tone, "imbalance": imbalance}


def example_usage():
    return bridge(0.7, 0.5)


if __name__ == "__main__":
    print(example_usage())

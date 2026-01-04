# rhythm_balance.py
import math, numpy as np
from typing import Dict, Tuple, Callable

EPS = 1e-12

def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a + EPS)
    return (c - x) / (c - b + EPS)

class FuzzyMetric:
    def __init__(self, low_thr: float, mid_thr: float, high_thr: float):
        self.a = low_thr; self.b = mid_thr; self.c = high_thr
    def mu(self, x: float) -> Dict[str,float]:
        low = triangular(x, 0.0, 0.0, self.a)
        med = triangular(x, self.a, self.b, self.c)
        high = triangular(x, self.c, 1.0, 1.0)
        s = low + med + high + EPS
        return {"Low": low / s, "Medium": med / s, "High": high / s}

def saliniim_oneri_probability(T: float, B: float, R: float,
                               T_metric: FuzzyMetric, B_metric: FuzzyMetric, R_metric: FuzzyMetric) -> float:
    muT = T_metric.mu(T)["High"]
    muB = B_metric.mu(B)["High"]
    muR = R_metric.mu(R)["High"]
    S_o = muT * muB * (1.0 - muR)
    return float(max(0.0, min(1.0, S_o)))

# örnek kullanımı
Tm = FuzzyMetric(0.3, 0.6, 0.85)
Bm = FuzzyMetric(0.25, 0.6, 0.85)
Rm = FuzzyMetric(0.2, 0.5, 0.8)
prob = saliniim_oneri_probability(0.8, 0.75, 0.2, Tm, Bm, Rm)

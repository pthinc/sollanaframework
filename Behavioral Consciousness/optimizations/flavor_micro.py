# flavor_micro.py
class FlavorEvolver:
    def __init__(self, eps=0.01, min_step=1e-4, max_step=0.05):
        self.eps = eps
        self.min_step = min_step
        self.max_step = max_step
    def step(self, current: float, delta_signal: float) -> float:
        # delta_signal in [-1,1]; compute micro-step
        step = max(self.min_step, min(self.max_step, self.eps * delta_signal))
        return float(max(0.0, min(1.0, current + step)))

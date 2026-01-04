# suglist_opt.py
import itertools
import numpy as np

def sug_score(resonance, context_consistency, alpha, beta):
    # normalize inputs 0..1
    r = float(np.clip(resonance, 0.0, 1.0))
    c = float(np.clip(context_consistency, 0.0, 1.0))
    return float(alpha * r + beta * c)

class SugListOptimizer:
    def __init__(self, alpha_grid=None, beta_grid=None, normalize=True):
        self.alpha_grid = alpha_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.beta_grid = beta_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.normalize = normalize
    def normalize_pair(self, a, b):
        s = a + b
        if s <= 0: return 0.5, 0.5
        return a / s, b / s
    def optimize(self, candidates):
        # candidates: list of {"action","resonance","context_consistency","meta":...}
        best = None
        best_config = None
        for a,b in itertools.product(self.alpha_grid, self.beta_grid):
            aa, bb = (a,b)
            if self.normalize:
                aa, bb = self.normalize_pair(a,b)
            scores = []
            for c in candidates:
                sc = sug_score(c.get("resonance",0.0), c.get("context_consistency",0.0), aa, bb)
                scores.append((c, sc))
            # aggregate policy: choose top-k sum or mean
            total = sum(s for (_,s) in scores)
            if best is None or total > best:
                best = total
                best_config = {"alpha": aa, "beta": bb, "total_score": total, "scores": scores}
        return best_config

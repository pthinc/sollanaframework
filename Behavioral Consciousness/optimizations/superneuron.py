"""
SuperNeuron: open/close epoch gating with decay and Bayesian weight update (inspired by superneuronandmidlob).
"""
from typing import Dict, Any
import numpy as np

EPS = 1e-12


class SuperNeuron:
    def __init__(self, dim: int = 8, v_th: float = 0.5, decay: float = 0.01):
        self.w_mu = np.zeros(dim, dtype=float)
        self.w_var = np.ones(dim, dtype=float)
        self.bias = 0.0
        self.v_th = v_th
        self.decay = decay
        self.membrane = 0.0
        self.epoch_open = False

    def open_epoch(self):
        self.epoch_open = True
        self.membrane = 0.0

    def close_epoch(self):
        self.epoch_open = False
        self.membrane = 0.0

    def bayes_update(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        # simple Gaussian posterior update
        prior_prec = 1.0 / (self.w_var + EPS)
        like_prec = 1.0
        post_prec = prior_prec + like_prec
        post_mu = (prior_prec * self.w_mu + like_prec * x) / post_prec
        self.w_mu = post_mu
        self.w_var = 1.0 / post_prec

    def step(self, x: np.ndarray) -> Dict[str, Any]:
        if not self.epoch_open:
            self.open_epoch()
        x = np.asarray(x, dtype=float)
        self.bayes_update(x)
        v = float(np.dot(self.w_mu, x) + self.bias)
        v = v * np.exp(-self.decay)
        self.membrane += v
        fired = self.membrane >= self.v_th
        if fired:
            self.membrane = 0.0  # reset after spike
        return {"potential": v, "membrane": self.membrane, "fired": fired}


def example():
    sn = SuperNeuron(dim=4, v_th=0.8, decay=0.02)
    sn.open_epoch()
    for _ in range(5):
        x = np.random.rand(4)
        res = sn.step(x)
        if res["fired"]:
            break
    sn.close_epoch()
    return res

if __name__ == "__main__":
    print(example())

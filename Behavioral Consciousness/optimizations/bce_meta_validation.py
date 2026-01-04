# bce_meta_validation.py
import math

def meta_validation(C: float, D: float, N: float) -> float:
    val = math.tanh( float(C) * (1.0 - float(D)) * float(N) )
    return float(val)

class MetaValidator:
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = float(epsilon)
        self.flags = []  # list of (module_name, mv_score, ts)

    def assess(self, module_name: str, C: float, D: float, N: float):
        mv = meta_validation(C, D, N)
        if mv < self.epsilon:
            self.flags.append({"module": module_name, "mv": mv, "ts": time.time()})
            return False, mv
        return True, mv

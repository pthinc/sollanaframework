# cyber_soul_tag.py
"""Cyber Soul Tag - reference implementation (prototype)."""
import time, math, json, heapq, random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# --- Utilities ---
EPS = 1e-12
PHI = 1.6180339887498948

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS; nb = np.linalg.norm(b) + EPS
    return float(np.dot(a,b)/(na*nb))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# --- Data structures ---
@dataclass
class Tag:
    vec: np.ndarray            # v_i: data embedding
    context: str               # c_i
    flavor: str                # f_i
    consistency: float         # τ_i in [0,1]
    resonance: float           # ρ_i in [0,1]
    meta: Dict[str,Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

# --- Segment summarizer (lightweight) ---
class SegmentSummarizer:
    def __init__(self, embed_dim: int = 128, seed: int = 0):
        self.embed_dim = embed_dim
        self.rng = np.random.RandomState(seed)
    def embed_block(self, block_text: str, meta: Dict[str,Any]) -> Tag:
        # prototype embedding: hashed random + simple stats. Replace with real embed model.
        rng = self.rng
        vec = rng.randn(self.embed_dim).astype(float)
        # normalize
        vec = vec / (np.linalg.norm(vec) + EPS)
        # meta extraction (placeholders)
        context = meta.get("context","unknown")
        flavor = meta.get("flavor","neutral")
        consistency = float(meta.get("consistency", 0.8))
        resonance = float(meta.get("resonance", 0.5))
        return Tag(vec=vec, context=context, flavor=flavor, consistency=consistency, resonance=resonance, meta=meta)

# --- TagChain assembly and link scoring ---
class TagChain:
    def __init__(self):
        self.tags: List[Tag] = []
    def append(self, tag: Tag):
        self.tags.append(tag)
    def link_score(self, i: int) -> float:
        if i < 0 or i >= len(self.tags)-1:
            return 0.0
        s1, s2 = self.tags[i], self.tags[i+1]
        sim = max(-1.0, min(1.0, cos_sim(s1.vec, s2.vec)))
        # PID drift placeholder factor (1 - abs(delta_consistency))
        drift = 1.0 - abs(s1.consistency - s2.consistency)
        return float(sim * drift)

# --- Fuzzy optimizer for tag smoothing ---
class FuzzyOptimizer:
    def __init__(self, weights: Optional[Dict[str,float]] = None):
        w = weights or {"w1":0.4,"w2":0.3,"w3":0.3,"w4":0.6,"w5":0.4}
        self.w = w
    def memberships(self, tag: Tag) -> Tuple[float,float]:
        v = float(np.mean(tag.vec))  # cheap proxy for data signal
        mu_crit = sigmoid(self.w["w1"]*v + self.w["w2"]*hash(tag.context)%10*0.01 + self.w["w3"]*tag.consistency)
        mu_res = math.tanh(self.w["w4"]*tag.resonance + self.w["w5"]*(len(tag.flavor)))
        return mu_crit, mu_res
    def smooth_chain(self, chain: TagChain, iters: int = 1):
        # simple local smoothing: nudge consistency/resonance towards neighbor mean
        for _ in range(iters):
            new_cons = []
            new_res = []
            for i, t in enumerate(chain.tags):
                neigh = []
                if i>0: neigh.append(chain.tags[i-1])
                if i<len(chain.tags)-1: neigh.append(chain.tags[i+1])
                if neigh:
                    mean_cons = sum(n.consistency for n in neigh)/len(neigh)
                    mean_res = sum(n.resonance for n in neigh)/len(neigh)
                    # weighted update
                    new_cons.append(t.consistency*0.6 + mean_cons*0.4)
                    new_res.append(t.resonance*0.6 + mean_res*0.4)
                else:
                    new_cons.append(t.consistency)
                    new_res.append(t.resonance)
            for i,t in enumerate(chain.tags):
                t.consistency = float(max(0.0, min(1.0, new_cons[i])))
                t.resonance = float(max(0.0, min(1.0, new_res[i])))

# --- Superego gating + priority token queue ---
@dataclass(order=True)
class PrioritizedToken:
    priority: float
    token_id: str = field(compare=False)
    payload: Dict[str,Any] = field(compare=False)

class SuperegoQueue:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha; self.beta = beta
        self.heap: List[PrioritizedToken] = []
    def push(self, token_id: str, context_importance: float, drift_risk: float, payload: Dict[str,Any]):
        p = self.alpha*context_importance + self.beta*drift_risk
        heapq.heappush(self.heap, PrioritizedToken(-p, token_id, payload))  # max-heap via negative
    def pop_best(self):
        if not self.heap: return None
        item = heapq.heappop(self.heap)
        return item
    def flush_below(self, threshold: float):
        kept = []
        flushed = []
        while self.heap:
            item = heapq.heappop(self.heap)
            p = -item.priority
            if p >= threshold:
                kept.append(item)
            else:
                flushed.append(item)
        for k in kept:
            heapq.heappush(self.heap, k)
        return flushed

# --- PID controller for adaptive epsilon thresholding ---
class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.01, dt=1.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.dt = dt
        self.integral = 0.0; self.prev_err = 0.0
    def step(self, error: float):
        self.integral += error * self.dt
        deriv = (error - self.prev_err) / (self.dt + EPS)
        out = self.kp*error + self.ki*self.integral + self.kd*deriv
        self.prev_err = error
        return out

# --- Decay controller and rollback blueprint selector ---
class DecayController:
    def __init__(self, gamma1=0.6, gamma2=0.4, delta=0.7):
        self.g1 = gamma1; self.g2 = gamma2; self.delta = delta
    def decay_score(self, buffer_level: float, error_rate: float) -> float:
        return float(self.g1*buffer_level + self.g2*error_rate)
    def should_rollback(self, buffer_level: float, error_rate: float) -> bool:
        return self.decay_score(buffer_level, error_rate) > self.delta
    def blueprint_selector(self, chain: TagChain):
        # lightweight: pick earliest tag with low consistency and low resonance
        candidates = sorted(chain.tags, key=lambda t: (t.consistency, -t.resonance))
        if candidates:
            return candidates[0]
        return None

# --- Flavor filter (semantic corpus placeholder) ---
class FlavorFilter:
    def __init__(self, metaphor_corpus: Optional[List[str]] = None, lambda_thresh: float = 0.8):
        self.corpus = metaphor_corpus or ["deep metaphor example"]
        self.lambda_thresh = lambda_thresh
    def flavor_score(self, token: str) -> float:
        # placeholder: higher for tokens containing metaphoric words
        for w in self.corpus:
            if w in token:
                return 1.0
        return 0.0
    def apply(self, token: str) -> Tuple[str,bool]:
        sc = self.flavor_score(token)
        if sc > self.lambda_thresh:
            # replacement strategy: simple neutralization
            return ("[phrase_adjusted]", True)
        return (token, False)

# --- Simple optimizer stub (GA/Bayes placeholders) ---
class HyperOptimizer:
    def __init__(self):
        pass
    def grid_search(self, fitness_fn, grid: Dict[str,List[Any]]):
        best = None; best_score = -1e9
        import itertools
        keys = list(grid.keys())
        for vals in itertools.product(*(grid[k] for k in keys)):
            conf = dict(zip(keys, vals))
            score = fitness_fn(conf)
            if score > best_score:
                best_score = score; best = (conf, score)
        return best

# --- Demo driver ---
def demo_run():
    summ = SegmentSummarizer()
    chain = TagChain()
    for i in range(6):
        meta = {"context":f"topic_{i%3}", "flavor":"playful" if i%2==0 else "neutral",
                "consistency":0.7 + 0.05*(i%2), "resonance":0.4 + 0.1*(i%3)}
        tag = summ.embed_block(f"block_{i}", meta)
        chain.append(tag)
    fuzz = FuzzyOptimizer()
    fuzz.smooth_chain(chain, iters=3)
    # priority queue example
    q = SuperegoQueue()
    for idx,t in enumerate(chain.tags):
        ci = 1.0 if t.context.startswith("topic_") else 0.2
        drift = 1.0 - t.consistency
        q.push(f"token_{idx}", ci, drift, {"tag_idx":idx})
    flushed = q.flush_below(0.2)
    dec = DecayController()
    print("Decay score:", dec.decay_score(buffer_level=0.6, error_rate=0.1))
    flt = FlavorFilter(metaphor_corpus=["metaphor"], lambda_thresh=0.5)
    print("Flavor adjust example:", flt.apply("this is a metaphor"))
    print("Demo done.")

if __name__ == "__main__":
    demo_run()

# decay_sim.py  (kısa, çalıştırılabilir örnek)
import math, random, numpy as np

def lambda_from_half(half_tokens):
    return math.log(2.0)/float(half_tokens)

# params
half_short = 4000
lam_short = lambda_from_half(half_short)
gamma = 0.2
alpha_freq = 0.5
eta = 0.1
tau = 0.1

# synthetic set-up
N = 2000
s0 = np.random.rand(N)  # initial scores 0..1
freq = np.random.rand(N) * 0.005  # baseline recent hit rate per token
hits_prob = lambda i: min(1.0, 0.001 + 5*freq[i])  # toy

def step_decay(s, lam, delta_t, hit, gamma):
    s_new = s * math.exp(-lam * delta_t) + (hit * gamma)
    return max(0.0, min(1.0, s_new))

# run simple token‑based sim
def run_sim(steps=10000, delta_t=1):
    s = s0.copy()
    hist = []
    for t in range(steps):
        # simulate hits
        hits = np.array([1 if random.random() < hits_prob(i) else 0 for i in range(N)])
        # frequency aware lambda
        f = np.minimum(1.0, 0.01 + 50*freq)  # toy normalized f
        lam_t = lam_short * (1.0 - (f ** alpha_freq))
        for i in range(N):
            s[i] = step_decay(s[i], lam_t[i], delta_t, hits[i], gamma)
        # reconstruction trigger (toy): if many items below s_min and cluster drift surrogate
        s_min = s0 * tau
        n_below = np.sum(s < s_min)
        if n_below > 0.01 * N:
            # local reconstruction: boost low items by cluster mean (simplified)
            cluster_mean = np.mean(s)
            low_idx = np.where(s < s_min)[0]
            s[low_idx] += eta * cluster_mean
            s = np.clip(s, 0.0, 1.0)
        if t % 1000 == 0:
            hist.append((t, np.mean(s), np.sum(s < 0.01)))
    return s, hist

if __name__ == "__main__":
    final_s, metrics = run_sim(steps=10000)
    print("metrics sample:", metrics[:5], "final mean", np.mean(final_s))

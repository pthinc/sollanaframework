"""
Adversarially robust coding helpers (ART-inspired):
- randomized smoothing encoder
- vote aggregator and margin scorer
- lightweight detox/clip utility
"""
from typing import Callable, Dict, Any, Tuple, List
import numpy as np

EPS = 1e-12


def art_encode(x: np.ndarray, noise_sigma: float = 0.05, clip: float = 1.0, temperature: float = 1.0) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    noise = np.random.normal(scale=noise_sigma, size=x.shape)
    perturbed = x + noise
    if clip is not None:
        perturbed = np.clip(perturbed, -clip, clip)
    logits = perturbed / max(EPS, temperature)
    encoded = np.tanh(logits)
    return {"encoded": encoded, "noise": noise, "perturbed": perturbed}


def smooth_vote(predict_fn: Callable[[np.ndarray], int], x: np.ndarray, rounds: int = 32, noise_sigma: float = 0.08) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    votes: Dict[int, int] = {}
    for _ in range(int(max(1, rounds))):
        encoded = art_encode(x, noise_sigma=noise_sigma)["encoded"]
        pred = int(predict_fn(encoded))
        votes[pred] = votes.get(pred, 0) + 1
    total = sum(votes.values()) + EPS
    probs = {k: v / total for k, v in votes.items()}
    top = max(probs.items(), key=lambda kv: kv[1])
    # margin between best and second best
    sorted_probs = sorted(probs.values(), reverse=True)
    margin = float(sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0))
    return {"probs": probs, "top_class": top[0], "top_prob": top[1], "margin": margin}


def detox_clip(x: np.ndarray, beta: float = 0.1, clip: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    softened = np.tanh(beta * x)
    if clip is not None:
        softened = np.clip(softened, -clip, clip)
    return softened


def robust_score(model_fn: Callable[[np.ndarray], float], x: np.ndarray, radius: float = 0.1, samples: int = 24) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    scores: List[float] = []
    for _ in range(int(max(1, samples))):
        noise = np.random.normal(scale=radius, size=x.shape)
        s = float(model_fn(x + noise))
        scores.append(s)
    mean_s = float(np.mean(scores))
    return {"mean_score": mean_s, "min_score": float(np.min(scores)), "max_score": float(np.max(scores))}


def example_usage() -> Dict[str, Any]:
    def toy_model(z: np.ndarray) -> float:
        return float(np.tanh(np.sum(z)))
    x = np.random.rand(4)
    vote = smooth_vote(lambda z: int(np.sum(z) > 2.0), x)
    score = robust_score(toy_model, x)
    return {"vote": vote, "score": score}


if __name__ == "__main__":
    print(example_usage())

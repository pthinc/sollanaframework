"""
Hyperlogic / Bayes solver utilities:
- Dynamic programming/backtracking validator
- Simple Bayes posterior helper
"""
from typing import Callable, List, Any, Dict, Tuple


def backtrack_solve(candidates: List[Any], is_valid: Callable[[List[Any]], bool], score_fn: Callable[[List[Any]], float]) -> Tuple[List[Any], float]:
    best_seq: List[Any] = []
    best_score = float('-inf')

    def dfs(prefix: List[Any]):
        nonlocal best_seq, best_score
        if not is_valid(prefix):
            return
        if len(prefix) == len(candidates):
            s = score_fn(prefix)
            if s > best_score:
                best_score = s
                best_seq = prefix.copy()
            return
        next_idx = len(prefix)
        dfs(prefix + [candidates[next_idx]])
    dfs([])
    return best_seq, best_score


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    denom = evidence if evidence != 0 else (prior * likelihood + 1e-12)
    return float((likelihood * prior) / denom)

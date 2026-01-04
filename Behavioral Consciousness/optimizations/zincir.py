"""
Zincir: chain-of-approval / multi-stage guardrail aggregator.
Evaluates a list of checks and returns overall pass/fail with details.
"""
from typing import List, Dict, Callable, Any


def evaluate_chain(checks: List[Callable[[], bool]]) -> Dict[str, Any]:
    results = []
    passed = True
    for idx, fn in enumerate(checks):
        try:
            ok = bool(fn())
        except Exception as e:
            ok = False
            results.append({"step": idx, "ok": ok, "error": str(e)})
            passed = False
            break
        results.append({"step": idx, "ok": ok})
        if not ok:
            passed = False
            break
    return {"passed": passed, "results": results}


def example_usage():
    checks = [lambda: True, lambda: True, lambda: False]
    return evaluate_chain(checks)


if __name__ == "__main__":
    print(example_usage())

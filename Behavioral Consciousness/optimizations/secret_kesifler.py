"""
Secret Kesifler: gated discovery logging with sensitivity tagging and audit trail.
"""
from typing import Dict, Any, List
import time
import json
import os

LOG_PATH = os.path.join(os.getcwd(), "secret_kesifler.log")


def record(discovery: Dict[str, Any], sensitivity: str = "medium", allow: bool = False) -> Dict[str, Any]:
    ts = time.time()
    entry = {
        "ts": ts,
        "discovery": discovery,
        "sensitivity": sensitivity,
        "allow": allow,
    }
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return entry


def gated(discovery: Dict[str, Any], sensitivity: str = "medium", policy: str = "deny_by_default") -> Dict[str, Any]:
    allow = False
    actions: List[str] = []
    if policy == "allow_low" and sensitivity == "low":
        allow = True
    if sensitivity == "high":
        actions.append("escalate")
    if not allow:
        actions.append("quarantine")
    entry = record(discovery, sensitivity=sensitivity, allow=allow)
    entry["actions"] = actions
    return entry


def example_usage():
    return gated({"hypothesis": "latent cluster found"}, sensitivity="high", policy="allow_low")


if __name__ == "__main__":
    print(example_usage())

"""
Sanal hipofiz guard utilities: immutable snapshot, alarms, canary rollbacks.
"""
from typing import Dict, Any, Callable
import time
import json


def immutable_snapshot(snapshot_fn: Callable[[str, Dict[str, Any]], None], item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not callable(snapshot_fn):
        return {"ok": False, "reason": "no_snapshot_fn"}
    try:
        snapshot_fn(item_id, payload)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_alarms(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, Any]:
    breaches = {}
    for k, thr in thresholds.items():
        val = float(metrics.get(k, 0.0))
        if val > thr:
            breaches[k] = val
    res = {"alarm": bool(breaches), "breaches": breaches, "ts": time.time()}
    try:
        with open("hipofiz_alarm.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return res


def canary_gate(current: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    """Return True if safe to promote canary->prod."""
    alarms = check_alarms(current, thresholds)
    return not alarms["alarm"]


def log_recovery(event_fn: Callable[[Dict[str, Any]], None], status: str, meta: Dict[str, Any]) -> None:
    if callable(event_fn):
        event_fn({"ts": time.time(), "status": status, **meta})

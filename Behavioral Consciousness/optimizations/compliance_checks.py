"""
Lightweight ISO compliance signal stubs for telemetry (not certification-grade).
Provides simple heuristic scores for ISO 9000, 13485, 27001, 26000, 14001 based on context flags.
"""
from __future__ import annotations
from typing import Dict, Any


def compliance_scores(context: Dict[str, Any]) -> Dict[str, Any]:
    # Heuristic toggles; replace with real audits/policies as needed.
    quality = float(context.get("quality_ok", 0.8))           # ISO 9000
    medical_qms = float(context.get("med_qms_ok", 0.7))       # ISO 13485
    infosec = float(context.get("infosec_ok", 0.75))          # ISO/IEC 27001
    social = float(context.get("social_ok", 0.7))             # ISO 26000
    env = float(context.get("env_ok", 0.7))                   # ISO 14001
    return {
        "iso9000": quality,
        "iso13485": medical_qms,
        "iso27001": infosec,
        "iso26000": social,
        "iso14001": env,
    }


__all__ = ["compliance_scores"]

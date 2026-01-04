"""
Sicimsel optimizasyon sabitleri ve kritik dinamik hesaplari.
Bu modül, 10->1 boyut indirgeme fit'inden gelen evrensel sabitleri
( k, gamma, tau, delta_sigma ) kullanarak residual/enerji metriklerini
hesaplar ve sisteme hafif bir stabilizasyon sinyali sunar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class SicimselConstants:
    k: float = 0.10288565627
    gamma: float = 4.0
    tau: float = 2.4298819588
    delta_sigma: float = 2.7813217407489662e-14

    def derived(self) -> Dict[str, float]:
        sigma_star = 1.0 - self.delta_sigma
        delta_u = 0.5 * self.k * (self.delta_sigma ** 2)
        r_over_l = sigma_star ** (1.0 / 9.0) / (2.0 * math.pi)
        kk_gap = 1.0 / max(r_over_l, EPS)
        residual_alpha = 1.0 / max(self.gamma, EPS)
        effective_depth = self.gamma * self.tau
        damping_ratio = self.gamma / max(2.0 * math.sqrt(max(self.k, EPS)), EPS)
        return {
            "sigma_crit": 1.0,
            "delta_u": delta_u,
            "r_over_l": r_over_l,
            "kk_gap": kk_gap,
            "residual_alpha": residual_alpha,
            "effective_depth": effective_depth,
            "damping_ratio": damping_ratio,
        }

    def as_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data.update(self.derived())
        return data


class SicimselOptimizer:
    def __init__(self, constants: SicimselConstants | None = None):
        self.constants = constants or SicimselConstants()

    def compute_constants(self) -> Dict[str, float]:
        return self.constants.as_dict()

    def _sigma(self, phi_vec: Sequence[float]) -> float:
        arr = np.asarray(phi_vec, dtype=float).flatten()
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.square(arr)))

    def evaluate(self, phi_vec: Sequence[float], dt: float = 1.0, residual_overdrive: float = 1.0) -> Dict[str, Any]:
        const = self.constants
        sigma_val = self._sigma(phi_vec)
        sigma_err = sigma_val - 1.0
        energy = 0.5 * const.k * (sigma_err ** 2)

        residual_alpha = (1.0 / max(const.gamma, EPS)) * residual_overdrive
        stabilization_factor = 1.0 - dt * (const.k / max(const.gamma, EPS)) * sigma_err

        arr = np.asarray(phi_vec, dtype=float).flatten()
        norm_before = float(np.linalg.norm(arr)) if arr.size else 0.0
        norm_after = float(np.linalg.norm(arr * stabilization_factor)) if arr.size else 0.0

        return {
            "sigma": sigma_val,
            "sigma_error": sigma_err,
            "energy": energy,
            "residual_alpha": residual_alpha,
            "stabilization_factor": stabilization_factor,
            "norm_before": norm_before,
            "norm_after": norm_after,
            "constants": self.compute_constants(),
        }


__all__ = ["SicimselOptimizer", "SicimselConstants"]

"""
Sanal Pons style string/Fourier vectorization helper.
Converts a time series x(t) into a 2N+2 vector: [a, b, A1, B1, ..., AN, BN].
"""

from typing import Iterable, List
import numpy as np


def fourier_mod_vector(series: Iterable[float], n_harmonics: int = 4) -> List[float]:
    arr = np.asarray(list(series), dtype=float)
    if arr.size == 0:
        return [0.0]*(2*n_harmonics+2)
    t = np.linspace(0, 1, num=arr.size)
    a = float(arr[0])
    b = float(arr[-1] - arr[0])
    comps: List[float] = [a, b]
    for n in range(1, n_harmonics+1):
        cos_part = float(np.sum(arr * np.cos(n*np.pi*t))) / (arr.size or 1)
        sin_part = float(np.sum(arr * np.sin(n*np.pi*t))) / (arr.size or 1)
        comps.extend([cos_part, sin_part])
    return comps

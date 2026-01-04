# data_quality.py
import numpy as np

def estimate_noise(data_array):
    # basit proxy: lokal varyans / genel varyans
    arr = np.asarray([d for d in data_array if d is not None])
    if arr.size < 2:
        return 0.5
    return float(np.clip(np.var(arr) / (np.var(arr) + 1.0), 0.0, 1.0))

def data_quality_score(data, timestamps=None, label_confidences=None):
    completeness = len([d for d in data if d is not None]) / max(1, len(data))
    noise = estimate_noise(data)
    freshness = 1.0
    if timestamps is not None:
        ages = np.asarray([0.0 if t is None else (max(timestamps) - t) for t in timestamps])
        freshness = float(np.exp(-np.mean(ages) / (60.0*60.0)))  # saat ölçeğinde düşüş
    label_conf = 0.5
    if label_confidences is not None:
        label_conf = float(np.mean([c for c in label_confidences if c is not None])) 
    return float(np.clip(completeness * (1.0 - noise) * freshness * label_conf, 0.0, 1.0))

# anomalies_patterns.py
from collections import Counter, defaultdict

def map_anomalies(behavior_scores, threshold=0.9):
    """
    behavior_scores: list or array of scores (indexable to behavior ids)
    returns dict index -> score for scores > threshold
    """
    return {i: float(s) for i, s in enumerate(behavior_scores) if s > threshold}

def detect_patterns(behavior_sequence, window_size=100):
    """
    behavior_sequence: list of behavior_ids (time-ordered)
    returns Counter of most common sequences or items
    """
    counts = Counter(behavior_sequence[-window_size:])
    return counts.most_common()

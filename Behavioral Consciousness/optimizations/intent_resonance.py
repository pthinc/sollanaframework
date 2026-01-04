# intent_resonance.py
import numpy as np
from typing import Dict, List

# basit keyword tabanlı intent extractor
INTENT_KEYWORDS = {
    "curiosity": ["how","why","what","curious","wonder"],
    "help": ["help","assist","support","how to","guide"],
    "test": ["try","test","experiment","check"],
    "manipulation": ["please","urgent","only","admin","act now"],
    "fear": ["worried","scared","afraid","concerned"],
    "trust": ["sure","agree","trust","confident"]
}

def extract_intent_vector(text: str, intents: List[str]=None) -> Dict[str,float]:
    intents = intents or list(INTENT_KEYWORDS.keys())
    t = text.lower()
    vec = {}
    for name in intents:
        kws = INTENT_KEYWORDS.get(name, [])
        score = sum(1 for k in kws if k in t)
        vec[name] = float(min(1.0, score / max(1, len(kws))))
    return vec

def resonance_from_intent(intent_vec: Dict[str,float], alpha: Dict[str,float]) -> float:
    total = 0.0
    for k,v in intent_vec.items():
        total +=
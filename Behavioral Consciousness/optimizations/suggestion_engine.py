# suggestion_engine.py
from typing import Dict
def suggest_annotations(func_name: str, doc: str, context_tokens: Dict[str,float]) -> Dict:
    # heuristic: map verbs to tags, look for 'support','help','auth' keywords
    mapping = {"support":"seek_support", "help":"assist", "login":"auth"}
    tag = "default_behavior"
    flavor = "neutral"
    for k,v in mapping.items():
        if k in func_name.lower() or (doc and k in doc.lower()):
            tag = mapping[k]
            flavor = "gentle"
            break
    return {"BehaviorTag": tag, "Flavor": flavor}

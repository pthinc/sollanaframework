# thought_to_cluster.py
from sklearn.metrics.pairwise import cosine_similarity

def match_thought_to_clusters(phi_thought, centers, tau=0.7):
    sims = cosine_similarity([phi_thought], centers).ravel()
    hits = [(i, float(s)) for i,s in enumerate(sims) if s>=tau]
    return hits

def sandbox_validate(thought_id, candidate_behavior, validator, max_steps=50):
    # validator returns True if safe and useful
    for step in range(max_steps):
        ok = validator(candidate_behavior)
        if ok:
            return True
    return False

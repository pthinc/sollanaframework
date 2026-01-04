# auto_meta_control.py
"""
AutoMetaControl v2.0
- Submodules: ThankYouAppender, SelfEvalPrompter, AutoDiscovery, ContextErrorChecker, SuggestionToDoBuilder
- Single-file module for integration with BCE pipeline
Dependencies: numpy
"""

import time
import json
import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -------------------------
# Utilities
# -------------------------
def now_ts() -> float:
    return time.time()

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# ThankYouAppender
# -------------------------
class ThankYouAppender:
    def __init__(self, suffix: str = "Teşekkür"):
        self.suffix = suffix

    def append(self, message: str) -> str:
        if not message:
            return self.suffix
        # ensure punctuation before suffix
        if message.strip()[-1] not in ".!?":
            message = message.strip() + "."
        return f"{message} {self.suffix}"

# -------------------------
# SelfEvalPrompter
# -------------------------
class SelfEvalPrompter:
    """
    Lightweight self-eval heuristics:
      - clarity: inverse of long sentence ratio
      - conciseness: target length window
      - repetition: token-level repetition heuristic
      - decay_risk estimate: based on provided meta (decay_level, novelty)
    """
    def __init__(self, templates: Optional[List[str]] = None):
        self.templates = templates or [
            "Kendini değerlendir: açıklık, uzunluk, tekrar, decay riski nedir?",
            "Bu yanıtın salınımı nasıl? (kısaca puanla)",
            "Decay riski var mı? Neden?"
        ]

    @staticmethod
    def _token_stats(text: str) -> Dict[str, float]:
        toks = text.split()
        n = len(toks)
        if n == 0:
            return {"n": 0, "avg_len": 0.0, "repetition": 0.0}
        avg_len = sum(len(t) for t in toks) / n
        uniq = len(set(toks))
        repetition = 1.0 - (uniq / n)
        return {"n": n, "avg_len": avg_len, "repetition": repetition}

    def eval_scores(self, response: str, meta: Dict[str, Any]) -> Dict[str, float]:
        stats = self._token_stats(response)
        # clarity heuristic: shorter sentences and low repetition
        clarity = float(max(0.0, min(1.0, 1.0 - 0.05 * (stats["avg_len"] - 5.0) - 0.5 * stats["repetition"])))
        # conciseness: optimal length around 20-50 tokens
        conc = float(max(0.0, min(1.0, 1.0 - abs(stats["n"] - 32) / 64.0)))
        # repetition already 0..1
        repetition = float(stats["repetition"])
        # decay risk estimate from meta if present, else heuristic from repetition and length
        decay_meta = float(meta.get("decay_level", 0.0))
        novelty = float(meta.get("novelty", 0.0))
        decay_risk = float(min(1.0, decay_meta * 1.2 + repetition * 0.6 + (1.0 - novelty) * 0.2))
        return {"clarity": clarity, "conciseness": conc, "repetition": repetition, "decay_risk": decay_risk}

    def prompt_for_response(self, response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        scores = self.eval_scores(response, meta)
        template = random.choice(self.templates)
        # short evaluate text
        eval_text = (
            f"Kısa değerlendirme: Açıklık {scores['clarity']:.2f}, "
            f"Öz {scores['conciseness']:.2f}, Tekrar {scores['repetition']:.2f}, "
            f"DecayRisk {scores['decay_risk']:.2f}."
        )
        return {"prompt": template, "eval_text": eval_text, "scores": scores}

# -------------------------
# AutoDiscovery
# -------------------------
class AutoDiscovery:
    """
    Novelty detection on pattern frequencies and optional embedding drift.
    N(t) = sum 1{P_i > mu_Pi + k * sigma_Pi}
    If embeddings provided, compute semantic_shift as cosine distance between recent centroid and historical centroid.
    """

    def __init__(self, k: float = 2.0, min_patterns: int = 8):
        self.k = float(k)
        self.min_patterns = int(min_patterns)
        self.history_patterns = []  # list of dicts {pattern_id:freq, ts:}
        self.history_embeddings = []  # list of (ts, centroid_vec)

    def ingest_pattern_snapshot(self, pattern_freq: Dict[str, int], ts: Optional[float] = None):
        ts = ts or now_ts()
        self.history_patterns.append({"freq": dict(pattern_freq), "ts": ts})
        # keep limited history
        if len(self.history_patterns) > 256:
            self.history_patterns.pop(0)

    def ingest_embedding_snapshot(self, emb_vec: np.ndarray, ts: Optional[float] = None):
        ts = ts or now_ts()
        self.history_embeddings.append((ts, np.asarray(emb_vec, dtype=float)))
        if len(self.history_embeddings) > 128:
            self.history_embeddings.pop(0)

    def novelty_score(self, current_freq: Dict[str, int]) -> Dict[str, Any]:
        # compute mean and std per pattern across history
        keys = set()
        for snap in self.history_patterns:
            keys.update(snap["freq"].keys())
        keys.update(current_freq.keys())
        if len(self.history_patterns) < 2:
            return {"N": 0.0, "novel_patterns": [], "counts": len(current_freq)}
        import numpy as np
        vals = {k: [] for k in keys}
        for snap in self.history_patterns:
            for k in keys:
                vals[k].append(snap["freq"].get(k, 0))
        novel = []
        for k in current_freq.keys():
            arr = np.array(vals[k], dtype=float)
            mu = float(arr.mean())
            sigma = float(arr.std())
            thr = mu + self.k * sigma
            if current_freq.get(k, 0) > thr:
                novel.append(k)
        N = float(len(novel))
        return {"N": N, "novel_patterns": novel, "counts": len(current_freq)}

    def semantic_shift(self) -> float:
        # compare last centroid to earlier centroid
        if len(self.history_embeddings) < 2:
            return 0.0
        last_ts, last = self.history_embeddings[-1]
        # compute historical centroid excluding last
        arr = np.vstack([e for (_, e) in self.history_embeddings[:-1]])
        hist_cent = arr.mean(axis=0)
        sim = safe_cosine(last, hist_cent)
        # shift = 1 - sim
        return float(1.0 - sim)

    def discovery_report(self, current_freq: Dict[str, int], current_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.ingest_pattern_snapshot(current_freq)
        if current_emb is not None:
            self.ingest_embedding_snapshot(current_emb)
        nov = self.novelty_score(current_freq)
        shift = self.semantic_shift()
        emergent = bool(nov["N"] > 0 or shift > 0.15)
        return {"novelty": nov, "semantic_shift": shift, "emergent": emergent}

# -------------------------
# ContextErrorChecker
# -------------------------
class ContextErrorChecker:
    """
    E_ctx(t) = 1 - sim(C(t), C(t-1)) / max(||C(t)||, ||C(t-1)||)
    returns 0..1 (0 good, 1 bad)
    """
    def __init__(self, threshold: float = 0.15):
        self.threshold = float(threshold)
        self.last_context = None  # store last context vector

    def check(self, current_context_vec: Optional[np.ndarray]) -> Dict[str, Any]:
        now = now_ts()
        if current_context_vec is None:
            return {"E_ctx": 0.0, "ok": True}
        cur = np.asarray(current_context_vec, dtype=float)
        if self.last_context is None:
            self.last_context = cur
            return {"E_ctx": 0.0, "ok": True}
        sim = safe_cosine(cur, self.last_context)
        denom = max(np.linalg.norm(cur), np.linalg.norm(self.last_context), 1e-9)
        e_ctx = float(1.0 - sim / denom)
        ok = e_ctx <= self.threshold
        # update last context only if not large error (so transient spikes not drift)
        if ok:
            self.last_context = cur
        return {"E_ctx": e_ctx, "ok": ok, "sim": sim}

# -------------------------
# SuggestionToDoBuilder
# -------------------------
class SuggestionToDoBuilder:
    """
    Templates produce suggestions and to-dos.
    SugList = f(Intent, E_ctx, N)
    """
    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = templates or {
            "advice": "Öneri: {advice}",
            "todo": "Yapılacak: {action}"
        }

    def generate(self, intent: str, e_ctx: float, novelty_N: float) -> Dict[str, Any]:
        suggestions = []
        todos = []
        # very simple policy
        if e_ctx > 0.4:
            suggestions.append("Bağlam net değil; netleştirici bir soru sor.")
            todos.append("Kullanıcı niyetini doğrula")
        else:
            suggestions.append(f"Bu konuda şunu deneyebilirsin: {intent} üzerine küçük bir adım at.")
            todos.append(f"{intent} için kısa bir kontrol listesi hazırla")
        # novelty adjustments
        if novelty_N > 0:
            suggestions.append("Yeni örüntü tespit edildi; kısa pilot test öner.")
            todos.append("Keşif örneklerini sandbox'ta test et")
        # format
        advice_text = [self.templates["advice"].format(advice=s) for s in suggestions]
        todo_text = [self.templates["todo"].format(action=t) for t in todos]
        return {"suggestions": advice_text, "todos": todo_text, "scores": {"E_ctx": e_ctx, "N": novelty_N}}

# -------------------------
# AutoMetaControl orchestrator
# -------------------------
class AutoMetaControl:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        # submodules with config
        self.thx = ThankYouAppender(suffix=cfg.get("thankyou_suffix", "Teşekkür"))
        self.selfeval = SelfEvalPrompter(templates=cfg.get("selfeval_templates"))
        self.discovery = AutoDiscovery(k=cfg.get("discovery_k", 2.0))
        self.ctxcheck = ContextErrorChecker(threshold=cfg.get("ctx_threshold", 0.15))
        self.sug = SuggestionToDoBuilder(templates=cfg.get("suggest_templates"))
        # IO config
        self.write_status = cfg.get("write_status", "auto")
        self.status_dir = cfg.get("status_dir", "autometa_status")
        if self.write_status != "off":
            ensure_dir(self.status_dir)

    def process_response(self,
                         response: str,
                         behavior_meta: Dict[str, Any],
                         context_vec: Optional[np.ndarray],
                         pattern_freq: Optional[Dict[str, int]] = None,
                         embed_vec: Optional[np.ndarray] = None,
                         intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry:
          - append thank you
          - self-eval prompt and scores
          - discovery report
          - context error check
          - suggestion list
        Returns dict with all outputs and optionally writes a status JSON
        behavior_meta may include: decay_level, novelty, trace_id, other tags
        """
        out = {}
        # 1 append thank you
        augmented = self.thx.append(response)
        out["augmented_response"] = augmented

        # 2 self-eval
        selfeval = self.selfeval.prompt_for_response(response, behavior_meta)
        out["selfeval"] = selfeval

        # 3 discovery
        freq = pattern_freq or {}
        discovery_report = self.discovery.discovery_report(freq, current_emb=embed_vec)
        out["discovery"] = discovery_report

        # 4 context check
        ctx_report = self.ctxcheck.check(context_vec)
        out["context_check"] = ctx_report

        # 5 suggestion builder
        intent_text = intent or behavior_meta.get("intent", "Öneri")
        suggestion = self.sug.generate(intent_text, ctx_report.get("E_ctx", 0.0), discovery_report.get("novelty", {}).get("N", 0.0))
        out["suggestion"] = suggestion

        # 6 bookkeeping and optional write
        out["meta"] = {"ts": now_ts(), "behavior_meta": behavior_meta}
        if self.write_status == "auto":
            fname = os.path.join(self.status_dir, f"autometa_{int(time.time()*1000)}.json")
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return out

# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    amc = AutoMetaControl()
    resp = "Bu konu hakkında kısa bir özet: BCE davranışsal izleri zamanla örüntüleşir"
    meta = {"decay_level": 0.02, "novelty": 0.1, "trace_id": "t123", "intent": "Kısa özetle"}
    # synthetic pattern freq
    freq = {"greet": 5, "ask_loc": 2, "help": 1}
    emb = np.random.rand(64)
    # process
    out = amc.process_response(resp, meta, context_vec=np.random.rand(64), pattern_freq=freq, embed_vec=emb, intent="Kısa özetle")
    print(json.dumps(out, ensure_ascii=False, indent=2))

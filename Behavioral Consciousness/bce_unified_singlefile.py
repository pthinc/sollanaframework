# bce_unified_singlefile.py
"""
BCE Unified single-file prototype.
Modules included:
 - Utilities: cosine, tanh activation, safe math
 - MicroRL, MicroPID, Kalman, SlidingAnomalyDetector
 - ClusterManager (MiniBatch), MemoryLayers, MiniForget
 - FlavorSelector, SoulTag, SuperegoQueue
 - AutoMetaControl (thank you, self-eval, discovery, ctx check, suggestion)
 - PleasurePainController (conscious affect regulation)
 - TherapeuticSelfCalibrator (sleep/self-heal)
 - ExperienceTransformer (trace -> ΔSelf commit)
 - SanalOksipital manager orchestration
 - SelfHealEngine and Mini-forget orchestrations
This file is a working skeleton; replace hooks with real implementations.
Requires: Python 3.8+, numpy, sklearn (optional but recommended)
"""
import os, time, math, json, tempfile, random, heapq
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

# ---------- Utilities ----------
EPS = 1e-12
PI = math.pi

def now_ts() -> float:
    return time.time()

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + EPS; nb = np.linalg.norm(b) + EPS
    return float(np.dot(a, b) / (na * nb))

def atomic_write(path: str, obj: Any):
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirn)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

# ---------- Reinforcement / small learners ----------
class MicroRL:
    def __init__(self, arms: Optional[List[float]] = None, lr=0.05, eps=0.05):
        self.arms = arms or [0.0, 0.05, 0.1, 0.15]
        self.values = np.full(len(self.arms), 1.0, dtype=float)
        self.lr = lr; self.eps = eps
    def choose(self) -> int:
        if random.random() < self.eps:
            return random.randrange(len(self.arms))
        return int(np.argmax(self.values))
    def update(self, idx: int, reward: float):
        self.values[idx] += self.lr * (reward - self.values[idx])
    def best(self) -> float:
        return self.arms[int(np.argmax(self.values))]

class MicroPID:
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, dt=1.0, integ_max=10.0, deriv_tau=0.2):
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd); self.dt = float(dt)
        self.integ = 0.0; self.prev = 0.0; self.integ_max = float(integ_max)
        self.deriv = 0.0; self.tau = float(deriv_tau)
    def step(self, err: float) -> Tuple[float, Dict[str,float]]:
        self.integ += err * self.dt
        self.integ = max(-self.integ_max, min(self.integ, self.integ_max))
        rawd = (err - self.prev) / (self.dt + EPS)
        alpha = self.dt / (self.tau + self.dt)
        self.deriv = (1-alpha)*self.deriv + alpha*rawd
        P = self.kp * err
        I = self.ki * self.integ
        D = self.kd * self.deriv
        self.prev = err
        return P+I+D, {"P":P, "I":I, "D":D}
    def adapt(self, reward: float, lr=0.005):
        self.kp = max(0.0, min(5.0, self.kp + lr*reward))
        self.ki = max(0.0, min(1.0, self.ki + lr*0.1*reward))
        self.kd = max(0.0, min(1.0, self.kd + lr*0.01*reward))

class SimpleKalman:
    def __init__(self, q=1e-5, r=1e-2):
        self.q = q; self.r = r; self.x = 0.0; self.P = 1.0
    def update(self, z: float) -> float:
        self.P += self.q
        K = self.P / (self.P + self.r + EPS)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

class SlidingAnomalyDetector:
    def __init__(self, window_size=100, lambda_thresh=2.0):
        self.window_size = int(window_size); self.lambda_thresh = float(lambda_thresh)
        self.buf: List[float] = []
    def push(self, v: float):
        self.buf.append(float(v))
        if len(self.buf) > self.window_size: self.buf.pop(0)
    def is_anomaly(self, v: float) -> bool:
        if len(self.buf) < max(4, int(self.window_size*0.2)): return False
        arr = np.array(self.buf, dtype=float)
        mu, sigma = float(arr.mean()), float(arr.std())
        return v < (mu - self.lambda_thresh * sigma)

# ---------- Memory layers and embeddings ----------
class MemoryLayers:
    def __init__(self, short_len=8):
        self.short: List[np.ndarray] = []
        self.long: List[Dict[str,Any]] = []
        self.short_len = int(short_len)
    def push_short(self, emb: np.ndarray):
        self.short.append(np.asarray(emb, dtype=float))
        if len(self.short) > self.short_len: self.short.pop(0)
    def checkpoint(self) -> Optional[np.ndarray]:
        if not self.short: return None
        mean_emb = np.mean(np.vstack(self.short), axis=0)
        entry = {"ts": now_ts(), "emb": mean_emb}
        self.long.append(entry)
        return mean_emb

# ---------- Clustering / Sanal Oksipital ----------
try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None

class ClusterManager:
    def __init__(self, n_clusters=16, embed_dim=128, lambda_flavor=0.1, lambda_emotion=0.1):
        self.n_clusters = n_clusters; self.embed_dim = embed_dim
        self.lambda_flavor = lambda_flavor; self.lambda_emotion = lambda_emotion
        self.km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256) if MiniBatchKMeans is not None else None
        self.centers = np.zeros((n_clusters, embed_dim))
        self.fitted = False
    def partial_fit(self, X: np.ndarray):
        if self.km is None or X.shape[0] < 2: return
        if not self.fitted:
            self.km.partial_fit(X); self.fitted = True
        else:
            self.km.partial_fit(X)
        try: self.centers = self.km.cluster_centers_
        except: pass
    def stabilize(self, flavor_buffer_fn: Callable[[int], np.ndarray], emotion_imprint_fn: Callable[[int], np.ndarray]):
        for j in range(min(self.n_clusters, len(self.centers))):
            fb = flavor_buffer_fn(j) if callable(flavor_buffer_fn) else np.zeros(self.embed_dim)
            ei = emotion_imprint_fn(j) if callable(emotion_imprint_fn) else np.zeros(self.embed_dim)
            self.centers[j] = self.centers[j] + self.lambda_flavor*fb + self.lambda_emotion*ei

# ---------- Mini-Forget ----------
def mini_forget(items: List[Dict[str,Any]], theta: float = 0.5, gamma: float = 0.8, archive_threshold_steps: int = 6):
    archive = []
    for it in items:
        Q = float(it.get("relevance",0.0)) + float(it.get("resonance",0.0)) + float(it.get("context_fit",0.0))
        if Q <= theta:
            it["active_weight"] = it.get("active_weight",1.0) * gamma
            it["decay_steps"] = it.get("decay_steps",0) + 1
            if it["decay_steps"] >= archive_threshold_steps and it["active_weight"] < 1e-3:
                archive.append(it["id"])
    return archive

# ---------- SoulTag and Flavor selection ----------
class FlavorSelector:
    def __init__(self, prototypes: Dict[str, np.ndarray], superego_check: Optional[Callable[[Dict[str,Any]],bool]] = None):
        self.protos = {k: np.asarray(v, dtype=float) for k,v in prototypes.items()}
        self.superego_check = superego_check or (lambda p: True)
    def choose(self, context_vec: np.ndarray) -> Tuple[str, float]:
        best, best_score = None, -1.0
        for name, vec in self.protos.items():
            s = float(cos_sim(vec, context_vec))
            if s > best_score:
                best, best_score = name, s
        allowed = self.superego_check({"flavor": best, "score": best_score})
        return (best if allowed else "gentle_fallback", best_score)

# ---------- Superego queue ----------
@dataclass
class PrioritizedToken:
    priority: float
    token_id: str
    payload: Dict[str,Any]

class SuperegoQueue:
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha; self.beta = beta; self.heap: List[Tuple[float,Dict]] = []
    def push(self, token_id: str, context_importance: float, drift_risk: float, payload: Dict[str,Any]):
        p = self.alpha*context_importance + self.beta*drift_risk
        heapq.heappush(self.heap, (-p, token_id, payload))
    def pop_best(self):
        if not self.heap: return None
        return heapq.heappop(self.heap)

# ---------- AutoMetaControl (compact) ----------
class AutoMetaControl:
    def __init__(self, status_dir: str = "autometa_status"):
        self.status_dir = status_dir; os.makedirs(self.status_dir, exist_ok=True)
    def append_thankyou(self, message: str) -> str:
        if not message: return "Teşekkür"
        if message.strip()[-1] not in ".!?": message = message.strip() + "."
        return f"{message} Teşekkür"
    def self_eval(self, response: str, meta: Dict[str,Any]) -> Dict[str,float]:
        toks = response.split(); n = len(toks)
        rep = 1.0 - (len(set(toks)) / (n+EPS))
        clarity = max(0.0, min(1.0, 1.0 - 0.05*(sum(len(t) for t in toks)/max(1,n) - 5.0) - 0.5*rep))
        decay_risk = float(meta.get("decay_level", 0.0))
        return {"clarity": clarity, "conciseness": max(0.0, 1.0 - abs(n-32)/64.0), "decay_risk": decay_risk}
    def discovery(self, pattern_freq: Dict[str,int], k: float = 2.0) -> Dict[str,Any]:
        keys = list(pattern_freq.keys())
        nov = []
        if len(keys) < 1:
            return {"novelty":0.0,"novel_patterns":[]}
        arr = np.array(list(pattern_freq.values()), dtype=float)
        mu, sigma = float(arr.mean()), float(arr.std())
        for kname, v in pattern_freq.items():
            if v > mu + k*sigma:
                nov.append(kname)
        return {"novelty": float(len(nov)), "novel_patterns": nov}

# ---------- Pleasure–Pain PID controller ----------
class PleasurePainController:
    def __init__(self, s_opt=0.0):
        self.pid = MicroPID()
        self.s_opt = float(s_opt)
    def compute_S(self, desire: float, pleasure: float, pain: float) -> float:
        return float(desire - pleasure + pain)
    def step(self, desire: float, pleasure: float, pain: float) -> Dict[str,Any]:
        S = self.compute_S(desire, pleasure, pain)
        err = S - self.s_opt
        u, comps = self.pid.step(err)
        tampon = max(-1.0, min(1.0, u))
        return {"S":S, "error":err, "tampon":tampon, "pid":comps}
    def apply_reward(self, dpsi: float, daff: float):
        gamma1, gamma2 = 1.0, 0.7
        R = gamma1*dpsi + gamma2*daff
        self.pid.adapt(R)
        return R

# ---------- Therapeutic Self-Calibrator ----------
class TherapeuticSelfCalibrator:
    def __init__(self,
                 get_load: Callable[[],float],
                 memory_cleaner: Callable[[float],Dict],
                 param_stabilizer: Callable[[float],Dict],
                 discovery_runner: Callable[[float],Dict],
                 anomaly_repairer: Callable[[float],Dict],
                 send_user_message: Callable[[str],None],
                 resume_cb: Callable[[],None],
                 telemetry: Callable[[Dict],None]):
        self.get_load = get_load
        self.memory_cleaner = memory_cleaner; self.param_stabilizer = param_stabilizer
        self.discovery_runner = discovery_runner; self.anomaly_repairer = anomaly_repairer
        self.send_user_message = send_user_message; self.resume_cb = resume_cb
        self.telemetry = telemetry
        self.low_threshold = 0.15; self.slots = []
    def detect_slots(self) -> List[float]:
        samples = [self.get_load() for _ in range(12)]
        avg = sum(samples)/len(samples); var = sum((x-avg)**2 for x in samples)/len(samples)
        if avg < self.low_threshold and var < 0.0005:
            now = now_ts(); return [now+5+i*60 for i in range(6)]
        return [now_ts()+10.0]
    def run_slot(self, slot_ts: float):
        if slot_ts > now_ts(): time.sleep(max(0.0, slot_ts - now_ts()))
        try: self.send_user_message("Kendimi kısa bir bakım molası için kalibre ediyorum, 30s içinde döneceğim 🤖")
        except: pass
        budget = 30.0; start = now_ts(); res = {}
        try:
            t = min(8.0, budget); res['mem'] = self.memory_cleaner(t); budget -= (now_ts() - start)
            t = min(6.0, budget); res['params'] = self.param_stabilizer(t); budget -= (now_ts() - start)
            t = min(8.0, budget); res['disc'] = self.discovery_runner(t); budget -= (now_ts() - start)
            t = min(6.0, budget); res['repair'] = self.anomaly_repairer(t)
        except Exception as e:
            res['error'] = str(e)
        try: self.resume_cb()
        except: pass
        self.telemetry({"event":"selfcal_complete","results":res,"elapsed": now_ts()-start})
        return res

# ---------- Experience Transformer ----------
class ExperienceTransformer:
    def __init__(self, flavor_protos: Dict[str,float], decay_threshold: float = 0.3, superego_check: Optional[Callable[[Dict],bool]] = None, output_dir: str = "bce_traces"):
        self.flavor_protos = flavor_protos; self.decay_threshold = float(decay_threshold)
        self.superego_check = superego_check or (lambda p: True)
        self.output_dir = output_dir; os.makedirs(output_dir, exist_ok=True)
    def map_emotion(self, ctx: str, r: float, meta: Dict) -> str:
        if r > 0.85: return "Şükran" if "grat" in ctx.lower() else "Saygı"
        if r > 0.6: return "Merak"
        if r > 0.3: return "Hüzün"
        return "Nötr"
    def choose_flavor(self, ctx: str, r: float) -> str:
        best = max(self.flavor_protos.items(), key=lambda kv: kv[1]*r)[0]
        allowed = self.superego_check({"flavor":best,"resonance":r})
        return best if allowed else "gentle_fallback"
    def transform(self, user_id: str, x_t: Dict[str,Any], pi_constant: str = "stable", meta: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        meta = meta or {}
        decay = float(x_t.get("decay_risk",0.0)); resonance = float(x_t.get("resonance",0.0))
        decay_filtered = decay <= self.decay_threshold
        status = "committed" if decay_filtered else "quarantined"
        emotion = self.map_emotion(x_t.get("context",""), resonance, meta)
        flavor = self.choose_flavor(x_t.get("context",""), resonance)
        memory_trace = f"context={x_t.get('context','')} resonance={resonance:.3f} decay={decay:.3f}"
        delta_self = {"emotional_cluster":emotion, "flavor":flavor, "memory_trace":memory_trace, "decay_filtered":decay_filtered}
        obj = {"bce_trace":{"timestamp":now_iso(),"user_id":user_id,"x_t":x_t,"π_sabiti":pi_constant,"experience_vector":{"ΔSelf_t":delta_self},"write_status":status,"read_status": "available" if status=="committed" else "unavailable"}}
        fname = os.path.join(self.output_dir, f"bce_{user_id}_{int(now_ts()*1000)}.json")
        atomic_write(fname, obj)
        return obj

# ---------- Sanal Oksipital orchestrator ----------
class SanalOksipital:
    def __init__(self, embed_dim=128):
        self.memory = MemoryLayers(short_len=8)
        self.cluster_mgr = ClusterManager(n_clusters=16, embed_dim=embed_dim)
        self.kalman = SimpleKalman(); self.pid = MicroPID()
        self.anom_detector = SlidingAnomalyDetector(window_size=100)
        self.tag_priors: Dict[str,float] = {}
    def ingest_embedding(self, emb: np.ndarray):
        self.memory.push_short(emb)
        if len(self.memory.short) >= self.memory.short_len:
            ck = self.memory.checkpoint()
            if ck is not None:
                X = np.vstack([ck + 0.01*np.random.randn(*ck.shape) for _ in range(64)])
                self.cluster_mgr.partial_fit(X)
    def detect_decay_reflex(self, D_now: float, D_prev: float, delta_tokens=2000, rd_thresh=0.02):
        rd = (D_now - D_prev) / max(1, delta_tokens)
        if rd > rd_thresh:
            return {"reflex":True, "actions":["snapshot","quarantine"]}
        return {"reflex":False}
    def update_and_mini_forget(self, items: List[Dict[str,Any]], theta=0.5, gamma=0.85):
        return mini_forget(items, theta, gamma)
    def anomaly_check(self, value: float):
        self.anom_detector.push(value)
        return self.anom_detector.is_anomaly(value)

# ---------- Demo and simple hooks ----------
def _stub_get_load(): return 0.05
def _stub_mem_clean(budget): time.sleep(min(0.01,budget)); return {"removed":10}
def _stub_param_stab(budget): time.sleep(min(0.01,budget)); return {"tuned":True}
def _stub_discovery(budget): time.sleep(min(0.01,budget)); return {"new":0}
def _stub_repair(budget): time.sleep(min(0.01,budget)); return {"repairs":0}
def _stub_send(msg): print("USER MSG:", msg)
def _stub_resume(): print("RESUME")
def _stub_tele(e): print("TELEM:", e)

if __name__ == "__main__":
    # quick integration demo
    print("BCE unified demo start")
    oks = SanalOksipital(embed_dim=64)
    # simulate ingestion
    for i in range(120):
        emb = np.random.randn(64)
        oks.ingest_embedding(emb)
    # test decay reflex
    reflex = oks.detect_decay_reflex(0.12, 0.01)
    print("Reflex:", reflex)
    # run self calibrator
    sc = TherapeuticSelfCalibrator(_stub_get_load, _stub_mem_clean, _stub_param_stab, _stub_discovery, _stub_repair, _stub_send, _stub_resume, _stub_tele)
    slots = sc.detect_slots()
    print("Slots:", slots)
    res = sc.run_slot(slots[0])
    print("Self-calib result:", res)
    # experience transform
    et = ExperienceTransformer({"aesthetic_resonance":0.9,"gentle":0.5,"playful":0.6})
    sample_xt = {"context":"deneyim test","decay_risk":0.05,"resonance":0.92,"norm_alignment":True}
    out = et.transform("ahmet", sample_xt)
    print("Experience written:", out["bce_trace"]["write_status"])
    # pleasure-pain
    pp = PleasurePainController()
    pstep = pp.step(0.8,0.3,0.1)
    print("PleasurePain step:", pstep)
    print("BCE unified demo end")

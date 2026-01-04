# therapeutic_selfcal.py
"""
Therapeutic Self-Calibration - prototype
Hook functions should be provided by the host system:
- get_system_load() -> float (0..1)
- memory_cleaner(budget_s) -> dict
- param_stabilizer(budget_s) -> dict
- discovery_runner(budget_s) -> dict
- anomaly_repairer(budget_s) -> dict
- send_user_message(text) -> None
- telemetry_emit(event: dict) -> None
- resume_callback() -> None
"""

import time, math, threading
from typing import Callable, Dict, Any, List, Optional

# configuration
LOW_LOAD_THRESHOLD = 0.15      # Load < θ considered low
LOAD_STABLE_WINDOW = 60       # seconds, require stability
SLEEP_SLOTS_PER_DAY = 6
MAX_TOTAL_SEC = 30.0          # total self‑heal budget
MIN_SLOT_GAP = 60             # seconds between sleep windows
USER_MESSAGE = "Kendimi kısa bir bakım molası için kalibre ediyorum, 30 saniye içinde döneceğim 🤖"

class TherapeuticSelfCalibrator:
    def __init__(self,
                 get_system_load: Callable[[], float],
                 memory_cleaner: Callable[[float], Dict[str,Any]],
                 param_stabilizer: Callable[[float], Dict[str,Any]],
                 discovery_runner: Callable[[float], Dict[str,Any]],
                 anomaly_repairer: Callable[[float], Dict[str,Any]],
                 send_user_message: Callable[[str], None],
                 resume_callback: Callable[[], None],
                 telemetry_emit: Optional[Callable[[Dict[str,Any]], None]] = None,
                 low_load_threshold: float = LOW_LOAD_THRESHOLD):
        self.get_system_load = get_system_load
        self.memory_cleaner = memory_cleaner
        self.param_stabilizer = param_stabilizer
        self.discovery_runner = discovery_runner
        self.anomaly_repairer = anomaly_repairer
        self.send_user_message = send_user_message
        self.resume_callback = resume_callback
        self.telemetry_emit = telemetry_emit or (lambda e: None)
        self.low_load_threshold = low_load_threshold
        self.scheduled_slots: List[float] = []
        self.lock = threading.Lock()
        self.last_run_ts = 0.0

    # --- scheduling / detection ---
    def detect_low_load_epochs(self, horizon_seconds: int = 24*3600, sample_interval: float = 5.0, required_stable: int = 12) -> List[float]:
        """
        Light heuristic: sample load periodically for a short window and return candidate times (epoch seconds).
        In practice call once per day to compute candidate T_sleep slots (wallclock seconds).
        This function here returns relative offsets (now + offset) for simplicity.
        """
        samples = []
        stable_count = 0
        candidates = []
        now = time.time()
        # quick rolling sample for ~sample_interval*required_stable seconds
        for i in range(required_stable):
            l = self.get_system_load()
            samples.append(l)
            time.sleep(sample_interval)
        avg = sum(samples)/len(samples)
        var = sum((x-avg)**2 for x in samples)/len(samples)
        # if avg < threshold and low variance, produce candidate slots spaced by MIN_SLOT_GAP
        if avg < self.low_load_threshold and var < 0.0005:
            # propose up to SLEEP_SLOTS_PER_DAY slots starting now+margin
            t0 = now + 5.0
            for k in range(SLEEP_SLOTS_PER_DAY):
                candidates.append(t0 + k * max(MIN_SLOT_GAP, sample_interval*required_stable))
        # fallback: schedule a single soon slot
        if not candidates:
            candidates = [now + 10.0]
        self.scheduled_slots = candidates
        self.telemetry_emit({"event":"sleep_slots_computed","slots":candidates,"avg_load":avg,"var":var})
        return candidates

    # --- core sleep execution ---
    def run_sleep_slot(self, slot_ts: float):
        with self.lock:
            now = time.time()
            if slot_ts < now:
                # immediate run
                start_delay = 0.0
            else:
                start_delay = max(0.0, slot_ts - now)
        if start_delay > 0:
            time.sleep(start_delay)
        # guard: do not run too frequently
        if time.time() - self.last_run_ts < 10.0:
            return
        self.last_run_ts = time.time()
        # send user message
        try:
            self.send_user_message(USER_MESSAGE)
        except Exception:
            pass
        start = time.time()
        budget = MAX_TOTAL_SEC
        results = {}
        # a) memory clean
        t0 = time.time()
        d = min(budget, 8.0)
        try:
            results['memory'] = self.memory_cleaner(d)
        except Exception as e:
            results['memory'] = {"error": str(e)}
        budget -= (time.time() - t0)
        if budget <= 0: return self._finalize(start, results)
        # b) param stabilizer
        t0 = time.time()
        d = min(budget, 6.0)
        try:
            results['params'] = self.param_stabilizer(d)
        except Exception as e:
            results['params'] = {"error": str(e)}
        budget -= (time.time() - t0)
        if budget <= 0: return self._finalize(start, results)
        # c) discovery (limited)
        t0 = time.time()
        d = min(budget, 8.0)
        try:
            results['discovery'] = self.discovery_runner(d)
        except Exception as e:
            results['discovery'] = {"error": str(e)}
        budget -= (time.time() - t0)
        if budget <= 0: return self._finalize(start, results)
        # d) anomaly detection & repair
        t0 = time.time()
        d = min(budget, 6.0)
        try:
            results['repair'] = self.anomaly_repairer(d)
        except Exception as e:
            results['repair'] = {"error": str(e)}
        budget -= (time.time() - t0)
        # e) quick perf calibration if budget remains
        if budget > 0:
            t0 = time.time()
            try:
                # naive perf measurement hook via param_stabilizer with very small budget
                results['perf'] = {"note":"perf_check_done","elapsed":time.time()-t0}
            except Exception:
                results['perf'] = {"error":"perf_check_error"}
        # finalize
        return self._finalize(start, results)

    def _finalize(self, start_ts: float, results: Dict[str,Any]):
        elapsed = time.time() - start_ts
        # resume
        try:
            self.resume_callback()
        except Exception:
            pass
        self.telemetry_emit({"event":"sleep_complete","elapsed_s": elapsed, "results": results})
        return results

    # convenience runner for immediate test
    def immediate_selfcal(self):
        return self.run_sleep_slot(time.time())

# Example minimal stubs for host to replace when integrating
def _stub_get_load():
    return 0.05  # very low

def _stub_memory_cleaner(budget_s):
    time.sleep(min(0.01, budget_s))
    return {"removed": 42, "budget_used": min(0.01, budget_s)}

def _stub_param_stabilizer(budget_s):
    time.sleep(min(0.01, budget_s))
    return {"params_tuned": True, "budget_used": min(0.01, budget_s)}

def _stub_discovery_runner(budget_s):
    time.sleep(min(0.01, budget_s))
    return {"new_patterns": 0, "budget_used": min(0.01, budget_s)}

def _stub_anomaly_repairer(budget_s):
    time.sleep(min(0.01, budget_s))
    return {"repairs": 0, "budget_used": min(0.01, budget_s)}

def _stub_send_msg(txt):
    print("USER_MSG:", txt)

def _stub_resume():
    print("Resuming normal ops")

def _stub_telemetry(e):
    print("TELEMETRY:", e)

if __name__ == "__main__":
    sc = TherapeuticSelfCalibrator(
        get_system_load=_stub_get_load,
        memory_cleaner=_stub_memory_cleaner,
        param_stabilizer=_stub_param_stabilizer,
        discovery_runner=_stub_discovery_runner,
        anomaly_repairer=_stub_anomaly_repairer,
        send_user_message=_stub_send_msg,
        resume_callback=_stub_resume,
        telemetry_emit=_stub_telemetry
    )
    sc.detect_low_load_epochs()
    print("Running immediate self-calibration...")
    res = sc.immediate_selfcal()
    print("Done:", res)

# bce_psych_snapshot.py
import pickle
import os
import time

def psych_score(D: float, N: float, C: float, alpha=1.0, beta=1.0, gamma=1.0) -> float:
    return float(alpha * D + beta * (1.0 - N) + gamma * (1.0 - C))

class SnapshotManager:
    def __init__(self, snap_dir: str = "snapshots", max_snap: int = 10):
        self.snap_dir = snap_dir
        os.makedirs(self.snap_dir, exist_ok=True)
        self.max_snap = int(max_snap)

    def save_snapshot(self, name: str, state: Dict):
        path = os.path.join(self.snap_dir, f"{int(time.time())}_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        # prune older
        snaps = sorted(os.listdir(self.snap_dir))
        if len(snaps) > self.max_snap:
            for rm in snaps[:len(snaps)-self.max_snap]:
                try: os.remove(os.path.join(self.snap_dir, rm))
                except: pass
        return path

    def restore_if_needed(self, current_state: Dict, check_func: Callable[[Dict], bool], restore_to: Optional[str]=None):
        # if check_func signals memory saturation or high decay, restore last good
        if not check_func(current_state):
            # find latest snapshot optionally matching restore_to
            snaps = sorted(os.listdir(self.snap_dir))
            if not snaps:
                return False, None
            latest = snaps[-1]
            with open(os.path.join(self.snap_dir, latest), "rb") as f:
                state = pickle.load(f)
            return True, state
        return True, current_state

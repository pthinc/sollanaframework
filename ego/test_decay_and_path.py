# test_decay_and_path.py
import time
from decay_and_path import DecayModel, PathTrace, default_transform

def test_decay_increase():
    dm = DecayModel(base_lambda=0.5)
    d0 = dm.decay(0.0)
    d1 = dm.decay(1.0)
    assert d0 < d1

def test_cumulative_phi_monotonic():
    dm = DecayModel(base_lambda=0.1)
    t = PathTrace("x", dm, decay_lambda=0.1)
    t.add_step("a", {"attention":0.5}, weight=1.0, transform=default_transform)
    time.sleep(0.01)
    now = time.time()
    c1 = t.cumulative_phi(now)
    time.sleep(0.05)
    c2 = t.cumulative_phi(time.time())
    assert c2 <= c1

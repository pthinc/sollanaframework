# pid_profiles.py
PROFILE_REGISTRY = {}

def PIDProfile(channel: str, Kp: float, Ki: float, Kd: float, dt: float = 0.1):
    def deco(fn):
        PROFILE_REGISTRY[channel] = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "dt": dt}
        return fn
    return deco

# usage
@PIDProfile("Connector", Kp=0.6, Ki=0.3, Kd=0.1)
def connect_behavior(*args, **kwargs):
    """Example PID profile stub."""
    return {"status": "ok", "channel": "Connector"}

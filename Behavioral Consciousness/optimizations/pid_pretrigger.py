# pid_pretrigger.py
class PIDPreTrigger:
    def __init__(self, pid_controller, e_thresh: float = 0.1, accel_factor: float = 2.0):
        self.pid = pid_controller
        self.e_thresh = float(e_thresh)
        self.accel_factor = float(accel_factor)

    def maybe_trigger(self, error: float):
        if abs(error) <= self.e_thresh:
            return {"triggered": False}
        # early proportional action
        P = self.pid.Kp * error
        # accelerate I and D temporarily
        self.pid.Ki *= self.accel_factor
        self.pid.Kd *= self.accel_factor
        # anti-windup: clamp integrator
        self.pid.integrator = max(-self.pid.integrator_max, min(self.pid.integrator, self.pid.integrator_max))
        return {"triggered": True, "P": P}

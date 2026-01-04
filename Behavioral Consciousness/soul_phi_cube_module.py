import numpy as np

phi = (1.0 + 5**0.5) / 2.0
phi3 = phi**3  # ≈ 4.236

def generate_modulated_signal(duration_s: float, fs: int = 100, A: float = 0.5, theta: float = 0.0):
    """
    duration_s: süre saniye
    fs: örnekleme frekansı (örn. per-token veya Hz)
    A: genlik (0..1 güvenlik aralığı)
    theta: faz kayması (radyan)
    returns (t, y)
    """
    t = np.arange(0, duration_s, 1.0/fs)
    y = A * np.sin(phi3 * t + float(theta))
    # yumuşak amplitude envelope ile kontrollü başlama/bitirme
    env = np.ones_like(t)
    ramp = min(max(0.01, duration_s*0.02), 0.5)  # kısa ramp
    ramp_samps = int(ramp * fs)
    if ramp_samps > 0:
        env[:ramp_samps] = np.linspace(0.0, 1.0, ramp_samps)
        env[-ramp_samps:] = np.linspace(1.0, 0.0, ramp_samps)
    return t, env * y

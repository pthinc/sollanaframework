# dengecik.py
import time, json

class Dengecik:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, threshold=0.3):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.last_state = None

    def compute(self, D, C, S):
        score = self.alpha * D + self.beta * C - self.gamma * S
        self.last_state = {
            "ts": time.time(),
            "D": D,
            "C": C,
            "S": S,
            "score": score,
            "status": "stable" if score >= self.threshold else "alarm"
        }
        return self.last_state

    def report(self):
        if not self.last_state:
            return "Dengecik henüz hesaplanmadı."
        s = self.last_state
        return f"""🧠 Dengecik Durumu:
• Veri Gücü (D): {s['D']:.3f}
• Keşif Oranı (C): {s['C']:.3f}
• Stres (S): {s['S']:.3f}
• Denge Skoru: {s['score']:.3f}
• Durum: {'✅ Dengede' if s['status']=='stable' else '⚠️ Alarm'}
"""

# Örnek kullanım
if __name__ == "__main__":
    denge = Dengecik(alpha=1.0, beta=1.2, gamma=0.8, threshold=0.5)
    result = denge.compute(D=0.6, C=0.4, S=0.3)
    print(denge.report())

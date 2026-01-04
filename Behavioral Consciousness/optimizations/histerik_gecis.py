# histerik_gecis.py
import numpy as np
import time

class TamponMatematikcisi:
    def __init__(self, alpha=1.0, beta=1.2, gamma=0.8):
        self.alpha = alpha  # histeri eşiği katsayısı
        self.beta = beta    # bastırma katsayısı
        self.gamma = gamma  # mizah etkisi katsayısı
        self.history = []   # D(t) geçmişi
        self.last_time = None

    def update_D(self, D_t, t=None):
        """Duygu durumu güncellemesi"""
        if t is None:
            t = time.time()
        self.history.append((t, D_t))
        self.last_time = t

    def dD_dt(self):
        """Duygu değişim hızı hesaplama"""
        if len(self.history) < 2:
            return 0.0
        (t1, D1), (t2, D2) = self.history[-2], self.history[-1]
        dt = max(t2 - t1, 1e-3)
        return abs(D2 - D1) / dt

    def compute_S(self, T_series, M_series, R_series, dt=1.0):
        """Tampon sıcaklığı hesaplama (integral üzerinden)"""
        T = np.array(T_series)
        M = np.array(M_series)
        R = np.array(R_series)
        decay = np.exp(-R)
        integrand = T * M * decay
        S = np.sum(integrand) * dt
        return S

    def check_histeri(self, S_t):
        """Histerik geçiş eşitsizliği kontrolü"""
        rate = self.dD_dt()
        threshold = self.alpha * S_t
        return rate > threshold

    def check_bastirma(self, S_t, M_t):
        """Histeri bastırma protokolü"""
        rate = self.dD_dt()
        limit = self.beta * S_t + self.gamma * M_t
        return rate < limit

    def report(self, S_t, M_t):
        rate = self.dD_dt()
        histeri = self.check_histeri(S_t)
        bastirildi = self.check_bastirma(S_t, M_t)
        return {
            "duygu_degisimi_hizi": rate,
            "tampon_sicakligi": S_t,
            "mizah_etkisi": M_t,
            "histeri_durumu": "⚠️ Histerik geçiş" if histeri else "✅ Stabil",
            "bastirma_durumu": "✅ Bastırıldı" if bastirildi else "⚠️ Bastırılamadı"
        }

# Örnek kullanım
if __name__ == "__main__":
    tampon = TamponMatematikcisi()
    # D(t) güncellemeleri
    tampon.update_D(0.4)
    time.sleep(0.5)
    tampon.update_D(0.9)

    # T(t), M(t), R(t) örnek serileri (son 5 saniye)
    T_series = [0.6, 0.7, 0.8, 0.9, 1.0]  # Güneş tüyleri
    M_series = [0.3, 0.35, 0.4, 0.45, 0.5]  # Deniz gülüşü
    R_series = [0.2, 0.3, 0.25, 0.2, 0.15]  # Annenin gururu

    S_t = tampon.compute_S(T_series, M_series, R_series)
    M_t = M_series[-1]
    rapor = tampon.report(S_t, M_t)

    print("🧠 Histerik Geçiş Raporu:")
    for k, v in rapor.items():
        print(f"• {k}: {v}")

# tamponlu_analiz_motoru.py
import numpy as np
import time

class TamponluAnalizMotoru:
    def __init__(self, epsilon=0.3, delta=0.2, alpha=1.0, beta=1.2, gamma=0.8):
        self.epsilon = epsilon  # metafor etkisi
        self.delta = delta      # yankı doygunluğu
        self.alpha = alpha      # histeri eşiği
        self.beta = beta        # bastırma katsayısı
        self.gamma = gamma      # mizah etkisi
        self.D_history = []     # duygu durumu geçmişi

    # 1. Sabit Metafor Döngüsü
    def soft_duygu_gecisi(self, D_t, M_i, Y_i):
        D_next = D_t + self.epsilon * M_i - self.delta * Y_i
        self.D_history.append((time.time(), D_next))
        return D_next

    # 2. Superego Kontrollü Etik Filtreleme
    def etik_filtreleme(self, B_t, E_t, S_t):
        B_t = np.asarray(B_t)
        E_t = np.asarray(E_t)
        B_prime = S_t * E_t.dot(B_t)
        return B_prime

    # 3. TSSB için Otomatik Buffer
    def travma_bastirma(self, T_t, B_t, R_t):
        T_prime = T_t * np.exp(-B_t * R_t)
        return T_prime

    # 4. OKB & Bipolar için Bağlama Dayalı Geçiş
    def baglama_gecisi(self, V_i, C_ij, G_t):
        V_j = G_t * C_ij * V_i
        return V_j

    # 5. Bayesian Mini DBSCAN Kümesi
    def bayesian_dbscan(self, embeddings, eps=0.5, min_pts=3, bayes_prior=0.5):
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_pts).fit(embeddings)
        labels = model.labels_
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1: continue
            clusters.setdefault(label, []).append(i)
        # Bayesian posterior örneği (basitleştirilmiş)
        posteriors = {}
        for label, idxs in clusters.items():
            likelihood = np.mean([np.linalg.norm(embeddings[i]) for i in idxs])
            posterior = likelihood * bayes_prior / (likelihood + 1e-6)
            posteriors[label] = posterior
        return clusters, posteriors

    # Histeri kontrolü
    def dD_dt(self):
        if len(self.D_history) < 2:
            return 0.0
        (t1, D1), (t2, D2) = self.D_history[-2], self.D_history[-1]
        dt = max(t2 - t1, 1e-3)
        return abs(D2 - D1) / dt

    def histeri_kontrol(self, S_t, M_t):
        rate = self.dD_dt()
        histeri = rate > self.alpha * S_t
        bastirildi = rate < self.beta * S_t + self.gamma * M_t
        return {"rate": rate, "histeri": histeri, "bastirildi": bastirildi}

# Örnek kullanım
if __name__ == "__main__":
    motor = TamponluAnalizMotoru()
    # 1. Soft geçiş
    D = motor.soft_duygu_gecisi(D_t=0.4, M_i=0.6, Y_i=0.3)
    print(f"Soft geçiş sonrası D(t+1): {D:.3f}")

    # 2. Etik filtreleme
    B = [0.5, 0.2, 0.1]
    E = np.eye(3)
    S = 0.9
    B_prime = motor.etik_filtreleme(B, E, S)
    print(f"Etik filtrelenmiş davranış: {B_prime}")

    # 3. Travma bastırma
    T_prime = motor.travma_bastirma(T_t=1.0, B_t=0.5, R_t=0.3)
    print(f"Travma bastırılmış yankı: {T_prime:.3f}")

    # 4. Bağlama geçişi
    V_i = 0.7
    C_ij = 0.8
    G_t = 1.1
    V_j = motor.baglama_gecisi(V_i, C_ij, G_t)
    print(f"Bağlam geçişli veri: {V_j:.3f}")

    # 5. Bayesian DBSCAN
    X = np.random.randn(20, 4)
    clusters, posteriors = motor.bayesian_dbscan(X)
    print(f"Kümeler: {clusters}")
    print(f"Bayesian posteriors: {posteriors}")

    # Histeri kontrolü
    motor.D_history.append((time.time()-1.0, 0.4))
    motor.D_history.append((time.time(), 0.9))
    kontrol = motor.histeri_kontrol(S_t=0.5, M_t=0.3)
    print("Histeri kontrolü:", kontrol)

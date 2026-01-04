# decay_and_norms.py
def dynamic_decay_threshold(N_t: float, grad_C_t: float, R_t: float, alpha: float=0.6, beta: float=0.3, gamma: float=0.1) -> float:
    """
    N_t norm uyumu 0..1
    grad_C_t karakter salınımındaki anlık değişim
    R_t rezonans 0..1
    returns delta threshold in 0..1
    """
    val = alpha * (1.0 - N_t) + beta * grad_C_t + gamma * (1.0 - R_t)
    return float(max(0.0, min(1.0, val)))

def norm_diagnostic(grad_N_t: float, D_t: float, R_t: float, S_t: float) -> float:
    """
    grad_N_t norm üretimindeki sapma
    D_t decay seviyesi
    R_t rezonans
    S_t SuperEGO sabitleyici 0..1
    returns diagnostic score higher means worse
    """
    return float(max(0.0, grad_N_t * D_t * (1.0 - R_t) * S_t))

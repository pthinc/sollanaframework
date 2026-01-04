# behavior_volume.py
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

EPS = 1e-12

def empirical_volume(x_vals: np.ndarray, m_vals: np.ndarray) -> float:
    """
    Fast relative volume: sum of x * M.
    x_vals shape (N,), m_vals shape (N,)
    """
    x_vals = np.asarray(x_vals)
    m_vals = np.asarray(m_vals)
    return float(np.sum(x_vals * m_vals))

def _kernel_regression_estimates(samples: np.ndarray, phi: np.ndarray, y: np.ndarray, bandwidth: float):
    """
    Kernel regression (RBF) to estimate E[y | sample] for each sample.
    Warning O(N*S) complexity; subsample phi if needed.
    """
    # gamma for rbf weight: exp(-||d||^2 / (2 * bw^2))
    bw2 = float(bandwidth)**2 + EPS
    dists = euclidean_distances(samples, phi, squared=True)  # shape (S, N)
    weights = np.exp(-0.5 * dists / bw2)
    wsum = np.sum(weights, axis=1, keepdims=True) + EPS
    est = (weights @ y.reshape(-1,1)) / wsum
    return est.ravel()  # shape (S,)

def kde_mc_volume(phi: np.ndarray, x_vals: np.ndarray, m_vals: np.ndarray,
                  bandwidth: float = 1.0, n_samples: int = 2000, pca_dim: int = None,
                  subsample: int = None, random_state: int = 0) -> float:
    """
    Estimate V = ∫ x(Φ)·M(Φ)·f(Φ) dΦ by sampling from KDE fit to phi.
    - phi: (N, d)
    - x_vals, m_vals: (N,)
    Returns scalar volume (relative).
    """
    N, d = phi.shape
    # optionally reduce dim for KDE
    phi_in = phi
    pca = None
    if pca_dim is not None and pca_dim < d:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        phi_in = pca.fit_transform(phi)

    # fit KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(phi_in)

    # sample from KDE (samples approximate f)
    samples = kde.sample(n_samples=n_samples, random_state=random_state)
    # optionally project back if PCA used
    # estimate E[x*M | sample] with kernel regression in phi_in space
    y = (np.asarray(x_vals) * np.asarray(m_vals)).astype(float)  # target = x*M
    # reduce expensive cost by subsampling training points if requested
    phi_train = phi_in
    y_train = y
    if subsample is not None and subsample < N:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, size=subsample, replace=False)
        phi_train = phi_in[idx]
        y_train = y[idx]

    est_y = _kernel_regression_estimates(samples, phi_train, y_train, bandwidth)
    # expectation under f approximated by sample mean
    V_est = float(np.mean(est_y))
    return V_est

def cluster_empirical_volumes(labels: np.ndarray, x_vals: np.ndarray, m_vals: np.ndarray) -> dict:
    """
    Return dict label -> empirical sum x*M for that cluster.
    label -1 or noise included.
    """
    labels = np.asarray(labels)
    x_vals = np.asarray(x_vals); m_vals = np.asarray(m_vals)
    out = {}
    for lab in np.unique(labels):
        mask = labels == lab
        out[int(lab)] = float(np.sum(x_vals[mask] * m_vals[mask]))
    return out

import numpy as np
from scipy import stats


def effect_from_z(z, n):
    n = np.asarray(n, dtype=np.float64)
    return z / np.sqrt(n)[..., None] if np.ndim(z) > np.ndim(n) else z / np.sqrt(n)


def heterogeneity(z1, z2, n1=None, n2=None):
    z1 = np.asarray(z1, dtype=np.float64); z2 = np.asarray(z2, dtype=np.float64)
    if n1 is None or n2 is None:
        Q = (z1 - z2) ** 2 / 2.0
    else:
        n1 = np.asarray(n1, dtype=np.float64)[..., None]; n2 = np.asarray(n2, dtype=np.float64)[..., None]
        e1 = z1 / np.sqrt(n1); e2 = z2 / np.sqrt(n2)
        Q = (e1 - e2) ** 2 / (1.0 / n1 + 1.0 / n2)
    return Q, stats.chi2.sf(Q, 1)


def bh(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    m = len(p); order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(adj, 1.0)
    return out


def replication_table(z1, z2, thr_discover=5.0, thr_replicate=2.0):
    d1 = np.abs(z1) >= thr_discover
    rep = d1 & (np.abs(z2) >= thr_replicate) & (np.sign(z1) == np.sign(z2))
    opp = d1 & (np.abs(z2) >= thr_replicate) & (np.sign(z1) != np.sign(z2))
    null = d1 & (np.abs(z2) < thr_replicate)
    return dict(discovered=int(d1.sum()), replicated=int(rep.sum()), opposite=int(opp.sum()), not_replicated=int(null.sum()))


def dose_adjusted_heterogeneity(z1, z2, disc_thr=5.0, min_edges=10):
    z1 = np.asarray(z1, dtype=np.float64); z2 = np.asarray(z2, dtype=np.float64)
    beta = np.ones(z1.shape[0])
    for r in range(z1.shape[0]):
        k = (np.abs(z1[r]) >= disc_thr) | (np.abs(z2[r]) >= disc_thr)
        if k.sum() >= min_edges:
            x = z1[r, k]; y = z2[r, k]
            sx = np.sign(x) * np.sqrt(np.abs(x)); sy = np.sign(y) * np.sqrt(np.abs(y))
            b = np.median(y / np.where(np.abs(x) > 0, x, np.nan)[~np.isnan(np.where(np.abs(x) > 0, x, np.nan))]) if np.any(np.abs(x) > 0) else 1.0
            beta[r] = float(np.clip(b, 0.2, 5.0)) if np.isfinite(b) else 1.0
    b = beta[:, None]
    Q = (z2 - b * z1) ** 2 / (1.0 + b ** 2)
    return Q, stats.chi2.sf(Q, 1), beta

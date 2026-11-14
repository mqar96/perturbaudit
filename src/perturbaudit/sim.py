import numpy as np


def random_dag(p, density, rng, wmin=0.4, wmax=1.0):
    order = rng.permutation(p)
    W = np.zeros((p, p))
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < density:
                W[order[a], order[b]] = rng.choice([-1, 1]) * rng.uniform(wmin, wmax)
    return W


def simulate_screen(p=40, density=0.06, n_obs=4000, n_per_pert=150, n_lat=4, lat_scale=0.6,
                    noise_df=5, kd_mean=2.0, off_frac=0.0, off_strength=1.6, inactive_frac=0.0,
                    misassign=0.0, seed=0):
    rng = np.random.default_rng(seed)
    W = random_dag(p, density, rng)
    L = rng.normal(0, lat_scale, (n_lat, p))
    T = np.linalg.inv(np.eye(p) - W)
    off = {i: int(rng.choice([k for k in range(p) if k != i])) for i in range(p) if rng.random() < off_frac}
    inactive = {i for i in range(p) if rng.random() < inactive_frac}

    def draw(n, kd=None):
        U = rng.normal(size=(n, n_lat)) @ L + rng.standard_t(noise_df, size=(n, p)) * 0.7
        s = np.zeros(p)
        if kd is not None and kd not in inactive:
            s[kd] = -kd_mean * rng.uniform(0.5, 1.0)
            if kd in off:
                s[off[kd]] = -off_strength
        return (U + s) @ T

    blocks, labels = [draw(n_obs)], ["non-targeting"] * n_obs
    for i in range(p):
        blocks.append(draw(n_per_pert, i)); labels += [i] * n_per_pert
    X = np.vstack(blocks)
    labels = np.array(labels, dtype=object)
    if misassign > 0:
        k = int(misassign * len(labels))
        idx = rng.choice(len(labels), k, replace=False)
        labels[idx] = labels[rng.permutation(idx)]
    desc = (np.abs(T) > 1e-12) & ~np.eye(p, dtype=bool)
    return dict(X=X, labels=labels, W=W, desc=desc, off=off, inactive=inactive)


def add_permutation_decoys(X, labels, K, n_cells, rng, kd_shift=None):
    p = X.shape[1]
    nt = np.flatnonzero(labels == "non-targeting")
    pick = rng.choice(nt, K * n_cells, replace=False)
    labels = labels.copy()
    for k in range(K):
        labels[pick[k * n_cells:(k + 1) * n_cells]] = f"DECOY{k:02d}"
    templates = rng.choice(p, K)
    Xd = np.empty((X.shape[0], K))
    for k in range(K):
        v = X[rng.permutation(X.shape[0]), templates[k]].copy()
        rows = pick[k * n_cells:(k + 1) * n_cells]
        v[rows] = v[rows] + (kd_shift if kd_shift is not None else -2.0 * X[:, templates[k]].std())
        Xd[:, k] = v
    return np.hstack([X, Xd]), labels


def gaussian_knockoffs(X_fit, X_apply, rng, shrink=0.1):
    mu = X_fit.mean(0); sd = X_fit.std(0) + 1e-9
    Zf = (X_fit - mu) / sd
    S = np.corrcoef(Zf, rowvar=False)
    S = (1 - shrink) * S + shrink * np.eye(S.shape[0])
    lam = np.linalg.eigvalsh(S).min()
    s = min(1.0, 2 * lam) * 0.999
    Sinv = np.linalg.inv(S)
    Dm = s * np.eye(S.shape[0])
    C2 = 2 * Dm - Dm @ Sinv @ Dm
    w, V = np.linalg.eigh((C2 + C2.T) / 2)
    C = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    Za = (X_apply - mu) / sd
    Zk = Za - Za @ Sinv @ Dm + rng.standard_normal(Za.shape) @ C
    return Zk * sd + mu


def add_decoys(X, labels, K_perm, K_knock, n_cells, rng, kd_sd=2.0):
    p = X.shape[1]
    nt = np.flatnonzero(labels == "non-targeting")
    K = K_perm + K_knock
    pick = rng.choice(nt, K * n_cells, replace=False)
    labels = labels.copy()
    names = []
    for k in range(K):
        nm = f"DECOYP{k:02d}" if k < K_perm else f"DECOYK{k - K_perm:02d}"
        labels[pick[k * n_cells:(k + 1) * n_cells]] = nm; names.append(nm)
    obs_rows = np.setdiff1d(nt, pick)
    Xk_all = gaussian_knockoffs(X[obs_rows], X, rng) if K_knock else None
    templates = rng.choice(p, K)
    Xd = np.empty((X.shape[0], K))
    for k in range(K):
        t = templates[k]
        v = X[rng.permutation(X.shape[0]), t].copy() if k < K_perm else Xk_all[:, t].copy()
        rows = pick[k * n_cells:(k + 1) * n_cells]
        v[rows] = v[rows] - kd_sd * X[obs_rows, t].std()
        Xd[:, k] = v
    return np.hstack([X, Xd]), labels, names

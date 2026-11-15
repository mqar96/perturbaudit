import numpy as np
from scipy import sparse


def split_nt(nt_gem, frac_ref=0.5, seed=0):
    rng = np.random.default_rng(seed)
    ref = np.zeros(len(nt_gem), dtype=bool)
    for g in np.unique(nt_gem):
        idx = np.flatnonzero(nt_gem == g)
        k = int(round(len(idx) * frac_ref))
        ref[rng.choice(idx, k, replace=False)] = True
    return ref


def make_decoy_groups(pool_gem, target_gem_counts, gem_levels, n_decoys=None, batch_matched=True, seed=0):
    rng = np.random.default_rng(seed)
    T = np.asarray(target_gem_counts)
    if n_decoys is None:
        n_decoys = T.shape[0]
    pick = rng.choice(T.shape[0], n_decoys, replace=n_decoys > T.shape[0])
    by_gem = {g: np.flatnonzero(pool_gem == g) for g in gem_levels}
    all_idx = np.arange(len(pool_gem))
    rows, cols = [], []
    for k, t in enumerate(pick):
        if batch_matched:
            members = []
            for gi, g in enumerate(gem_levels):
                c = int(T[t, gi])
                if c == 0:
                    continue
                cand = by_gem[g]
                if len(cand) == 0:
                    cand = all_idx
                members.append(rng.choice(cand, c, replace=c > len(cand)))
            members = np.concatenate(members) if members else np.array([], dtype=int)
        else:
            c = int(T[t].sum())
            members = rng.choice(all_idx, c, replace=c > len(all_idx))
        rows.append(np.full(len(members), k)); cols.append(members)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_decoys, len(pool_gem)))
    return M, pick


def group_moments(M, X):
    n = np.asarray(M.sum(1)).ravel()
    s1 = M @ X
    s2 = M @ (X.astype(np.float64) ** 2)
    return n, np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)


def null_var_z(n, s1, ref_mean, ref_var, ref_n):
    n = np.asarray(n, dtype=np.float64)[:, None]
    se = np.sqrt(ref_var[None, :] * (1.0 / n + 1.0 / ref_n))
    se = np.where(se > 0, se, np.inf)
    return (s1 / n - ref_mean[None, :]) / se


def welch_z(n, s1, s2, ref_mean, ref_var, ref_n):
    n = np.asarray(n, dtype=np.float64)[:, None]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 0) * n / np.maximum(n - 1, 1)
    se = np.sqrt(var / n + ref_var[None, :] / ref_n)
    se = np.where(se > 0, se, np.inf)
    return (mu - ref_mean[None, :]) / se


def target_decoy_fdr(real_scores, decoy_scores, thresholds):
    real = np.sort(np.asarray(real_scores, dtype=np.float64).ravel())
    dec = np.sort(np.asarray(decoy_scores, dtype=np.float64).ravel())
    n_real, n_dec = len(real), len(dec)
    out = []
    for t in thresholds:
        r = n_real - np.searchsorted(real, t, side="left")
        d = n_dec - np.searchsorted(dec, t, side="left")
        est = ((d + 1) / n_dec) * n_real / max(r, 1)
        out.append((t, int(r), int(d), min(est, 1.0)))
    return out


def qvalues(real_scores, decoy_scores):
    """Target-decoy q-values for larger-is-more-significant scores. Non-finite real scores get NaN;
    non-finite decoy scores are dropped."""
    allreal = np.asarray(real_scores, dtype=np.float64).ravel()
    fin = np.isfinite(allreal)
    out = np.full(len(allreal), np.nan)
    if not fin.any():
        return out
    dec = np.asarray(decoy_scores, dtype=np.float64).ravel()
    dec = np.sort(dec[np.isfinite(dec)])
    if len(dec) == 0:
        return out
    out[fin] = _qvalues_finite(allreal[fin], dec)
    return out


def _qvalues_finite(real, dec):
    order = np.argsort(-real)
    n_real, n_dec = len(real), len(dec)
    r_count = np.arange(1, n_real + 1)
    d_count = n_dec - np.searchsorted(dec, real[order], side="left")
    fdr = np.minimum(((d_count + 1) / n_dec) * n_real / r_count, 1.0)
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]
    q = np.empty(n_real); q[order] = q_sorted
    return q

import numpy as np


def top_k_graph(W, k):
    A = np.abs(np.asarray(W, dtype=np.float64)).copy()
    np.fill_diagonal(A, 0)
    if k >= A.size:
        return A > 0
    t = np.partition(A.ravel(), -k)[-k]
    return (A >= t) & (A > 0)


def descendants(G):
    G = np.asarray(G, dtype=bool)
    R = G.copy()
    while True:
        R2 = R | ((R.astype(np.uint8) @ G.astype(np.uint8)) > 0)
        if (R2 == R).all():
            break
        R = R2
    np.fill_diagonal(R, False)
    return R


def implied_effect_scores(G, Z, sig_thr=5.0, null_thr=2.0, perturbed=None):
    G = np.asarray(G, dtype=bool)
    p = G.shape[0]
    D = descendants(G)
    rows = np.arange(p) if perturbed is None else np.asarray(perturbed)
    Zs = np.abs(np.asarray(Z, dtype=np.float64))[rows][:, :p]
    Dr = D[rows]
    off = ~np.eye(p, dtype=bool)[rows]
    sig = (Zs >= sig_thr) & off
    recall = (sig & Dr).sum() / max(sig.sum(), 1)
    nondesc = (~Dr) & off
    violation = (sig & nondesc).sum() / max(nondesc.sum(), 1)
    support = ((Zs >= null_thr) & Dr).sum() / max(Dr.sum(), 1)
    density = G.sum() / (p * (p - 1))
    return dict(recall=float(recall), violation_rate=float(violation), descendant_support=float(support),
                n_descendant_pairs=int(Dr.sum()), n_sig_pairs=int(sig.sum()), density=float(density))


def random_graph_baseline(G, Z, n=50, seed=0, **kw):
    rng = np.random.default_rng(seed)
    p = G.shape[0]
    m = int(np.asarray(G, dtype=bool).sum())
    out = []
    for _ in range(n):
        R = np.zeros((p, p), dtype=bool)
        idx = rng.choice(p * p, m, replace=False)
        R.flat[idx] = True
        np.fill_diagonal(R, False)
        out.append(implied_effect_scores(R, Z, **kw))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def intervention_support(edges_mask, Ztce, low=2.0, high=5.0, kd_effective=None, return_untestable=False):
    E = np.asarray(edges_mask, dtype=bool)
    Za = np.abs(np.asarray(Ztce, dtype=np.float64))
    p = E.shape[0]
    Zp = Za[:p, :p]
    unsupported = E & (Zp < low)
    untestable = np.zeros_like(E)
    if kd_effective is not None:
        eff = np.asarray(kd_effective, dtype=bool)[:p]
        untestable = unsupported & ~eff[:, None]
        unsupported = unsupported & eff[:, None]
    reversed_ = unsupported & (Zp.T >= high)
    if return_untestable:
        return unsupported, reversed_, untestable
    return unsupported, reversed_

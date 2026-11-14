import sys, os, time, json, numpy as np, pandas as pd

def load(panel_dir):
    d = np.load(f"{panel_dir}/panel.npz", allow_pickle=True)
    return d["X"].astype(np.float64), d["labels"], list(d["names"])

def run_tce(X, labels, names):
    obs = X[labels == "non-targeting"]; mu, var, n0 = obs.mean(0), obs.var(0, ddof=1), len(obs)
    W = np.zeros((len(names), len(names)))
    for i, g in enumerate(names):
        m = labels == g
        if m.sum() < 5: continue
        W[i] = (X[m].mean(0) - mu) / np.sqrt(var * (1 / m.sum() + 1 / n0))
    np.fill_diagonal(W, 0)
    return np.abs(W)

def run_dotears(X, labels, names, lambda1=0.1):
    sys.path.insert(0, os.environ.get("DOTEARS_DIR", "repos/dotears"))
    from dotears import DOTEARS
    Xs = (X - X.mean(0)) / X.std(0)
    data = {"obs": Xs[labels == "non-targeting"]}
    for i, g in enumerate(names):
        m = labels == g
        if m.sum(): data[str(i)] = Xs[m]
    return np.abs(DOTEARS(data, lambda1=lambda1, scaled=False, w_threshold=0, max_iter=100).fit())

def subsample(X, labels, per_group=60, n_obs=3000, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        k = n_obs if g == "non-targeting" else per_group
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], labels[keep], keep


def run_sdcd(X, labels, names):
    import torch
    X, labels, _ = subsample(X, labels, int(os.environ.get("SUB_PER_GROUP", "60")), int(os.environ.get("SUB_OBS", "3000")))
    from sdcd.models import SDCD
    from sdcd.utils import create_intervention_dataset
    Xs = (X - X.mean(0)) / X.std(0)
    df = pd.DataFrame(Xs, columns=[str(n) for n in names]); df["perturbation_label"] = [str(l) for l in labels]
    ds = create_intervention_dataset(df, perturbation_colname="perturbation_label")
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))
    dev = 'cuda' if torch.cuda.is_available() else None
    ep = int(os.environ.get('SDCD_EPOCHS', '300')); bs = int(os.environ.get('SDCD_BATCH', '512'))
    m = SDCD(); m.train(ds, finetune=True, device=dev, stage1_kwargs={'n_epochs': ep, 'batch_size': bs}, stage2_kwargs={'n_epochs': ep, 'batch_size': bs})
    return np.abs(m.get_adjacency_matrix(threshold=False))

def run_gbm_grn(X, labels, names):
    from sklearn.ensemble import GradientBoostingRegressor
    Xs, _, _ = subsample(X, labels, int(os.environ.get("GBM_PER_GROUP", "30")), int(os.environ.get("GBM_OBS", "2000")))
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    p = Xs.shape[1]
    def fit_one(j):
        others = np.delete(np.arange(p), j)
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9, max_features=0.1,
                                      n_iter_no_change=10, validation_fraction=0.1, random_state=0)
        m.fit(Xs[:, others], Xs[:, j])
        return j, others, m.feature_importances_
    nj = int(os.environ.get("N_JOBS", "1"))
    if nj > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=nj)(delayed(fit_one)(j) for j in range(p))
    else:
        res = [fit_one(j) for j in range(p)]
    W = np.zeros((p, p))
    for j, others, imp in res:
        W[others, j] = imp
    return W


def run_grnboost2(X, labels, names):
    from arboreto.algo import grnboost2
    from distributed import LocalCluster, Client
    cl = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=False))
    df = pd.DataFrame(X, columns=[str(n) for n in names])
    net = grnboost2(expression_data=df, client_or_address=cl, seed=0)
    cl.close()
    idx = {n: i for i, n in enumerate(map(str, names))}
    W = np.zeros((len(names), len(names)))
    for r in net.itertuples():
        W[idx[r.TF], idx[r.target]] = r.importance
    return W

def run_inspre(X, labels, names):
    import subprocess, tempfile
    tmp = tempfile.mkdtemp()
    pd.DataFrame(X, columns=[str(n) for n in names]).to_csv(f"{tmp}/X.csv", index=False)
    open(f"{tmp}/targets.txt", "w").write("\n".join(str(l) for l in labels))
    subprocess.run(["Rscript", "perturbaudit/wrappers/run_inspre.R", f"{tmp}/X.csv", f"{tmp}/targets.txt", f"{tmp}/score.tsv"], check=True)
    S = pd.read_csv(f"{tmp}/score.tsv", sep="\t", index_col=0)
    W = np.zeros((len(names), len(names)))
    pos = {str(n): i for i, n in enumerate(names)}
    for r in S.index:
        for c in S.columns:
            W[pos[str(r)], pos[str(c)]] = S.at[r, c]
    import shutil; shutil.rmtree(tmp)
    return W


def run_adapre(X, labels, names):
    import subprocess, tempfile, shutil
    tmp = tempfile.mkdtemp()
    pd.DataFrame(X, columns=[str(n) for n in names]).to_csv(f"{tmp}/X.csv", index=False)
    open(f"{tmp}/targets.txt", "w").write("\n".join(str(l) for l in labels))
    subprocess.run(["Rscript", "perturbaudit/wrappers/run_adapre.R", f"{tmp}/X.csv", f"{tmp}/targets.txt", f"{tmp}/score.tsv"], check=True)
    S = pd.read_csv(f"{tmp}/score.tsv", sep="\t", index_col=0)
    W = np.zeros((len(names), len(names))); pos = {str(n): i for i, n in enumerate(names)}
    for r in S.index:
        for c in S.columns:
            W[pos[str(r)], pos[str(c)]] = S.at[r, c]
    shutil.rmtree(tmp)
    return W


def run_inspre_default(X, labels, names):
    os.environ["INSPRE_RHO"] = "100"
    try:
        return run_inspre(X, labels, names)
    finally:
        os.environ.pop("INSPRE_RHO", None)


def run_inspre_dag(X, labels, names):
    os.environ["INSPRE_DAG"] = "TRUE"
    try:
        return run_inspre(X, labels, names)
    finally:
        os.environ.pop("INSPRE_DAG", None)


def _run_r_graph(X, labels, names, mode, panel_dir, keep):
    import subprocess, tempfile, shutil
    tmp = tempfile.mkdtemp()
    umi = np.load(f"{panel_dir}/panel.npz", allow_pickle=True)["umi"][keep]
    pd.DataFrame(X, columns=[str(n) for n in names]).to_csv(f"{tmp}/X.csv", index=False)
    np.savetxt(f"{tmp}/umi.txt", umi, fmt="%.0f")
    open(f"{tmp}/grp.txt", "w").write("\n".join(str(l) for l in labels))
    subprocess.run(["Rscript", "perturbaudit/wrappers/run_causalgrn.R", f"{tmp}/X.csv", f"{tmp}/umi.txt", f"{tmp}/grp.txt", f"{tmp}/A.tsv", mode], check=True)
    S = pd.read_csv(f"{tmp}/A.tsv", sep="\t", index_col=0)
    pos = {str(n): i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for r in S.index:
        for c in S.columns:
            W[pos[str(r)], pos[str(c)]] = S.at[r, c]
    shutil.rmtree(tmp)
    return W


if __name__ == "__main__":
    method, panel_dir = sys.argv[1], sys.argv[2]
    X, labels, names = load(panel_dir)
    t0 = time.time()
    if method in ("causalgrn", "gies"):
        Xs, ls, keep = subsample(X, labels, int(os.environ.get("SUB_PER_GROUP", "60")), int(os.environ.get("SUB_OBS", "3000")))
        W = _run_r_graph(Xs, ls, names, method, panel_dir, keep)
    else:
        W = {"tce": run_tce, "dotears": run_dotears, "sdcd": run_sdcd, "grnboost2": run_grnboost2, "inspre": run_inspre, "inspre_dag": run_inspre_dag, "adapre": run_adapre, "inspre_default": run_inspre_default, "gbm_grn": run_gbm_grn}[method](X, labels, names)
    W = np.nan_to_num(np.asarray(W, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    np.save(f"{panel_dir}/W_{method}.npy", W)
    json.dump({"method": method, "seconds": round(time.time() - t0, 1), "shape": list(W.shape)}, open(f"{panel_dir}/run_{method}.json", "w"))
    print(method, "done", round(time.time() - t0, 1), "s")

import numpy as np
import pandas as pd
import pytest
from perturbaudit import decoy, guidesplit, heldout, proximity, offtarget, sim


def test_target_decoy_fdr_exact():
    real = np.array([10, 9, 8, 1, 0.5])
    dec = np.array([9.5, 0.1, 0.2, 0.3])
    out = dict((t, (r, d, f)) for t, r, d, f in decoy.target_decoy_fdr(real, dec, [8.5]))
    r, d, f = out[8.5]
    assert r == 2 and d == 1
    assert f == pytest.approx(min(1.0, (2 / 4) * 5 / 2))


def test_qvalues_monotone_and_bounded():
    rng = np.random.default_rng(0)
    real = np.r_[rng.normal(0, 1, 500), rng.normal(6, 1, 50)]
    dec = rng.normal(0, 1, 2000)
    q = decoy.qvalues(np.abs(real), np.abs(dec))
    order = np.argsort(-np.abs(real))
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert q.min() >= 0 and q.max() <= 1


def test_null_var_z_zero_variance_group_not_extreme():
    ref_mean = np.array([0.1]); ref_var = np.array([0.05])
    z = decoy.null_var_z(np.array([130]), np.array([[0.0]]), ref_mean, ref_var, 5000)
    assert abs(z[0, 0]) < 6


def test_decoy_groups_batch_matched():
    gem = np.repeat(np.arange(4), 50)
    T = np.array([[3, 0, 2, 1], [1, 1, 1, 1]])
    M, pick = decoy.make_decoy_groups(gem, T, np.arange(4), n_decoys=2, seed=0)
    for k in range(2):
        members = M[k].indices
        counts = np.bincount(gem[members], minlength=4)
        assert np.array_equal(counts, T[pick[k]])


def test_heterogeneity_null_calibrated():
    rng = np.random.default_rng(1)
    z = rng.normal(3, 1, (200, 50))
    Q, p = guidesplit.heterogeneity(z + rng.normal(0, 1, z.shape), z + rng.normal(0, 1, z.shape))
    assert 0.03 < (p < 0.05).mean() < 0.07


def test_bh_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    q = guidesplit.bh(p)
    assert q[0] == pytest.approx(0.04) and q[3] == pytest.approx(0.5)


def test_descendants_chain():
    G = np.zeros((4, 4), bool); G[0, 1] = G[1, 2] = True
    D = heldout.descendants(G)
    assert D[0, 2] and D[0, 1] and not D[2, 0] and not D[3].any()


def test_implied_effect_scores_perfect_graph():
    G = np.zeros((3, 3), bool); G[0, 1] = True
    Z = np.zeros((3, 3)); Z[0, 1] = 10
    s = heldout.implied_effect_scores(G, Z)
    assert s["recall"] == 1.0 and s["violation_rate"] == 0.0


def test_proximity_flags():
    tss = {"A": ("chr1", 1000, "+"), "B": ("chr1", 1500, "-"), "C": ("chr2", 10, "+")}
    e = pd.DataFrame({"parent": ["A", "A"], "child": ["B", "C"]})
    f = proximity.flag_edges(e, tss)
    assert f.neighbour_flag.tolist() == [True, False]


def test_offtarget_flags():
    e = pd.DataFrame({"parent": ["P", "P", "Q"], "child": ["O", "X", "O"]})
    f = offtarget.flag_edges(e, {"P": {"O"}}, downstream={"O": {"X"}})
    assert f.offtarget_direct.tolist() == [True, False, False]
    assert f.offtarget_flag.tolist() == [True, True, False]


def test_knockoffs_preserve_correlation():
    rng = np.random.default_rng(2)
    C = np.array([[1, .7, .2], [.7, 1, .3], [.2, .3, 1]])
    X = rng.multivariate_normal(np.zeros(3), C, 20000)
    Xk = sim.gaussian_knockoffs(X, X, rng, shrink=0.0)
    assert np.allclose(np.corrcoef(Xk, rowvar=False), C, atol=0.05)
    cross = np.corrcoef(np.hstack([X, Xk]), rowvar=False)[:3, 3:]
    off = cross[~np.eye(3, dtype=bool)]
    assert np.allclose(off, C[~np.eye(3, dtype=bool)], atol=0.05)


def test_intervention_support_untestable():
    E = np.zeros((3, 3), bool); E[0, 1] = E[2, 1] = True
    Z = np.zeros((3, 3)); Z[1, 0] = 9
    u, r, ut = heldout.intervention_support(E, Z, low=3, high=5, kd_effective=[True, True, False], return_untestable=True)
    assert u[0, 1] and r[0, 1] and not u[2, 1] and ut[2, 1]


def test_preflight_flags_zero_variance():
    import numpy as np
    from perturbaudit.preflight import own_intervention_variance
    rng = np.random.default_rng(0)
    X = rng.normal(1, 1, (300, 2))
    labels = np.array(["non-targeting"] * 100 + ["A"] * 100 + ["B"] * 100)
    X[100:200, 0] = 0.0
    df = own_intervention_variance(X, labels, ["A", "B"])
    assert df.set_index("gene").loc["A", "degenerate"] and not df.set_index("gene").loc["B", "degenerate"]


def _sim_screen():
    import numpy as np
    from perturbaudit import sim
    s = sim.simulate_screen(p=30, density=0.1, n_obs=3000, n_per_pert=150, n_lat=3, lat_scale=1.0, noise_df=5, kd_mean=-2.0,
                            off_frac=0.1, off_strength=1.0, inactive_frac=0.1, misassign=0.0, seed=1)
    return s, np.asarray(s["X"]), np.asarray(s["labels"]).astype(str), [str(i) for i in range(s["X"].shape[1])]


def test_api_audit_separates_true_from_correlation_edges():
    import numpy as np
    from perturbaudit.screen import Screen
    from perturbaudit.api import audit_network
    s, X, labels, names = _sim_screen()
    sc = Screen(X, labels, names)
    C = np.abs(np.corrcoef(X[labels == "non-targeting"].T)); np.fill_diagonal(C, 0)
    E, S = audit_network(sc, C, names, top_k=60)
    W = np.abs(s["W"]) > 0
    E["true"] = [W[int(a), int(b)] for a, b in zip(E.parent, E.child)]
    pa, pu = E[E.verdict == "audited"].true.mean(), E[E.verdict == "unsupported"].true.mean()
    assert pa > 0.7 and pa - pu > 0.4
    assert set(S.columns) >= {"top_k", "audited", "unsupported"}


def test_decoy_z_is_null_calibrated():
    import numpy as np
    from perturbaudit.screen import Screen
    s, X, labels, names = _sim_screen()
    Zr, Zd = Screen(X, labels, names).decoys(n_decoys=100, seed=3)
    z = Zd.values.ravel()
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.25  # decoys share one reference half, so their mean carries its sampling error


def test_cli_roundtrip(tmp_path):
    import numpy as np, pandas as pd, anndata as ad
    from perturbaudit.cli import main
    s, X, labels, names = _sim_screen()
    a = ad.AnnData(X.astype(np.float32)); a.var_names = names; a.obs["gene"] = labels
    a.write_h5ad(tmp_path / "s.h5ad")
    W = np.abs(s["W"]); i, j = np.nonzero(W)
    pd.DataFrame(dict(parent=np.array(names)[i], child=np.array(names)[j], score=W[i, j])).to_csv(tmp_path / "n.csv", index=False)
    main(["audit", "--h5ad", str(tmp_path / "s.h5ad"), "--network", str(tmp_path / "n.csv"), "--perturbation-col", "gene", "--out", str(tmp_path / "r")])
    E = pd.read_csv(tmp_path / "r_edges.csv")
    assert len(E) == len(i) and "verdict" in E


def test_html_report(tmp_path):
    import numpy as np, pandas as pd, anndata as ad
    from perturbaudit.cli import main
    s, X, labels, names = _sim_screen()
    a = ad.AnnData(X.astype(np.float32)); a.var_names = names; a.obs["gene"] = labels
    a.write_h5ad(tmp_path / "s.h5ad")
    W = np.abs(s["W"]); i, j = np.nonzero(W)
    pd.DataFrame(dict(parent=np.array(names)[i], child=np.array(names)[j], score=W[i, j])).to_csv(tmp_path / "n.csv", index=False)
    main(["audit", "--h5ad", str(tmp_path / "s.h5ad"), "--network", str(tmp_path / "n.csv"), "--perturbation-col", "gene", "--html", "--out", str(tmp_path / "r")])
    h = open(tmp_path / "r_report.html", encoding="utf-8").read()
    assert h.startswith("<!DOCTYPE html>") and "<svg" in h and "audited" in h


def test_seedscan(tmp_path):
    import numpy as np, pandas as pd
    from perturbaudit import seedscan as ss
    rng = np.random.default_rng(0)
    bg = lambda n: "".join(rng.choice(list("ACT"), n))  # no G in background, so the only NGG PAMs are planted
    guide = "ACGTACGTACCTTACATCAA"
    prom = tmp_path / "p.tsv"
    with open(prom, "w") as f:
        f.write(f"GENEA\tchr1\t100\t+\t{bg(50)}{guide}AGG{bg(50)}\n")              # full 20-nt match
        f.write(f"GENEB\tchr2\t100\t+\t{bg(50)}AAAAAAAAACCTTACATCAA{'T'}GG{bg(50)}\n")  # 12-nt seed
        f.write(f"GENEC\tchr3\t100\t+\t{bg(120)}\n")                                # no PAM
    S = ss.build_sites(str(prom))
    R = ss.scan(pd.DataFrame(dict(guide=["g1"], target=["GENEA"], protospacer=[guide])), S, min_len=8)
    d = dict(zip(R.gene, R.seed_len))
    assert d.get("GENEA") == 20 and d.get("GENEB") == 12 and "GENEC" not in d
    m = ss.offtarget_map_from_guides(pd.DataFrame(dict(guide=["g1"], target=["GENEA"], protospacer=[guide])), S, min_len=12)
    assert m == {"GENEA": {"GENEB"}}


def test_audit_anndata_and_promoters(tmp_path):
    import numpy as np, pandas as pd, anndata as ad, gzip
    import perturbaudit as pa
    rng = np.random.default_rng(1); genes = [f"G{i}" for i in range(6)]
    lab = np.array(["non-targeting"] * 600 + ["G0"] * 150 + ["G1"] * 150)
    X = rng.normal(size=(900, 6)); X[lab == "G0", 0] -= 3; X[lab == "G0", 2] -= 1.5; X[lab == "G1", 1] -= 3
    A = ad.AnnData(X, obs=pd.DataFrame({"pert": lab}, index=[str(i) for i in range(900)]), var=pd.DataFrame(index=genes))
    E, S = pa.audit_anndata(A, pd.DataFrame({"parent": ["G0", "G1"], "child": ["G2", "G3"]}), "pert", n_decoys=20)
    v = dict(zip(E.parent + ">" + E.child, E.verdict))
    assert v["G0>G2"] == "audited" and v["G1>G3"] == "unsupported"
    fa = tmp_path / "g.fa"; fa.write_text(">chr1\n" + "A" * 3000 + "\n")
    gtf = tmp_path / "a.gtf"; gtf.write_text('chr1\tx\ttranscript\t1500\t1600\t.\t+\t.\tgene_name "GENEX";\n')
    n = pa.seedscan.promoter_windows(str(fa), str(gtf), str(tmp_path / "p.tsv"), window=100)
    row = open(tmp_path / "p.tsv").read().split("\t")
    assert n == 1 and row[0] == "GENEX" and len(row[4].strip()) == 201  # TSS +/- 100, inclusive

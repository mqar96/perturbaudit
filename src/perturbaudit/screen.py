"""Knockdown-effect statistics from a Perturb-seq AnnData, with batch-matched non-targeting references and decoys."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _indicator(codes, n_groups):
    n = len(codes)
    keep = codes >= 0
    return sp.csr_matrix((np.ones(keep.sum()), (codes[keep], np.flatnonzero(keep))), shape=(n_groups, n))


class Screen:
    """Group-level sufficient statistics of a single-guide Perturb-seq screen.

    Effects are null-variance z-scores: (group mean - batch-matched control mean) / SE, with the SE taken from
    control-cell variance. This keeps decoys (control cells relabelled as fake perturbations) well calibrated.
    """

    def __init__(self, X, perturbation, genes, control_label="non-targeting", batch=None, guide=None):
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        self.genes = np.asarray(genes).astype(str)
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        pert = np.asarray(perturbation).astype(str)
        self.control_label = control_label
        batch = np.zeros(len(pert), dtype=int) if batch is None else pd.factorize(np.asarray(batch))[0]
        self.n_batches = batch.max() + 1
        ctrl = pert == control_label
        if ctrl.sum() < 20:
            raise ValueError("fewer than 20 control cells with label %r" % control_label)
        Ib = _indicator(np.where(ctrl, batch, -1), self.n_batches)
        self.c_n = np.asarray(Ib.sum(1)).ravel()
        self.c_s1 = np.asarray((Ib @ X).toarray())
        X2 = X.multiply(X).tocsr()
        self.c_s2 = np.asarray((Ib @ X2).toarray())
        self.c_mu = self.c_s1 / np.maximum(self.c_n[:, None], 1)
        self.c_var = np.maximum(self.c_s2 / np.maximum(self.c_n[:, None], 1) - self.c_mu ** 2, 1e-12)
        self._X, self._batch, self._ctrl = X, batch, ctrl
        self.groups = {}
        for key, labels in [("target", pert)] + ([("guide", np.asarray(guide).astype(str))] if guide is not None else []):
            lab = np.where(ctrl, "", labels)
            names, codes = np.unique(lab, return_inverse=True)
            codes = np.where(ctrl, -1, codes)
            self.groups[key] = self._group_stats(X, codes, names, batch)
        if guide is not None:
            g2t = pd.Series(pert[~ctrl], index=np.asarray(guide).astype(str)[~ctrl])
            self.guide_target = g2t[~g2t.index.duplicated()].to_dict()

    def _group_stats(self, X, codes, names, batch):
        G = len(names)
        I = _indicator(codes, G)
        n = np.asarray(I.sum(1)).ravel()
        s1 = np.asarray((I @ X).toarray()) if sp.issparse(X) else I @ X
        nb = np.zeros((G, self.n_batches))
        np.add.at(nb, (codes[codes >= 0], batch[codes >= 0]), 1)
        keep = n > 0
        return dict(names=names[keep], n=n[keep], s1=s1[keep], nb=nb[keep])

    def _z(self, n, s1, nb):
        w = nb / np.maximum(n[:, None], 1)
        ref = w @ self.c_mu
        var_cell = w @ self.c_var
        var_ref = (w ** 2) @ (self.c_var / np.maximum(self.c_n[:, None], 1))
        return (s1 / np.maximum(n[:, None], 1) - ref) / np.sqrt(var_cell / np.maximum(n[:, None], 1) + var_ref)

    def effects(self, by="target", min_cells=5):
        g = self.groups[by]
        ok = g["n"] >= min_cells
        Z = self._z(g["n"][ok], g["s1"][ok], g["nb"][ok])
        return pd.DataFrame(Z, index=g["names"][ok], columns=self.genes)

    def knockdown(self, by="target", min_cells=5):
        """z of each perturbed gene in its own cells (negative = knocked down); NaN if the gene is not measured."""
        Z = self.effects(by, min_cells)
        tgt = [self.guide_target.get(i, i) for i in Z.index] if by == "guide" else list(Z.index)
        out = [Z.iat[r, self.gene_index[t]] if t in self.gene_index else np.nan for r, t in enumerate(tgt)]
        return pd.Series(out, index=Z.index)

    def decoys(self, n_decoys=200, cells_per_decoy=None, seed=0, by="target", min_cells=5):
        """Decoy effects: control cells split into a reference half and decoy groups matched to the
        perturbations' cell numbers and batch composition. Returns (Z_real, Z_decoy) computed against the
        same reference half, so decoy z-scores share the real ones' null distribution."""
        rng = np.random.default_rng(seed)
        g = self.groups[by]
        ok = g["n"] >= min_cells
        ctrl_idx = np.flatnonzero(self._ctrl)
        perm = rng.permutation(ctrl_idx)
        ref, pool = np.sort(perm[: len(perm) // 2]), perm[len(perm) // 2:]
        X, b = self._X, self._batch
        Ib = _indicator(np.where(np.isin(np.arange(X.shape[0]), ref), b, -1), self.n_batches)
        c_n = np.asarray(Ib.sum(1)).ravel()
        c_mu = np.asarray((Ib @ X).toarray()) / np.maximum(c_n[:, None], 1)
        c_var = np.maximum(np.asarray((Ib @ X.multiply(X).tocsr()).toarray()) / np.maximum(c_n[:, None], 1) - c_mu ** 2, 1e-12)
        saved = (self.c_n, self.c_mu, self.c_var)
        self.c_n, self.c_mu, self.c_var = c_n, c_mu, c_var
        try:
            Zr = pd.DataFrame(self._z(g["n"][ok], g["s1"][ok], g["nb"][ok]), index=g["names"][ok], columns=self.genes)
            sizes = rng.choice(g["n"][ok], n_decoys) if cells_per_decoy is None else np.full(n_decoys, cells_per_decoy)
            sizes = np.minimum(sizes, max(len(pool) // 4, 5)).astype(int)
            pool_b = b[pool]
            rows, nbs, ns = [], [], []
            for s in sizes:
                pick = rng.choice(len(pool), s, replace=False)
                cells = pool[pick]
                rows.append(np.asarray(X[cells].sum(0)).ravel())
                nb = np.bincount(pool_b[pick], minlength=self.n_batches)
                nbs.append(nb); ns.append(s)
            Zd = self._z(np.array(ns), np.vstack(rows), np.vstack(nbs).astype(float))
            Zd = pd.DataFrame(Zd, index=[f"DECOY{i:04d}" for i in range(n_decoys)], columns=self.genes)
        finally:
            self.c_n, self.c_mu, self.c_var = saved
        return Zr, Zd

    @classmethod
    def from_anndata(cls, adata, perturbation_col, control_label="non-targeting", batch_col=None, guide_col=None,
                     layer=None, gene_col=None):
        X = adata.layers[layer] if layer else adata.X
        genes = adata.var[gene_col].values if gene_col else adata.var_names.values
        return cls(X, adata.obs[perturbation_col].values, genes, control_label,
                   adata.obs[batch_col].values if batch_col else None,
                   adata.obs[guide_col].values if guide_col else None)

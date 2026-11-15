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

    def __init__(self, X, perturbation, genes, control_label="non-targeting", batch=None, guide=None,
                 min_ref_cells=20):
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        self.genes = np.asarray(genes).astype(str)
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        pert = np.asarray(perturbation).astype(str)
        self.control_label = control_label
        self.min_ref_cells = int(min_ref_cells)
        if batch is None:
            batch = np.zeros(len(pert), dtype=int)
            self.batch_names = np.array(["all"])
        else:
            batch = np.asarray(batch)
            if pd.isna(batch).any():
                raise ValueError("batch labels contain missing values; fill or drop those cells first")
            batch, self.batch_names = pd.factorize(batch)
            self.batch_names = np.asarray(self.batch_names).astype(str)
        self.n_batches = batch.max() + 1
        ctrl = pert == control_label
        if ctrl.sum() < self.min_ref_cells:
            raise ValueError("fewer than %d control cells with label %r" % (self.min_ref_cells, control_label))
        self.c_n, self.c_mu, self.c_var, self.ref_invalid = self._reference(X, np.flatnonzero(ctrl), batch)
        self._X, self._batch, self._ctrl = X, batch, ctrl
        # perturbed cells in batches without a valid reference cannot be compared to batch-matched controls:
        # they are excluded (and counted), never compared against an empty reference
        bad_batch = self.c_n < self.min_ref_cells
        self.excluded = (~ctrl) & bad_batch[batch]
        self.groups = {}
        self.excluded_cells = {}
        for key, labels in [("target", pert)] + ([("guide", np.asarray(guide).astype(str))] if guide is not None else []):
            lab = np.where(ctrl, "", labels)
            names, codes = np.unique(lab, return_inverse=True)
            codes = np.where(ctrl | self.excluded, -1, codes)
            self.groups[key] = self._group_stats(X, codes, names, batch)
            exc = pd.Series(np.asarray(labels).astype(str)[self.excluded]).value_counts()
            self.excluded_cells[key] = exc.to_dict()
        tot = pd.Series(pert[~ctrl]).value_counts()
        kept = pd.Series(self.groups["target"]["n"], index=self.groups["target"]["names"])
        self.no_reference_groups = set(tot.index) - set(kept[kept > 0].index)
        if guide is not None:
            g2t = pd.Series(pert[~ctrl], index=np.asarray(guide).astype(str)[~ctrl])
            self.guide_target = g2t[~g2t.index.duplicated()].to_dict()

    def _reference(self, X, ref_cells, batch):
        """Per-batch control mean/variance. ref_invalid[b, j] is True when batch b has fewer than
        min_ref_cells controls or gene j is constant in its controls; z-scores touching such strata are NaN."""
        codes = np.full(X.shape[0], -1)
        codes[ref_cells] = batch[ref_cells]
        Ib = _indicator(codes, self.n_batches)
        c_n = np.asarray(Ib.sum(1)).ravel()
        s1 = np.asarray((Ib @ X).toarray())
        s2 = np.asarray((Ib @ X.multiply(X).tocsr()).toarray())
        c_mu = s1 / np.maximum(c_n[:, None], 1)
        raw_var = s2 / np.maximum(c_n[:, None], 1) - c_mu ** 2
        invalid = (c_n[:, None] < self.min_ref_cells) | (raw_var <= 1e-12)
        return c_n, c_mu, np.where(invalid, 1.0, raw_var), invalid

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
        Z = (s1 / np.maximum(n[:, None], 1) - ref) / np.sqrt(var_cell / np.maximum(n[:, None], 1) + var_ref)
        bad = ((w > 0).astype(float) @ self.ref_invalid.astype(float)) > 0
        Z[bad] = np.nan
        return Z

    def reference_coverage(self, by="target", min_cells=5):
        """Per-batch control counts and, per perturbation group, how many of its cells were excluded because they sit
        in batches without a valid reference (fewer than min_ref_cells controls). Check this before trusting verdicts."""
        g = self.groups[by]
        bad_b = self.c_n < self.min_ref_cells
        pert_cells = np.bincount(self._batch[~self._ctrl], minlength=self.n_batches)
        batches = pd.DataFrame(dict(batch=self.batch_names, control_cells=self.c_n, perturbed_cells=pert_cells,
                                    reference_ok=~bad_b,
                                    constant_genes=np.where(bad_b, 0, self.ref_invalid.sum(1))))
        exc = self.excluded_cells[by]
        names = sorted(set(g["names"]) | set(exc))
        kept = dict(zip(g["names"], g["n"]))
        groups = pd.DataFrame(dict(group=names, cells_used=[int(kept.get(n, 0)) for n in names],
                                   cells_without_reference=[int(exc.get(n, 0)) for n in names]))
        groups["testable"] = groups.cells_used >= min_cells
        return batches, groups

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
        """Batch-matched decoys. Control cells are split within each batch into a reference half and a decoy
        pool. Each decoy copies the cell number and batch composition of a randomly chosen real perturbation
        group (scaled down if the pool is too small) and draws that many pool cells from each of its batches.
        Real and decoy z-scores are both computed against the reference half, so they share one null.
        Returns (Z_real, Z_decoy); self.decoy_info records the templates and any that could not be matched."""
        rng = np.random.default_rng(seed)
        g = self.groups[by]
        ok = g["n"] >= min_cells
        b = self._batch
        ref, pool = [], []
        for k in range(self.n_batches):
            cells = rng.permutation(np.flatnonzero(self._ctrl & (b == k)))
            ref.append(cells[: len(cells) // 2]); pool.append(cells[len(cells) // 2:])
        ref_cells = np.concatenate(ref)
        X = self._X
        saved = (self.c_n, self.c_mu, self.c_var, self.ref_invalid, self.min_ref_cells)
        self.min_ref_cells = max(int(np.ceil(self.min_ref_cells / 2)), 2)  # the reference half holds half the controls
        self.c_n, self.c_mu, self.c_var, self.ref_invalid = self._reference(X, ref_cells, b)
        try:
            Zr = pd.DataFrame(self._z(g["n"][ok], g["s1"][ok], g["nb"][ok]), index=g["names"][ok], columns=self.genes)
            nb_real = g["nb"][ok]
            pool_n = np.array([len(p) for p in pool])
            rows, nbs, ns, templ, skipped = [], [], [], [], 0
            tries = 0
            while len(rows) < n_decoys and tries < 20 * n_decoys:
                tries += 1
                t = rng.integers(len(nb_real))
                want = nb_real[t].copy() if cells_per_decoy is None else np.round(
                    nb_real[t] / nb_real[t].sum() * cells_per_decoy)
                cap = np.floor(pool_n / 4)
                scale = np.min(np.where(want > 0, cap / np.maximum(want, 1), np.inf))
                if scale < 1:
                    want = np.floor(want * scale)
                if want.sum() < min_cells or np.any((want > 0) & (pool_n == 0)):
                    skipped += 1
                    continue
                cells = np.concatenate([rng.choice(pool[k], int(m), replace=False) for k, m in enumerate(want) if m > 0])
                rows.append(np.asarray(X[cells].sum(0)).ravel()); nbs.append(want); ns.append(want.sum())
                templ.append(str(g["names"][ok][t]))
            if not rows:
                raise ValueError("no decoy could be batch-matched: too few control cells in the perturbed batches")
            Zd = self._z(np.array(ns, dtype=float), np.vstack(rows), np.vstack(nbs).astype(float))
            Zd = pd.DataFrame(Zd, index=[f"DECOY{i:04d}" for i in range(len(rows))], columns=self.genes)
            self.decoy_info = dict(templates=templ, skipped_templates=skipped, n_decoys=len(rows))
        finally:
            self.c_n, self.c_mu, self.c_var, self.ref_invalid, self.min_ref_cells = saved
        return Zr, Zd

    @classmethod
    def from_anndata(cls, adata, perturbation_col, control_label="non-targeting", batch_col=None, guide_col=None,
                     layer=None, gene_col=None, min_ref_cells=20):
        X = adata.layers[layer] if layer else adata.X
        genes = adata.var[gene_col].values if gene_col else adata.var_names.values
        return cls(X, adata.obs[perturbation_col].values, genes, control_label,
                   adata.obs[batch_col].values if batch_col else None,
                   adata.obs[guide_col].values if guide_col else None, min_ref_cells=min_ref_cells)

"""Pre-flight checks on a perturbation panel before running network methods."""
import numpy as np
import pandas as pd


def own_intervention_variance(X, labels, names, control="non-targeting", ratio_thr=0.05):
    """For each gene, compare its variance in the cells where it is perturbed with its variance in control cells.

    Variance-weighted methods (e.g. dotears) invert the own-intervention variance; near-zero values
    (complete knockdown, zero-inflated counts, or badly built decoys) make their loss degenerate.
    Returns a DataFrame with n_cells, var_own, var_ctrl, ratio, frac_zero and a `degenerate` flag.
    """
    X = np.asarray(X); labels = np.asarray(labels).astype(str)
    ctrl = labels == control
    rows = []
    for i, g in enumerate(names):
        m = labels == str(g)
        if not m.any():
            continue
        vo = float(X[m, i].var()); vc = float(X[ctrl, i].var()) if ctrl.any() else np.nan
        r = vo / vc if vc and vc > 0 else np.nan
        rows.append(dict(gene=str(g), n_cells=int(m.sum()), var_own=vo, var_ctrl=vc, ratio=r,
                         frac_zero=float((X[m, i] == 0).mean()), degenerate=bool(vo == 0 or (r == r and r < ratio_thr))))
    return pd.DataFrame(rows)

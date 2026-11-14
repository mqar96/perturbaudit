"""perturbaudit: internal-control falsification of causal gene networks inferred from Perturb-seq."""
__version__ = "0.2.0"
from .screen import Screen
from .api import audit_network, summarize
from . import seedscan, decoy, proximity, report


def audit_anndata(adata, network, perturbation_key, control="non-targeting", names=None, batch_key=None, guide_key=None,
                  layer=None, gene_key=None, **kwargs):
    """Audit a network against a screen stored as AnnData (scverse / pertpy layout: cells x genes, perturbation in obs).
    network: DataFrame with parent, child[, score] columns, or a square matrix with `names`. Extra kwargs go to audit_network
    (top_k, support_z, n_decoys, tss, offtarget_map, replicate_networks, ...). Returns (edges, summary)."""
    sc = Screen.from_anndata(adata, perturbation_key, control, batch_key, guide_key, layer, gene_key)
    return audit_network(sc, network, names, **kwargs)

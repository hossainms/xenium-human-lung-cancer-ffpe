"""Spatial domains (BANKSY, CellCharter), domain naming and comparison, spillover-aware paired pseudobulk DE (notebook 04)."""

from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

COMPARTMENT = {"Epithelial": "tumour", "T / NK": "immune", "B / plasma": "immune", "Myeloid": "immune", "Mast": "immune",
               "Stromal": "stroma / vessels", "Endothelial": "stroma / vessels"}


def banksy_grid(adata, lambdas, resolutions, k: int = 15, n_pcs: int = 20, seed: int = 0) -> dict:
    """BANKSY (own + neighbour-averaged expression, weighted by lambda) -> PCA -> Leiden, for every (lambda, resolution)."""
    import scanpy as sc
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.initialize_banksy import initialize_banksy

    xy = adata.obsm["spatial"]
    adata.obs["x"], adata.obs["y"] = xy[:, 0], xy[:, 1]
    adata.obsm["coord_xy"] = xy
    with contextlib.redirect_stdout(io.StringIO()):   # pybanksy prints its progress
        bdict = initialize_banksy(adata, ("x", "y", "coord_xy"), num_neighbours=k, max_m=0,
                                  plt_edge_hist=False, plt_nbr_weights=False, plt_theta=False)
        bdict, _ = generate_banksy_matrix(adata, bdict, list(lambdas), max_m=0, verbose=False)
    grid = {}
    for lam in lambdas:
        bm = bdict["scaled_gaussian"][lam]["adata"]
        sc.pp.pca(bm, n_comps=n_pcs, random_state=seed)
        sc.pp.neighbors(bm, n_neighbors=15, use_rep="X_pca", random_state=seed)
        for res in resolutions:
            sc.tl.leiden(bm, resolution=res, flavor="igraph", n_iterations=2, directed=False, random_state=seed, key_added="leiden")
            grid[(lam, res)] = bm.obs["leiden"].to_numpy()
    return grid


def grid_stability(grid: dict, chosen, lambdas, resolutions):
    """ARI of every setting with the chosen one, and the number of domains per setting."""
    from sklearn.metrics import adjusted_rand_score as ari

    ref = grid[chosen]
    idx = [f"lambda {l}" for l in lambdas]
    cols = [f"res {r}" for r in resolutions]
    stab = pd.DataFrame([[ari(ref, grid[(l, r)]) for r in resolutions] for l in lambdas], index=idx, columns=cols)
    n = pd.DataFrame([[len(set(grid[(l, r)])) for r in resolutions] for l in lambdas], index=idx, columns=cols)
    return stab, n


def quiet_lightning() -> None:
    """Lightning (CellCharter's GMM trainer) sets its own log levels on import, so silence it afterwards."""
    for name in ["pytorch_lightning", "lightning", "lightning.pytorch", "lightning.fabric", "torchgmm"]:
        logging.getLogger(name).setLevel(logging.ERROR)


def cellcharter_embedding(adata, n_pcs: int = 30, n_layers: int = 3, seed: int = 0) -> None:
    """obsm['X_cellcharter']: PCA of scaled expression, concatenated over 3 layers of the Delaunay graph."""
    import cellcharter as cc
    import scanpy as sc
    import squidpy as sq

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, key_added="cc")
    cc.gr.remove_long_links(adata, connectivity_key="cc_connectivities", distances_key="cc_distances")
    adata.obsm["X_pca30"] = sc.pp.pca(sc.pp.scale(adata.layers["lognorm"], max_value=10, copy=True), n_comps=n_pcs, random_state=seed)
    cc.gr.aggregate_neighbors(adata, n_layers=n_layers, use_rep="X_pca30", connectivity_key="cc_connectivities", out_key="X_cellcharter")


def cellcharter_autok(adata, k_range=(4, 12), runs: int = 3, seed: int = 0):
    """Gaussian mixtures for K in range, `runs` times each; the most stable K wins.

    CellCharter's random_state does not seed PyTorch, so the global seed is set first (runs are then identical).
    CPU is used: faster than the Mac GPU for this model. Returns (labels, mean stability per K).
    """
    import cellcharter as cc
    import pytorch_lightning as pl

    quiet_lightning()
    pl.seed_everything(seed, verbose=False)
    autok = cc.tl.ClusterAutoK(n_clusters=tuple(k_range), max_runs=runs, model_params={
        "random_state": seed, "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False}})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        autok.fit(adata, use_rep="X_cellcharter")
        labels = np.asarray(autok.predict(adata, use_rep="X_cellcharter")).astype(str)
    stability = pd.Series(np.asarray(autok.stability).mean(axis=1), index=autok.n_clusters[1:-1], name="stability")   # one row per K
    return labels, stability


def name_domains(labels: pd.Series, lineage: pd.Series, lineages) -> pd.Series:
    """'D<i>: <top lineage> x% [+ <second> y%]', numbered by decreasing epithelial share."""
    comp = pd.crosstab(labels, lineage, normalize="index").reindex(columns=lineages).fillna(0) * 100
    names = {}
    for d, row in comp.iterrows():
        top = row.sort_values(ascending=False)
        names[d] = f"{top.index[0]} {top.iloc[0]:.0f}%" + (f" + {top.index[1]} {top.iloc[1]:.0f}%" if top.iloc[1] >= 15 else "")
    prefix = {d: f"D{i + 1}" for i, d in enumerate(comp.sort_values("Epithelial", ascending=False).index)}
    return labels.map(lambda d: f"{prefix[d]}: {names[d]}")


def compartment(labels: pd.Series, lineage: pd.Series) -> pd.Series:
    """Each domain's majority tissue compartment (tumour / immune / stroma + vessels), per cell."""
    return labels.map(pd.crosstab(labels, lineage.map(COMPARTMENT), normalize="index").idxmax(axis=1))


def foreign_genes(adata, lineage: pd.Series, own: str = "Myeloid", fold: float = 4) -> pd.Series:
    """Genes whose mean expression in another lineage is >= fold x the own-lineage mean (likely spillover)."""
    norm = adata.layers["lognorm"]
    lins = sorted(lineage.unique())
    mean = pd.DataFrame({l: np.asarray(np.expm1(norm[(lineage == l).to_numpy()]).mean(axis=0)).ravel() for l in lins}, index=adata.var_names)
    other = mean.drop(columns=own)
    return other.idxmax(axis=1).where(other.max(axis=1) >= fold * mean[own].clip(lower=1e-3))


def blocks(xy: np.ndarray, block_um: float, index) -> pd.Series:
    """Spatial pseudo-replicate id per cell: the block_um x block_um square it falls in."""
    return pd.Series([f"b{int(x // block_um)}_{int(y // block_um)}" for x, y in xy], index=index)


def pseudobulk_de(counts, genes, cells, block: pd.Series, domain: pd.Series, dom_a: str, dom_b: str, min_cells: int = 10,
                  min_blocks: int = 5, n_cpus: int = 8):
    """Paired pseudobulk DESeq2 (PyDESeq2): cells summed per (block, domain), only blocks holding both domains,
    design ~ block + domain. Returns (results: baseMean, log2FoldChange [a vs b], padj; {"blocks", "cells"} used).
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    X = sp.csr_matrix(counts)
    key = (block[cells] + "|" + domain[cells]).to_numpy()
    groups = pd.Series(key).value_counts()
    groups = groups[groups >= min_cells].index
    meta = pd.DataFrame([g.split("|") for g in groups], columns=["block", "domain"], index=groups)
    paired = meta.groupby("block")["domain"].nunique().pipe(lambda s: s[s == 2].index)
    meta = meta[meta["block"].isin(paired)]
    if len(paired) < min_blocks:
        raise ValueError(f"only {len(paired)} blocks with >= {min_cells} cells in both domains; too few to test")
    bulk = pd.DataFrame(np.vstack([np.asarray(X[key == g].sum(axis=0)).ravel() for g in meta.index]).astype(int), index=meta.index, columns=genes)
    bulk = bulk.loc[:, bulk.sum() >= 10]
    meta["domain"] = pd.Categorical(np.where(meta["domain"] == dom_a, "A", "B"), categories=["B", "A"])
    dds = DeseqDataSet(counts=bulk, metadata=meta, design="~ block + domain", inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["domain", "A", "B"], quiet=True)
    with contextlib.redirect_stdout(io.StringIO()):
        stats.summary()
    info = {"blocks": len(paired), "cells": int(sum((key == g).sum() for g in meta.index))}
    return stats.results_df[["baseMean", "log2FoldChange", "padj"]], info


def call_genes(r10: pd.DataFrame, rP: pd.DataFrame, foreign: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Join 10x and Proseg results; a gene is 'macrophage-intrinsic' only if significant, not foreign, and
    significant in the same direction with Proseg counts."""
    r = r10.join(rP[["log2FoldChange", "padj"]], rsuffix="_proseg")
    r["foreign (lineage)"] = foreign.reindex(r.index)
    sig = r["padj"] < alpha
    same = (np.sign(r["log2FoldChange"]) == np.sign(r["log2FoldChange_proseg"])) & (r["padj_proseg"] < alpha)
    r["call"] = np.select([~sig, r["foreign (lineage)"].notna(), ~same], ["n.s.", "spillover (foreign gene)", "not confirmed with Proseg"],
                          "macrophage-intrinsic")
    return r


def explorer_groups(labels: pd.Series, cell_ids, colors: dict | None = None) -> pd.DataFrame:
    """A Xenium Explorer cell-groups table (cell_id, group, color)."""
    import matplotlib.pyplot as plt

    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = {c: "#%02x%02x%02x" % tuple(int(255 * v) for v in cmap(i % 20)[:3]) for i, c in enumerate(sorted(labels.unique()))}
    return pd.DataFrame({"cell_id": cell_ids, "group": labels.to_numpy(), "color": labels.map(colors).to_numpy()})

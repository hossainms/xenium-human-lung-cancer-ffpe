"""Normalisation, PCA, neighbour graph, Leiden clustering, UMAP, marker genes (Step 5)."""

from __future__ import annotations

import pandas as pd


def normalise_and_pca(adata, n_comps: int = 50) -> None:
    """Counts -> median library size -> log1p (layers['lognorm']); scaled copy for PCA only; X = lognorm.

    No highly-variable-gene selection: all panel genes were chosen by design.
    """
    import scanpy as sc

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=n_comps, svd_solver="arpack")
    adata.X = adata.layers["lognorm"]


def cluster(adata, n_pcs: int = 30, n_neighbors: int = 15, resolutions=(0.3, 0.5, 1.0), seed: int = 0, umap: bool = True) -> None:
    """kNN graph in PCA space, Leiden at each resolution (obs['leiden_<r>']), and a UMAP for viewing."""
    import scanpy as sc

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    for r in resolutions:
        sc.tl.leiden(adata, resolution=r, key_added=f"leiden_{r}", flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    if umap:
        sc.tl.umap(adata, random_state=seed)


def benchmark_vs_tenx(adata, tenx: pd.Series, resolutions) -> pd.DataFrame:
    """Clusters per resolution and ARI / NMI against a vendor clustering (e.g. io.xenium.vendor_clusters); obs['tenx_graphclust']."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    shared = adata.obs_names.intersection(tenx.index)
    adata.obs["tenx_graphclust"] = pd.Categorical(tenx.reindex(adata.obs_names).astype("Int64").astype(str))
    return pd.DataFrame({
        "clusters": [adata.obs[f"leiden_{r}"].nunique() for r in resolutions],
        "ARI vs 10x": [adjusted_rand_score(tenx[shared], adata.obs.loc[shared, f"leiden_{r}"]) for r in resolutions],
        "NMI vs 10x": [normalized_mutual_info_score(tenx[shared], adata.obs.loc[shared, f"leiden_{r}"]) for r in resolutions],
    }, index=pd.Index(list(resolutions), name="resolution"))


def marker_table(adata, key: str, n: int = 6) -> pd.DataFrame:
    """Wilcoxon markers per cluster (stored in uns['markers']); top n with adjusted p < 0.01 and log2FC > 1."""
    import scanpy as sc

    sc.tl.rank_genes_groups(adata, key, method="wilcoxon", use_raw=False, layer="lognorm", key_added="markers")
    markers = sc.get.rank_genes_groups_df(adata, group=None, key="markers")
    top = markers[(markers.pvals_adj < 0.01) & (markers.logfoldchanges > 1)].groupby("group", observed=True).head(n)
    table = top.groupby("group", observed=True)["names"].apply(", ".join).rename(f"top {n} markers").to_frame()
    table.insert(0, "cells", adata.obs[key].value_counts().reindex(table.index).to_numpy())
    return table

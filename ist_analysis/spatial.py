"""Spatial organisation: neighbour graph, neighbourhood enrichment, niches, distance to tumour, TLS (Step 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import TUMOUR_TYPES

TIME_TYPES = ["Tumour epithelial (MALL/TCIM)", "Tumour epithelial (CYP2B6/CFTR)", "Tumour epithelial (MYC/CAPN8)",
              "Proliferating tumour", "CD8 T (GZMK+)", "CD4 T", "Treg", "NK", "B cell", "Plasma cell",
              "CD163+ macrophage", "FCGR1A+ macrophage", "CXCL9+ macrophage", "Alveolar macrophage",
              "cDC2", "mregDC (LAMP3+)", "Fibroblast", "Endothelial"]
IMMUNE_TYPES = ["CD8 T (GZMK+)", "CD4 T", "Treg", "NK", "B cell", "Plasma cell", "CD163+ macrophage", "FCGR1A+ macrophage",
                "CXCL9+ macrophage", "Alveolar macrophage", "LYVE1+ macrophage", "cDC1 (IRF8+)", "cDC2", "mregDC (LAMP3+)"]


def high_confidence(adata):
    """Cells annotated with high confidence; 'mixed' (spillover) cells would blur contact statistics."""
    return adata[adata.obs["annotation_confidence"] == "high"].copy()


def delaunay_graph(adata, max_edge_um: float = 30) -> dict:
    """Delaunay neighbour graph (squidpy) with edges longer than max_edge_um removed, in place. Returns stats."""
    import squidpy as sq

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    conn = adata.obsp["spatial_connectivities"].tocsr()
    dist = adata.obsp["spatial_distances"].tocsr()
    stats = {"edges": dist.nnz, "median_length_um": float(np.median(dist.data))}
    long_edges = dist.data > max_edge_um
    stats["frac_removed"] = float(long_edges.mean())
    conn.data[long_edges] = 0
    dist.data[long_edges] = 0
    conn.eliminate_zeros()
    dist.eliminate_zeros()
    adata.obsp["spatial_connectivities"], adata.obsp["spatial_distances"] = conn, dist
    degree = conn.getnnz(axis=1)
    stats.update(median_degree=float(np.median(degree)), mean_degree=float(degree.mean()), isolated=int((degree == 0).sum()))
    return stats


def add_time_types(adata, types=TIME_TYPES) -> list[str]:
    """obs['time_type']: the tumour-immune cell types present, everything else 'other'. Returns the types used."""
    present = [t for t in types if t in set(adata.obs["cell_type"])]
    adata.obs["time_type"] = pd.Categorical(adata.obs["cell_type"].astype(str).where(adata.obs["cell_type"].isin(present), "other"))
    return present


def nhood_zscores(adata, key: str, n_perms: int = 1000, order=None, keep_uns: bool = False) -> pd.DataFrame:
    """Neighbourhood enrichment z-scores (squidpy, label permutations)."""
    import squidpy as sq

    sq.gr.nhood_enrichment(adata, cluster_key=key, n_perms=n_perms, seed=0, show_progress_bar=False)
    cats = adata.obs[key].cat.categories
    z = pd.DataFrame(adata.uns[f"{key}_nhood_enrichment"]["zscore"], index=cats, columns=cats)
    if not keep_uns:
        adata.uns.pop(f"{key}_nhood_enrichment", None)
    return z.loc[order, order] if order is not None else z


def top_pairs(z: pd.DataFrame, n: int = 8, largest: bool = True) -> pd.DataFrame:
    """Strongest off-diagonal pairs of a symmetric z-score matrix."""
    pairs = (z.where(~np.eye(len(z), dtype=bool)).stack().rename("z").reset_index()
             .rename(columns={"level_0": "cell type A", "level_1": "cell type B"}))
    pairs = pairs[pairs["cell type A"] < pairs["cell type B"]]
    return (pairs.nlargest(n, "z") if largest else pairs.nsmallest(n, "z")).reset_index(drop=True)


def niche_name(row: pd.Series) -> str:
    """Name by dominant lineages, e.g. 'Epithelial 91%' or 'T / NK 46% + B / plasma 14%'."""
    top = row.sort_values(ascending=False)
    if top.iloc[0] >= 0.85:
        return f"{top.index[0]} {top.iloc[0]:.0%}"
    return f"{top.index[0]} {top.iloc[0]:.0%} + {top.index[1]} {top.iloc[1]:.0%}"


def composition_niches(adata, radius_um: float = 50, n_niches: int = 10, seed: int = 0):
    """Cellular niches: k-means on each cell's lineage mix within radius_um (Schurch et al. 2020).

    Sets obs['niche'] (named N1.. by decreasing epithelial share). Returns (centres table, median neighbours).
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import MiniBatchKMeans

    xy = adata.obsm["spatial"]
    lineages = list(adata.obs["lineage"].cat.categories)
    codes = adata.obs["lineage"].cat.codes.to_numpy()
    neigh = cKDTree(xy).query_ball_point(xy, r=radius_um)
    comp = np.zeros((adata.n_obs, len(lineages)))
    for i, nb in enumerate(neigh):
        comp[i] = np.bincount(codes[nb], minlength=len(lineages))
    comp /= comp.sum(axis=1, keepdims=True)
    km = MiniBatchKMeans(n_clusters=n_niches, random_state=seed, n_init=10, batch_size=8192).fit(comp)
    centres = pd.DataFrame(km.cluster_centers_, columns=lineages)
    order = centres["Epithelial"].sort_values(ascending=False).index
    names = {old: f"N{new + 1}: {niche_name(centres.loc[old])}" for new, old in enumerate(order)}
    adata.obs["niche"] = pd.Categorical([names[l] for l in km.labels_], categories=[names[o] for o in order])
    centres = centres.loc[order]
    centres.index = [names[o] for o in order]
    centres["cells"] = adata.obs["niche"].value_counts().reindex(centres.index).to_numpy()
    return centres, float(np.median([len(n) for n in neigh]))


def distance_to_tumour(adata, tumour_types=TUMOUR_TYPES) -> np.ndarray:
    """obs['dist_to_tumour_um']: distance from every cell to the nearest tumour cell (0 for tumour cells)."""
    from scipy.spatial import cKDTree

    xy = adata.obsm["spatial"]
    is_tumour = adata.obs["cell_type"].isin(tumour_types).to_numpy()
    d, _ = cKDTree(xy[is_tumour]).query(xy, k=1)
    adata.obs["dist_to_tumour_um"] = d
    return is_tumour


def infiltration_table(adata, is_tumour: np.ndarray, contact_um: float = 15, types=None):
    """Per cell type: median distance to tumour and % within contact_um, against the non-tumour baseline."""
    d = adata.obs["dist_to_tumour_um"].to_numpy()
    base_median = float(np.median(d[~is_tumour]))
    base_contact = float((d[~is_tumour] <= contact_um).mean())
    types = (types or IMMUNE_TYPES) + ["Fibroblast"]
    g = adata.obs[adata.obs["cell_type"].isin(types)].groupby("cell_type", observed=True)["dist_to_tumour_um"]
    col = f"% within {contact_um} um"
    infil = pd.DataFrame({"cells": g.size(), "median distance (um)": g.median(), col: 100 * g.apply(lambda x: (x <= contact_um).mean()),
                          "% beyond 100 um": 100 * g.apply(lambda x: (x > 100).mean())})
    infil["vs. baseline"] = np.where(infil[col] > 100 * base_contact * 1.2, "infiltrating",
                                     np.where(infil[col] < 100 * base_contact * 0.8, "excluded", "similar"))
    return infil.sort_values("median distance (um)"), base_median, base_contact


def find_tls(adata, eps_um: float = 25, min_cells: int = 20, rim_um: float = 20, tumour_types=TUMOUR_TYPES):
    """TLS-like aggregates: DBSCAN on B-cell positions; each aggregate = B core (90th pct radius) + rim.

    Sets obs['tls_id'] (TLS1 = largest). Returns (table, B cells in aggregates, total B cells).
    """
    from sklearn.cluster import DBSCAN

    xy = adata.obsm["spatial"]
    is_b = (adata.obs["cell_type"] == "B cell").to_numpy()
    b_xy = xy[is_b]
    labels = DBSCAN(eps=eps_um, min_samples=min_cells).fit_predict(b_xy)
    adata.obs["tls_id"] = "none"
    rows = []
    for k in pd.Series(labels[labels >= 0]).value_counts().index:      # largest first
        pts = b_xy[labels == k]
        centre = pts.mean(axis=0)
        radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90) + rim_um
        inside = np.linalg.norm(xy - centre, axis=1) <= radius
        name = f"TLS{len(rows) + 1}"
        adata.obs.loc[inside, "tls_id"] = name
        ct = adata.obs.loc[inside, "cell_type"]
        row = {"TLS": name, "x (mm)": centre[0] / 1000, "y (mm)": centre[1] / 1000, "radius (um)": radius,
               "cells": int(inside.sum()), "% B": 100 * (ct == "B cell").mean(),
               "% T / NK": 100 * (adata.obs.loc[inside, "lineage"] == "T / NK").mean(),
               "% tumour": 100 * ct.isin(tumour_types).mean(), "mregDC": int((ct == "mregDC (LAMP3+)").sum())}
        if "CCL19" in adata.var_names:
            row["CCL19+ cells"] = int((adata[inside, "CCL19"].layers["counts"] > 0).sum())
        rows.append(row)
    adata.obs["tls_id"] = pd.Categorical(adata.obs["tls_id"])
    return pd.DataFrame(rows).set_index("TLS"), int((labels >= 0).sum()), int(is_b.sum())

"""Step 7: spatial analysis of the tumour immune microenvironment (high-confidence cells).

Delaunay neighbour graph, neighbourhood enrichment (squidpy), composition niches (k-means on the
lineage mix within a radius), distance to the nearest tumour cell, and TLS-like B-cell aggregates.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from common import Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 7: spatial analysis", inputs=["input"], outputs=["output"])
cfg = step.config["spatial"]

full = sc.read_h5ad(step.path("input"))
adata = full[full.obs["annotation_confidence"] == "high"].copy()   # 'mixed' cells would blur contacts
step.say(f"  {adata.n_obs:,} high-confidence cells of {full.n_obs:,}")
xy = adata.obsm["spatial"]

# ---- Delaunay graph, long edges removed ---------------------------------------------------------
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
conn, dist = adata.obsp["spatial_connectivities"].tocsr(), adata.obsp["spatial_distances"].tocsr()
long_edges = dist.data > cfg["max_edge_um"]
conn.data[long_edges] = 0
dist.data[long_edges] = 0
conn.eliminate_zeros(); dist.eliminate_zeros()
adata.obsp["spatial_connectivities"], adata.obsp["spatial_distances"] = conn, dist
step.say(f"  Delaunay: {long_edges.mean():.1%} of edges > {cfg['max_edge_um']} um removed; "
         f"median {np.median(conn.getnnz(axis=1)):.0f} neighbours per cell")

# ---- Neighbourhood enrichment ----------------------------------------------------------------------
TIME_TYPES = [t for t in ["Tumour epithelial (MALL/TCIM)", "Tumour epithelial (CYP2B6/CFTR)", "Tumour epithelial (MYC/CAPN8)",
                          "Proliferating tumour", "CD8 T (GZMK+)", "CD4 T", "Treg", "NK", "B cell", "Plasma cell",
                          "CD163+ macrophage", "FCGR1A+ macrophage", "CXCL9+ macrophage", "Alveolar macrophage",
                          "cDC2", "mregDC (LAMP3+)", "Fibroblast", "Endothelial"] if t in set(adata.obs["cell_type"])]
adata.obs["time_type"] = pd.Categorical(adata.obs["cell_type"].astype(str).where(adata.obs["cell_type"].isin(TIME_TYPES), "other"))
for key, order, clip in [("lineage", None, 50), ("time_type", TIME_TYPES, 30)]:
    sq.gr.nhood_enrichment(adata, cluster_key=key, n_perms=cfg["n_perms"], seed=0, show_progress_bar=False)
    cats = adata.obs[key].cat.categories
    z = pd.DataFrame(adata.uns[f"{key}_nhood_enrichment"]["zscore"], index=cats, columns=cats)
    z = z.loc[order, order] if order else z
    step.save_table(z.round(2), f"step07_nhood_enrichment_{key}")
    fig, ax = plt.subplots(figsize=(10, 8.5) if order else (7.5, 6))
    im = ax.imshow(z.clip(-clip, clip).to_numpy(), cmap="RdBu_r", vmin=-clip, vmax=clip)
    ax.set_xticks(range(len(z)), z.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(z)), z.index, fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label=f"z-score (clipped at +/-{clip})")
    ax.set_title(f"Neighbourhood enrichment ({key})", loc="left")
    step.save_fig(fig, f"step07_nhood_enrichment_{key}")
    adata.uns.pop(f"{key}_nhood_enrichment", None)

# ---- Composition niches ---------------------------------------------------------------------------------
lineages = list(adata.obs["lineage"].cat.categories)
codes = adata.obs["lineage"].cat.codes.to_numpy()
neigh = cKDTree(xy).query_ball_point(xy, r=cfg["niche_radius_um"])
comp = np.zeros((adata.n_obs, len(lineages)))
for i, nb in enumerate(neigh):
    comp[i] = np.bincount(codes[nb], minlength=len(lineages))
comp /= comp.sum(axis=1, keepdims=True)
km = MiniBatchKMeans(n_clusters=cfg["n_niches"], random_state=0, n_init=10, batch_size=8192).fit(comp)
centres = pd.DataFrame(km.cluster_centers_, columns=lineages)


def niche_name(row):
    top = row.sort_values(ascending=False)
    if top.iloc[0] >= 0.85:
        return f"{top.index[0]} {top.iloc[0]:.0%}"
    return f"{top.index[0]} {top.iloc[0]:.0%} + {top.index[1]} {top.iloc[1]:.0%}"


order = centres["Epithelial"].sort_values(ascending=False).index
names = {old: f"N{new + 1}: {niche_name(centres.loc[old])}" for new, old in enumerate(order)}
adata.obs["niche"] = pd.Categorical([names[l] for l in km.labels_], categories=[names[o] for o in order])
centres = centres.loc[order]
centres.index = [names[o] for o in order]
centres["cells"] = adata.obs["niche"].value_counts().reindex(centres.index).to_numpy()
step.save_table(centres.round(3), "step07_niches")

# ---- Distance to the nearest tumour cell -------------------------------------------------------------------
is_tumour = adata.obs["cell_type"].isin(xu.TUMOUR_TYPES).to_numpy()
d_tumour, _ = cKDTree(xy[is_tumour]).query(xy, k=1)
adata.obs["dist_to_tumour_um"] = d_tumour
contact = cfg["contact_um"]
base_contact = (d_tumour[~is_tumour] <= contact).mean()
IMMUNE = ["CD8 T (GZMK+)", "CD4 T", "Treg", "NK", "B cell", "Plasma cell", "CD163+ macrophage", "FCGR1A+ macrophage",
          "CXCL9+ macrophage", "Alveolar macrophage", "LYVE1+ macrophage", "cDC1 (IRF8+)", "cDC2", "mregDC (LAMP3+)", "Fibroblast"]
g = adata.obs[adata.obs["cell_type"].isin(IMMUNE)].groupby("cell_type", observed=True)["dist_to_tumour_um"]
infil = pd.DataFrame({"cells": g.size(), "median distance (um)": g.median(),
                      f"% within {contact} um": 100 * g.apply(lambda d: (d <= contact).mean())})
infil["vs baseline"] = np.where(infil.iloc[:, 2] > 120 * base_contact, "infiltrating",
                                np.where(infil.iloc[:, 2] < 80 * base_contact, "excluded", "similar"))
step.save_table(infil.sort_values("median distance (um)").round(1), "step07_infiltration")
step.say(f"  baseline: {base_contact:.0%} of non-tumour cells within {contact} um; CD8 T: "
         f"{infil.loc['CD8 T (GZMK+)', f'% within {contact} um']:.0f}% ({infil.loc['CD8 T (GZMK+)', 'vs baseline']})")

# ---- TLS-like aggregates: DBSCAN on B cells ------------------------------------------------------------------
b_xy = xy[(adata.obs["cell_type"] == "B cell").to_numpy()]
labels = DBSCAN(eps=cfg["tls_eps_um"], min_samples=cfg["tls_min_cells"]).fit_predict(b_xy)
adata.obs["tls_id"] = "none"
rows = []
for k in pd.Series(labels[labels >= 0]).value_counts().index:   # largest first
    pts = b_xy[labels == k]
    centre = pts.mean(axis=0)
    radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90) + 20   # B-cell core + 20 um rim
    inside = np.linalg.norm(xy - centre, axis=1) <= radius
    name = f"TLS{len(rows) + 1}"
    adata.obs.loc[inside, "tls_id"] = name
    ct = adata.obs.loc[inside, "cell_type"]
    rows.append({"TLS": name, "x (mm)": centre[0] / 1000, "y (mm)": centre[1] / 1000, "radius (um)": radius,
                 "cells": int(inside.sum()), "% B": 100 * (ct == "B cell").mean(),
                 "% T / NK": 100 * (adata.obs.loc[inside, "lineage"] == "T / NK").mean(), "% tumour": 100 * ct.isin(xu.TUMOUR_TYPES).mean()})
adata.obs["tls_id"] = pd.Categorical(adata.obs["tls_id"])
tls = pd.DataFrame(rows).set_index("TLS")
step.save_table(tls.round(2), "step07_tls")
step.say(f"  {len(tls)} TLS-like aggregates; largest at ({tls['x (mm)'].iloc[0]:.2f}, {tls['y (mm)'].iloc[0]:.2f}) mm")

fig, ax = plt.subplots(figsize=(15, 5.2))
for lin, col in xu.LINEAGE_COLORS.items():
    m = (adata.obs["lineage"] == lin).to_numpy()
    ax.scatter(xy[m, 0] / 1000, xy[m, 1] / 1000, s=0.15, c=col, linewidths=0, rasterized=True, label=lin)
for name, r in tls.iterrows():
    ax.add_patch(Circle((r["x (mm)"], r["y (mm)"]), r["radius (um)"] / 1000 + 0.05, fill=False, lw=1.2))
    ax.annotate(name, (r["x (mm)"] - r["radius (um)"] / 1000 - 0.08, r["y (mm)"]), ha="right", fontsize=8.5, weight="bold")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.legend(markerscale=30, frameon=False, ncol=7, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
ax.set_title("TLS-like aggregates (circled) on the lineage map", loc="left")
step.save_fig(fig, "step07_tls")

adata.uns["step7_params"] = {k: v for k, v in cfg.items()}
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

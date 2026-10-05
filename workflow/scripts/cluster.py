"""Step 5: normalise, PCA, neighbour graph, Leiden at several resolutions, UMAP, markers.

Same recipe and seeds as the notebook, so the cluster numbers match and Step 6's label tables apply.
"""

import tarfile

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

from common import Step

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 5: normalise and cluster", inputs=["input"], outputs=["output"])
cfg = step.config["cluster"]
seed = cfg["seed"]

adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)               # to the median library size
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
sc.pp.scale(adata, max_value=10)           # for PCA only
sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
adata.X = adata.layers["lognorm"]
vr = adata.uns["pca"]["variance_ratio"]
step.say(f"  PCA: {cfg['n_pcs']} PCs explain {vr[:cfg['n_pcs']].sum():.0%} of variance")

sc.pp.neighbors(adata, n_neighbors=cfg["n_neighbors"], n_pcs=cfg["n_pcs"], random_state=seed)
for r in cfg["resolutions"]:
    sc.tl.leiden(adata, resolution=r, key_added=f"leiden_{r}", flavor="igraph", n_iterations=2, directed=False, random_state=seed)
sc.tl.umap(adata, random_state=seed)

# ---- Benchmark against the 10x graph-based clustering ----------------------------------------
with tarfile.open(step.outs_dir / "analysis.tar.gz") as tf:
    tenx = pd.read_csv(tf.extractfile("analysis/clustering/gene_expression_graphclust/clusters.csv"), index_col="Barcode")["Cluster"]
shared = adata.obs_names.intersection(tenx.index)
adata.obs["tenx_graphclust"] = pd.Categorical(tenx.reindex(adata.obs_names).astype("Int64").astype(str))
summary = pd.DataFrame({"clusters": [adata.obs[f"leiden_{r}"].nunique() for r in cfg["resolutions"]],
                        "ARI vs 10x": [adjusted_rand_score(tenx[shared], adata.obs.loc[shared, f"leiden_{r}"]) for r in cfg["resolutions"]]},
                       index=pd.Index(cfg["resolutions"], name="resolution")).round(3)
step.save_table(summary, "step05_clusters")
step.say(summary.to_string())

# ---- Markers for the working clustering ----------------------------------------------------------
key = cfg["working"]
sc.tl.rank_genes_groups(adata, key, method="wilcoxon", use_raw=False, layer="lognorm", key_added="markers")
markers = sc.get.rank_genes_groups_df(adata, group=None, key="markers")
top = markers[(markers.pvals_adj < 0.01) & (markers.logfoldchanges > 1)].groupby("group", observed=True).head(6)
top_table = top.groupby("group", observed=True)["names"].apply(", ".join).rename("top markers").to_frame()
top_table.insert(0, "cells", adata.obs[key].value_counts().reindex(top_table.index).to_numpy())
step.save_table(top_table, "step05_markers")

# ---- Figures: UMAP and tissue map of the working clusters ------------------------------------------
labels = adata.obs[key].to_numpy()
clusters = adata.obs[key].cat.categories
colors = dict(zip(clusters, sc.pl.palettes.default_20))
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 2.4]})
for ax, coords in zip(axes, [adata.obsm["X_umap"], adata.obsm["spatial"] / 1000]):
    ax.scatter(coords[:, 0], coords[:, 1], s=0.2, c=[colors[c] for c in labels], linewidths=0, rasterized=True)
    for c in clusters:
        m = np.median(coords[labels == c], axis=0)
        ax.text(m[0], m[1], c, fontsize=9, weight="bold", ha="center", bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.set_aspect("equal" if coords is not adata.obsm["X_umap"] else "auto")
    ax.set_xticks([]); ax.set_yticks([])
axes[1].invert_yaxis()
axes[0].set_title(f"UMAP, {len(clusters)} clusters ({key})", loc="left")
axes[1].set_title("Clusters in the tissue", loc="left")
step.save_fig(fig, "step05_clusters")

adata.uns["step5_params"] = {"normalisation": "normalize_total + log1p; layers['lognorm']", "n_pcs": cfg["n_pcs"],
                             "n_neighbors": cfg["n_neighbors"], "resolutions": cfg["resolutions"], "working_clusters": key}
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

"""Step 5: normalise, PCA, neighbour graph, Leiden, UMAP, markers, benchmark against 10x's clustering (ist_analysis.cluster).

Same recipe and seeds as the notebook, so cluster numbers match and Step 6's label tables apply.
"""

import matplotlib
import numpy as np
import scanpy as sc

from common import Step
from ist_analysis import cluster
from ist_analysis.io import xenium

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 5: normalise and cluster", inputs=["input"], outputs=["output"])
cfg = step.config["cluster"]

adata = sc.read_h5ad(step.path("input"))
cluster.normalise_and_pca(adata)
vr = adata.uns["pca"]["variance_ratio"]
step.say(f"  PCA: {cfg['n_pcs']} PCs explain {vr[:cfg['n_pcs']].sum():.0%} of variance")
cluster.cluster(adata, cfg["n_pcs"], cfg["n_neighbors"], cfg["resolutions"], cfg["seed"])

summary = cluster.benchmark_vs_tenx(adata, xenium.vendor_clusters(step.outs_dir), cfg["resolutions"]).round(3)
step.save_table(summary, "step05_clusters")
step.say(summary.to_string())
key = cfg["working"]
step.save_table(cluster.marker_table(adata, key), "step05_markers")

labels = adata.obs[key].to_numpy()
clusters = adata.obs[key].cat.categories
colors = dict(zip(clusters, sc.pl.palettes.default_20))
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 2.4]})
for ax, coords, equal in zip(axes, [adata.obsm["X_umap"], adata.obsm["spatial"] / 1000], [False, True]):
    ax.scatter(coords[:, 0], coords[:, 1], s=0.2, c=[colors[c] for c in labels], linewidths=0, rasterized=True)
    for c in clusters:
        m = np.median(coords[labels == c], axis=0)
        ax.text(m[0], m[1], c, fontsize=9, weight="bold", ha="center", bbox=dict(fc="white", ec="none", alpha=0.7))
    if equal:
        ax.set_aspect("equal")
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

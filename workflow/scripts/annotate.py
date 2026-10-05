"""Step 6: cell-type annotation from marker scores, coarse cluster labels and lineage subclustering.

All labels come from config/annotation.yaml. Each subcluster label carries a guard marker that must be
among the subcluster's top-10 markers, so a changed clustering stops the pipeline instead of mislabelling.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from common import REPO_ROOT, Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 6: annotate cell types", inputs=["input"], outputs=["output"])
key = step.config["cluster"]["working"]
seed = step.config["cluster"]["seed"]
with open(REPO_ROOT / step.config["annotation_file"]) as fh:
    ann_cfg = yaml.safe_load(fh)

adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"]

# ---- Marker-set scores -----------------------------------------------------------------------
panel = set(adata.var_names)
sets = {k: [g for g in v if g in panel] for k, v in ann_cfg["marker_sets"].items()}
sets = {k: v for k, v in sets.items() if len(v) >= 2}
for name, genes in sets.items():
    sc.tl.score_genes(adata, genes, score_name=f"score_{name}", random_state=seed)
cl = adata.obs.groupby(key, observed=True)[[f"score_{k}" for k in sets]].mean()
cl.columns = list(sets)
step.save_table(((cl - cl.mean()) / cl.std()).round(2), "step06_marker_scores_by_cluster")

# ---- Coarse labels ------------------------------------------------------------------------------
coarse = ann_cfg["coarse"]
missing = set(adata.obs[key].cat.categories) - set(coarse)
if missing:
    raise ValueError(f"Clusters without a label in annotation.yaml: {sorted(missing)} (did Step 5 change?)")
adata.obs["cell_type_coarse"] = adata.obs[key].map({k: v[0] for k, v in coarse.items()}).astype("category")
adata.obs["lineage"] = adata.obs[key].map({k: v[1] for k, v in coarse.items()}).astype("category")


def subcluster(clusters, resolution, n_pcs=20):
    """Re-cluster one lineage with its own PCA (same recipe as the notebook)."""
    sub = adata[adata.obs[key].isin(clusters)].copy()
    sub.X = sub.layers["lognorm"].copy()
    sc.pp.scale(sub, max_value=10)
    sc.pp.pca(sub, n_comps=30, svd_solver="arpack", random_state=seed)
    sub.X = sub.layers["lognorm"]
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
    sc.tl.leiden(sub, resolution=resolution, key_added="sub", flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    sc.tl.rank_genes_groups(sub, "sub", method="wilcoxon", layer="lognorm", key_added="sub_markers")
    return sub


def apply_labels(sub, labels, name):
    top10 = pd.DataFrame(sub.uns["sub_markers"]["names"]).head(10)
    problems = [f"sub {k}: guard '{g}' not in top markers {list(top10[k][:5])}" for k, (_, _, g) in labels.items()
                if k not in top10 or g not in set(top10[k])]
    if problems or set(labels) != set(sub.obs["sub"].cat.categories):
        raise ValueError(f"{name} subclusters changed; re-check annotation.yaml.\n" + "\n".join(problems))
    sub.obs["cell_type_fine"] = sub.obs["sub"].map({k: v[0] for k, v in labels.items()})
    sub.obs["annotation_confidence"] = sub.obs["sub"].map({k: v[1] for k, v in labels.items()})
    table = pd.DataFrame({"cells": sub.obs["sub"].value_counts(), "top markers": top10.head(6).T.apply(", ".join, axis=1),
                          "label": {k: v[0] for k, v in labels.items()}, "confidence": {k: v[1] for k, v in labels.items()}})
    step.save_table(table.loc[sub.obs["sub"].cat.categories], f"step06_subclusters_{name}")


subs = {}
for name, spec in ann_cfg["subclusters"].items():
    subs[name] = subcluster(spec["clusters"], ann_cfg["subcluster_resolution"])
    apply_labels(subs[name], spec["labels"], name)
    step.say(f"  {name}: {subs[name].n_obs:,} cells in {subs[name].obs['sub'].nunique()} subclusters, labels verified")

# ---- Merge fine labels, set lineages -------------------------------------------------------------
fine = adata.obs["cell_type_coarse"].astype(str)
conf = pd.Series("high", index=adata.obs_names)
for sub in subs.values():
    fine.loc[sub.obs_names] = sub.obs["cell_type_fine"].astype(str)
    conf.loc[sub.obs_names] = sub.obs["annotation_confidence"].astype(str)
adata.obs["cell_type"] = pd.Categorical(fine)
adata.obs["annotation_confidence"] = pd.Categorical(conf)
lineage = adata.obs["cell_type"].astype(str).map(ann_cfg["lineage_overrides"])
lineage.loc[subs["Myeloid"].obs_names] = ann_cfg["all_myeloid_subclusters_are"]
adata.obs["lineage"] = pd.Categorical(lineage.fillna(adata.obs["lineage"].astype(str)))

comp = adata.obs.groupby(["lineage", "cell_type"], observed=True).size().rename("cells").reset_index()
comp["% of cells"] = (100 * comp["cells"] / adata.n_obs).round(2)
step.save_table(comp.sort_values(["lineage", "cells"], ascending=[True, False]), "step06_cell_types", index=False)
step.say(f"  {adata.obs['cell_type'].nunique()} cell types; {(adata.obs['annotation_confidence'] == 'mixed').mean():.1%} flagged mixed")

xy = adata.obsm["spatial"] / 1000
fig, ax = plt.subplots(figsize=(15, 5.6))
for lin, col in xu.LINEAGE_COLORS.items():
    m = (adata.obs["lineage"] == lin).to_numpy()
    ax.scatter(xy[m, 0], xy[m, 1], s=0.25, c=col, linewidths=0, rasterized=True, label=f"{lin} ({m.sum():,})")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.legend(markerscale=12, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
ax.set_title("Cell lineages in the tissue", loc="left")
step.save_fig(fig, "step06_lineages")

adata.uns["annotation"] = {"coarse_from": key, "subcluster_resolution": ann_cfg["subcluster_resolution"],
                           "label_tables": "config/annotation.yaml",
                           "caveats": "Panel lacks SFTPC/NAPSA/KRT5/TP63/SCGB1A1; tumour call inferred, not CNV-confirmed."}
adata.uns["lineage_colors"] = [xu.LINEAGE_COLORS[l] for l in adata.obs["lineage"].cat.categories]
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

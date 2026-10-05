"""Cell-type annotation: marker-set scores, cluster labels with marker guards, lineage subclustering (Step 6).

The label decisions live in config/annotation.yaml. Every subcluster label names a guard marker that must
be among that subcluster's top-10 markers, so a changed clustering raises an error instead of silently
mislabelling cells.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_annotation(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def score_marker_sets(adata, marker_sets: dict, seed: int = 0, min_genes: int = 2) -> dict:
    """Score every cell for each marker set (genes absent from the panel skipped); obs['score_<set>']."""
    import scanpy as sc

    panel = set(adata.var_names)
    sets = {k: [g for g in v if g in panel] for k, v in marker_sets.items()}
    sets = {k: v for k, v in sets.items() if len(v) >= min_genes}
    for name, genes in sets.items():
        sc.tl.score_genes(adata, genes, score_name=f"score_{name}", random_state=seed)
    return sets


def cluster_score_z(adata, key: str, sets: dict) -> pd.DataFrame:
    """Mean score per cluster, z-scored across clusters (columns comparable)."""
    cl = adata.obs.groupby(key, observed=True)[[f"score_{k}" for k in sets]].mean()
    cl.columns = list(sets)
    return (cl - cl.mean()) / cl.std()


def apply_coarse(adata, key: str, coarse: dict) -> None:
    """obs['cell_type_coarse'] and obs['lineage'] from the cluster -> [label, lineage] table."""
    missing = set(adata.obs[key].cat.categories) - set(coarse)
    if missing:
        raise ValueError(f"Clusters without a label: {sorted(missing)} (did the clustering change?)")
    adata.obs["cell_type_coarse"] = adata.obs[key].map({k: v[0] for k, v in coarse.items()}).astype("category")
    adata.obs["lineage"] = adata.obs[key].map({k: v[1] for k, v in coarse.items()}).astype("category")


def subcluster(adata, key: str, clusters, resolution: float = 0.6, n_pcs: int = 20, seed: int = 0):
    """Re-cluster a subset with its own PCA, so within-lineage differences surface; markers in uns['sub_markers']."""
    import scanpy as sc

    sub = adata[adata.obs[key].isin(clusters)].copy()
    sub.X = sub.layers["lognorm"].copy()
    sc.pp.scale(sub, max_value=10)
    sc.pp.pca(sub, n_comps=30, svd_solver="arpack", random_state=seed)
    sub.X = sub.layers["lognorm"]
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=n_pcs, random_state=seed)
    sc.tl.leiden(sub, resolution=resolution, key_added="sub", flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    sc.tl.rank_genes_groups(sub, "sub", method="wilcoxon", layer="lognorm", key_added="sub_markers")
    return sub


def apply_labels(sub, labels: dict, name: str = "subset") -> pd.DataFrame:
    """Label subclusters from {sub: [label, confidence, guard marker]}; raise if any guard fails.

    Returns a table: cells, top markers, label and confidence per subcluster.
    """
    top10 = pd.DataFrame(sub.uns["sub_markers"]["names"]).head(10)
    problems = [f"sub {k}: guard '{g}' not in top markers {list(top10[k][:5]) if k in top10 else []}"
                for k, (_, _, g) in labels.items() if k not in top10 or g not in set(top10[k])]
    if problems or set(labels) != set(sub.obs["sub"].cat.categories):
        raise ValueError(f"{name} subclusters changed; re-check the label table.\n" + "\n".join(problems))
    sub.obs["cell_type_fine"] = sub.obs["sub"].map({k: v[0] for k, v in labels.items()})
    sub.obs["annotation_confidence"] = sub.obs["sub"].map({k: v[1] for k, v in labels.items()})
    table = pd.DataFrame({"cells": sub.obs["sub"].value_counts(), "top markers": top10.head(6).T.apply(", ".join, axis=1),
                          "label": {k: v[0] for k, v in labels.items()}, "confidence": {k: v[1] for k, v in labels.items()}})
    return table.loc[sub.obs["sub"].cat.categories]


def merge_labels(adata, subs: dict, lineage_overrides: dict, myeloid_subset: str = "Myeloid", myeloid_lineage: str = "Myeloid") -> None:
    """Write fine labels back (obs['cell_type'], obs['annotation_confidence']) and set the final lineage."""
    fine = adata.obs["cell_type_coarse"].astype(str)
    conf = pd.Series("high", index=adata.obs_names)
    for sub in subs.values():
        fine.loc[sub.obs_names] = sub.obs["cell_type_fine"].astype(str)
        conf.loc[sub.obs_names] = sub.obs["annotation_confidence"].astype(str)
    adata.obs["cell_type"] = pd.Categorical(fine)
    adata.obs["annotation_confidence"] = pd.Categorical(conf)
    lineage = adata.obs["cell_type"].astype(str).map(lineage_overrides)
    if myeloid_subset in subs:
        lineage.loc[subs[myeloid_subset].obs_names] = myeloid_lineage
    adata.obs["lineage"] = pd.Categorical(lineage.fillna(adata.obs["lineage"].astype(str)))


def annotate(adata, ann: dict, key: str, seed: int = 0):
    """The whole Step 6 annotation from an annotation.yaml dict. Returns (marker-set z table, subclusters, sub tables)."""
    sets = score_marker_sets(adata, ann["marker_sets"], seed=seed)
    z = cluster_score_z(adata, key, sets)
    apply_coarse(adata, key, ann["coarse"])
    subs, tables = {}, {}
    for name, spec in ann["subclusters"].items():
        subs[name] = subcluster(adata, key, spec["clusters"], ann["subcluster_resolution"], seed=seed)
        tables[name] = apply_labels(subs[name], spec["labels"], name)
    merge_labels(adata, subs, ann["lineage_overrides"], myeloid_lineage=ann["all_myeloid_subclusters_are"])
    return z, subs, tables


def composition(adata) -> pd.DataFrame:
    """Cells per lineage and cell type."""
    comp = adata.obs.groupby(["lineage", "cell_type"], observed=True).size().rename("cells").reset_index()
    comp["% of cells"] = (100 * comp["cells"] / adata.n_obs).round(2)
    return comp.sort_values(["lineage", "cells"], ascending=[True, False])

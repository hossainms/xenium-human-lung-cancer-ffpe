"""Notebook 03: validate the manual cell types against the Lung Cancer Atlas (LuCA).

A1: restrict the atlas to the panel genes (block-streamed with h5py, cached). B: train CellTypist's model
(scaled log-normalised expression + multinomial logistic regression, fitted with scikit-learn) on a
balanced sample, with held-out donors as the ceiling. C: annotate the 10x and Proseg cells. D: agreement
with the manual labels by lineage and cell type, and the spillover / platform check for weak types.
"""

import pickle

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import Step, mad_bounds

step = Step("Notebook 03: reference mapping to LuCA", inputs=["atlas", "annotated", "proseg"], outputs=["marker"])
cfg = step.config["reference"]
atlas_path = step.path("atlas")
cache = atlas_path.with_name("luca_core_panel_genes.h5ad")
ann = sc.read_h5ad(step.path("annotated"))

# ---- A1: atlas restricted to the panel genes (matched by Ensembl id), cached ------------------------
if not cache.exists():
    obs = ad.read_h5ad(atlas_path, backed="r").obs[["cell_type_major", "cell_type_tumor", "ann_coarse", "origin", "tissue",
                                                    "platform", "dataset", "donor_id"]].copy()
    with h5py.File(atlas_path) as f:
        ids = f["raw/var/_index"][:].astype(str)
        X, indptr = f["raw/X"], f["raw/X/indptr"][:]
        panel_ids = ann.var["gene_ids"]
        shared = panel_ids[panel_ids.isin(ids)]
        new_col = np.full(len(ids), -1)
        new_col[pd.Index(ids).get_indexer(shared)] = np.arange(len(shared))
        blocks = []
        for start in range(0, len(indptr) - 1, 50_000):
            stop = min(start + 50_000, len(indptr) - 1)
            a, b = indptr[start], indptr[stop]
            cols = new_col[X["indices"][a:b]]
            rows = np.repeat(np.arange(stop - start), np.diff(indptr[start:stop + 1]))
            keep = cols >= 0
            blocks.append(sp.csr_matrix((X["data"][a:b][keep], (rows[keep], cols[keep])), shape=(stop - start, len(shared)), dtype=np.float32))
    ad.AnnData(sp.vstack(blocks).tocsr(), obs=obs, var=pd.DataFrame({"gene_ids": shared.to_numpy()}, index=shared.index)).write_h5ad(cache, compression="gzip")
ref = ad.read_h5ad(cache)
genes = ref.var_names
step.say(f"  atlas: {ref.n_obs:,} cells, {ref.obs['donor_id'].nunique()} donors; {ref.n_vars} of {ann.n_vars} panel genes shared")

# ---- B: balanced training set, held-out donors -----------------------------------------------------------------
o = ref.obs
eligible = (o["tissue"] == "lung") & (o["platform"] != "Smart-seq2") & (o["cell_type_major"] != "other") & o["origin"].notna()
rng = np.random.default_rng(0)
donors = o.loc[eligible, "donor_id"].astype(str).unique()
test_donors = set(rng.choice(donors, size=int(cfg["test_donor_fraction"] * len(donors)), replace=False))
is_test = o["donor_id"].astype(str).isin(test_donors)
train_idx = o[eligible & ~is_test].groupby("cell_type_major", observed=True, group_keys=False).apply(
    lambda d: d.sample(min(len(d), cfg["max_cells_per_type"]), random_state=0)).index
test_idx = o[eligible & is_test].groupby("cell_type_major", observed=True, group_keys=False).apply(
    lambda d: d.sample(min(len(d), 1000), random_state=0)).index


def lognorm(adata):
    """Raw counts on the shared genes -> 10,000 per cell -> log1p (normalised over the same genes on both sides)."""
    sub = adata[:, genes]
    X = sp.csr_matrix(sub.layers["counts"] if "counts" in sub.layers else sub.X, dtype=np.float32)
    X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)) @ X
    X.data = np.log1p(X.data)
    return X


model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=0.1, max_iter=1000))
model.fit(lognorm(ref[train_idx]), o.loc[train_idx, "cell_type_major"].astype(str).to_numpy())
with open(step.results / "reference_mapping_model.pkl", "wb") as fh:
    pickle.dump({"model": model, "genes": list(genes)}, fh)


def annotate(adata):
    P = model.predict_proba(lognorm(adata))
    return pd.DataFrame({"atlas_label": model.classes_[P.argmax(axis=1)], "atlas_prob": P.max(axis=1)}, index=adata.obs_names)


truth = o.loc[test_idx, "cell_type_major"].astype(str).to_numpy()
held = annotate(ref[test_idx])["atlas_label"].to_numpy()
ceiling = pd.Series(held == truth).groupby(truth).mean()
step.save_table(ceiling.rename("held-out accuracy").round(3).to_frame(), "nb03_ceiling")
step.say(f"  ceiling on {len(test_donors)} held-out donors: {(held == truth).mean():.1%}")

# ---- C: annotate both segmentations --------------------------------------------------------------------------------
atlas10 = annotate(ann)
pro = sc.read_h5ad(step.path("proseg"))
pro.obs["n_genes"] = np.asarray((pro.X > 0).sum(axis=1)).ravel()
lo, hi = mad_bounds(pro.obs["cell_area"].to_numpy(), step.config["qc"]["area_nmads"])
pro = pro[((pro.obs["transcript_counts"] >= 10) & (pro.obs["n_genes"] >= 5) & pro.obs["cell_area"].between(lo, hi)).to_numpy()].copy()
atlasP = annotate(pro)

# ---- D: agreement with the manual labels ------------------------------------------------------------------------------
ATLAS_LINEAGE = {"Tumor cells": "Epithelial", "Alveolar cell type 1": "Epithelial", "Alveolar cell type 2": "Epithelial",
                 "transitional club/AT2": "Epithelial", "Club": "Epithelial", "Ciliated": "Epithelial", "T cell CD4": "T / NK",
                 "T cell CD8": "T / NK", "T cell regulatory": "T / NK", "NK cell": "T / NK", "B cell": "B / plasma", "Plasma cell": "B / plasma",
                 "Macrophage": "Myeloid", "Macrophage alveolar": "Myeloid", "Monocyte": "Myeloid", "Neutrophils": "Myeloid", "cDC1": "Myeloid",
                 "cDC2": "Myeloid", "DC mature": "Myeloid", "pDC": "Myeloid", "Mast cell": "Mast", "Stromal": "Stromal", "Endothelial cell": "Endothelial"}
CONFIRMED_BY = {"Tumour epithelial (MALL/TCIM)": ["Tumor cells"], "Tumour epithelial (CYP2B6/CFTR)": ["Tumor cells"],
                "Tumour epithelial (MYC/CAPN8)": ["Tumor cells"], "Proliferating tumour": ["Tumor cells"], "Ciliated epithelial": ["Ciliated"],
                "CD8 T (GZMK+)": ["T cell CD8"], "CD4 T": ["T cell CD4"], "Treg": ["T cell regulatory"], "NK": ["NK cell"],
                "Proliferating T": ["T cell CD4", "T cell CD8", "T cell regulatory"], "B cell": ["B cell"], "Plasma cell": ["Plasma cell"],
                "CD163+ macrophage": ["Macrophage"], "FCGR1A+ macrophage": ["Macrophage"], "LYVE1+ macrophage": ["Macrophage"],
                "CXCL9+ macrophage": ["Macrophage"], "Alveolar macrophage": ["Macrophage alveolar"],
                "Proliferating myeloid": ["Macrophage", "Macrophage alveolar", "Monocyte"], "Monocyte": ["Monocyte", "Neutrophils"],
                "cDC1 (IRF8+)": ["cDC1"], "cDC2": ["cDC2"], "mregDC (LAMP3+)": ["DC mature"], "Mast cell": ["Mast cell"],
                "Fibroblast": ["Stromal"], "Smooth muscle": ["Stromal"], "Endothelial": ["Endothelial cell"], "Lymphatic endothelial": ["Endothelial cell"]}
manual = ann.obs[["cell_type", "lineage", "annotation_confidence"]].astype(str)
high = manual.index[manual["annotation_confidence"] == "high"]
lin_agree = (manual.loc[high, "lineage"] == atlas10.loc[high, "atlas_label"].map(ATLAS_LINEAGE)).mean()
lin_tab = pd.crosstab(manual.loc[high, "lineage"], atlas10.loc[high, "atlas_label"].map(ATLAS_LINEAGE), normalize="index") * 100
step.save_table(lin_tab.round(1), "nb03_lineage_agreement")
step.say(f"  lineage agreement on {len(high):,} high-confidence cells: {lin_agree:.1%}")


def agreement(atlas, cells):
    rows = []
    for t, ok in CONFIRMED_BY.items():
        idx = cells[manual.loc[cells, "cell_type"].to_numpy() == t].intersection(atlas.index)
        if len(idx):
            lab = atlas.loc[idx, "atlas_label"]
            rows.append({"manual type": t, "cells": len(idx), "% confirmed": 100 * lab.isin(ok).mean(),
                         "most common atlas call": lab.value_counts().index[0], "median probability": atlas.loc[idx, "atlas_prob"].median(),
                         "atlas ceiling (%)": 100 * ceiling.reindex(ok).mean()})
    return pd.DataFrame(rows).set_index("manual type")


step.save_table(agreement(atlas10, high).round(2), "nb03_agreement_by_cell_type")
paired = manual.index.intersection(atlasP.index).intersection(high)
seg = pd.DataFrame({"% confirmed, 10x": agreement(atlas10, paired)["% confirmed"], "% confirmed, Proseg": agreement(atlasP, paired)["% confirmed"]})
step.save_table(seg.round(1), "nb03_agreement_10x_vs_proseg")

export = manual.join(atlas10.add_prefix("tenx_"), how="left").join(atlasP.add_prefix("proseg_"), how="left")
export.index.name = "cell_id"
export.to_csv(step.tables / "nb03_atlas_labels.csv.gz")
marker = step.path("marker")
marker.write_text("notebook 03 complete\n")
step.done(marker)

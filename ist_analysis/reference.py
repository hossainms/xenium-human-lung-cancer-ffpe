"""Reference mapping to the Lung Cancer Atlas (LuCA core, CELLxGENE) with CellTypist's model (notebook 03).

The model is CellTypist's recipe (scale each gene, then L2-regularised logistic regression) fitted with
scikit-learn directly: celltypist 1.7.1's default trainer fails on scikit-learn 1.9, and its SGD fallback
gives uncalibrated probabilities on Xenium cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ATLAS_LINEAGE = {"Tumor cells": "Epithelial", "Alveolar cell type 1": "Epithelial", "Alveolar cell type 2": "Epithelial",
                 "transitional club/AT2": "Epithelial", "Club": "Epithelial", "Ciliated": "Epithelial", "T cell CD4": "T / NK",
                 "T cell CD8": "T / NK", "T cell regulatory": "T / NK", "NK cell": "T / NK", "B cell": "B / plasma", "Plasma cell": "B / plasma",
                 "Macrophage": "Myeloid", "Macrophage alveolar": "Myeloid", "Monocyte": "Myeloid", "Neutrophils": "Myeloid", "cDC1": "Myeloid",
                 "cDC2": "Myeloid", "DC mature": "Myeloid", "pDC": "Myeloid", "Mast cell": "Mast", "Stromal": "Stromal", "Endothelial cell": "Endothelial"}
CONFIRMED_BY = {   # manual type (Step 6) -> atlas types that agree with it, written down before comparing
    "Tumour epithelial (MALL/TCIM)": ["Tumor cells"], "Tumour epithelial (CYP2B6/CFTR)": ["Tumor cells"],
    "Tumour epithelial (MYC/CAPN8)": ["Tumor cells"], "Proliferating tumour": ["Tumor cells"], "Ciliated epithelial": ["Ciliated"],
    "CD8 T (GZMK+)": ["T cell CD8"], "CD4 T": ["T cell CD4"], "Treg": ["T cell regulatory"], "NK": ["NK cell"],
    "Proliferating T": ["T cell CD4", "T cell CD8", "T cell regulatory"], "B cell": ["B cell"], "Plasma cell": ["Plasma cell"],
    "CD163+ macrophage": ["Macrophage"], "FCGR1A+ macrophage": ["Macrophage"], "LYVE1+ macrophage": ["Macrophage"],
    "CXCL9+ macrophage": ["Macrophage"], "Alveolar macrophage": ["Macrophage alveolar"],
    "Proliferating myeloid": ["Macrophage", "Macrophage alveolar", "Monocyte"], "Monocyte": ["Monocyte", "Neutrophils"],
    "cDC1 (IRF8+)": ["cDC1"], "cDC2": ["cDC2"], "mregDC (LAMP3+)": ["DC mature"], "Mast cell": ["Mast cell"],
    "Fibroblast": ["Stromal"], "Smooth muscle": ["Stromal"], "Endothelial": ["Endothelial cell"], "Lymphatic endothelial": ["Endothelial cell"]}
OBS_COLUMNS = ["cell_type_major", "cell_type_tumor", "ann_coarse", "origin", "tissue", "platform", "dataset", "donor_id"]


def build_panel_reference(atlas_path: Path, panel_ids: pd.Series, out_path: Path, block: int = 50_000):
    """The atlas restricted to the panel genes (matched by Ensembl id), raw counts, streamed in row blocks.

    Reading the 12.9 GB file with anndata's backed mode row by row was far slower than streaming
    contiguous blocks of the CSR arrays with h5py. panel_ids: Ensembl ids indexed by gene symbol.
    """
    import anndata as ad
    import h5py

    obs = ad.read_h5ad(atlas_path, backed="r").obs[OBS_COLUMNS].copy()
    with h5py.File(atlas_path) as f:
        atlas_ids = f["raw/var/_index"][:].astype(str)
        X, indptr = f["raw/X"], f["raw/X/indptr"][:]
        shared = panel_ids[panel_ids.isin(atlas_ids)]
        new_col = np.full(len(atlas_ids), -1)                  # atlas column -> panel column (-1 = drop)
        new_col[pd.Index(atlas_ids).get_indexer(shared)] = np.arange(len(shared))
        blocks = []
        for start in range(0, len(indptr) - 1, block):
            stop = min(start + block, len(indptr) - 1)
            a, b = indptr[start], indptr[stop]
            cols = new_col[X["indices"][a:b]]
            rows = np.repeat(np.arange(stop - start), np.diff(indptr[start:stop + 1]))
            keep = cols >= 0
            blocks.append(sp.csr_matrix((X["data"][a:b][keep], (rows[keep], cols[keep])), shape=(stop - start, len(shared)), dtype=np.float32))
    ref = ad.AnnData(sp.vstack(blocks).tocsr(), obs=obs, var=pd.DataFrame({"gene_ids": shared.to_numpy()}, index=shared.index))
    ref.write_h5ad(out_path, compression="gzip")
    return ref


def eligible(obs: pd.DataFrame) -> pd.Series:
    """Training cells: lung tissue, UMI platforms (no Smart-seq2), labelled (no 'other'), known origin."""
    return (obs["tissue"] == "lung") & (obs["platform"] != "Smart-seq2") & (obs["cell_type_major"] != "other") & obs["origin"].notna()


def donor_split(obs: pd.DataFrame, ok: pd.Series, test_fraction: float = 0.2, max_per_type: int = 5000, max_test_per_type: int = 1000, seed: int = 0):
    """Balanced training set from (1 - test_fraction) of donors; test set from the held-out donors."""
    rng = np.random.default_rng(seed)
    donors = obs.loc[ok, "donor_id"].astype(str).unique()
    test_donors = set(rng.choice(donors, size=int(test_fraction * len(donors)), replace=False))
    is_test = obs["donor_id"].astype(str).isin(test_donors)
    sample = lambda d, n: d.sample(min(len(d), n), random_state=seed)   # noqa: E731
    train = obs[ok & ~is_test].groupby("cell_type_major", observed=True, group_keys=False).apply(lambda d: sample(d, max_per_type)).index
    test = obs[ok & is_test].groupby("cell_type_major", observed=True, group_keys=False).apply(lambda d: sample(d, max_test_per_type)).index
    return train, test, test_donors, donors


def lognorm(adata, genes) -> sp.csr_matrix:
    """Raw counts on `genes` -> 10,000 per cell -> log1p. Normalised over the same genes on both sides."""
    sub = adata[:, genes]
    X = sp.csr_matrix(sub.layers["counts"] if "counts" in sub.layers else sub.X, dtype=np.float32)
    X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)) @ X
    X.data = np.log1p(X.data)
    return X


def train_model(X, y, C: float = 0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=C, max_iter=1000))   # multinomial, L2
    return model.fit(X, y)


def predict(model, X, index) -> pd.DataFrame:
    """Most probable atlas type and its probability (probabilities sum to 1 across types)."""
    P = model.predict_proba(X)
    return pd.DataFrame({"atlas_label": model.classes_[P.argmax(axis=1)], "atlas_prob": P.max(axis=1)}, index=index)


def agreement(atlas: pd.DataFrame, manual: pd.DataFrame, cells, ceiling: pd.Series) -> pd.DataFrame:
    """Per manual type: % confirmed by the atlas, most common atlas call, median probability, atlas ceiling."""
    rows = []
    for t, ok in CONFIRMED_BY.items():
        idx = cells[manual.loc[cells, "cell_type"].to_numpy() == t].intersection(atlas.index)
        if len(idx) == 0:
            continue
        lab = atlas.loc[idx, "atlas_label"]
        top = lab.value_counts(normalize=True)
        rows.append({"manual type": t, "lineage": manual.loc[idx[0], "lineage"], "cells": len(idx),
                     "% confirmed by atlas": 100 * lab.isin(ok).mean(), "most common atlas call": f"{top.index[0]} ({100 * top.iloc[0]:.0f}%)",
                     "median probability": atlas.loc[idx, "atlas_prob"].median(), "atlas ceiling (%)": 100 * ceiling.reindex(ok).mean()})
    return pd.DataFrame(rows).set_index("manual type")

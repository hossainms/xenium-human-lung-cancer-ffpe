"""Small synthetic datasets with known answers, so tests run in seconds without the real data."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tissue(rng):
    """A 2 x 1 mm tissue: a tumour block on the left, stroma on the right, and two tight B-cell clusters (TLS)
    in the stroma, each with a T-cell rim. Counts: 6 genes, lineage-specific."""
    n_tum, n_str = 1500, 1000
    tum = np.column_stack([rng.uniform(0, 900, n_tum), rng.uniform(0, 1000, n_tum)])
    stroma = np.column_stack([rng.uniform(902, 2000, n_str), rng.uniform(0, 1000, n_str)])   # touching the tumour edge
    centres = np.array([[1400, 300], [1700, 750]])
    b = np.vstack([c + rng.normal(0, 8, (60, 2)) for c in centres])
    t = np.vstack([c + np.column_stack([np.cos(a), np.sin(a)]) * 35 for c in centres
                   for a in [rng.uniform(0, 2 * np.pi, 40)]])
    xy = np.vstack([tum, stroma, b, t])
    cell_type = (["Tumour epithelial (MYC/CAPN8)"] * n_tum + ["Fibroblast"] * n_str + ["B cell"] * len(b) + ["CD8 T (GZMK+)"] * len(t))
    lineage = {"Tumour epithelial (MYC/CAPN8)": "Epithelial", "Fibroblast": "Stromal", "B cell": "B / plasma", "CD8 T (GZMK+)": "T / NK"}
    genes = ["EPCAM", "FBN1", "MS4A1", "CD3E", "CCL19", "CCR7"]
    marker = {"Epithelial": "EPCAM", "Stromal": "FBN1", "B / plasma": "MS4A1", "T / NK": "CD3E"}
    X = np.zeros((len(xy), len(genes)))
    for i, ct in enumerate(cell_type):
        X[i, genes.index(marker[lineage[ct]])] = rng.poisson(5) + 1
    obs = pd.DataFrame({"cell_type": pd.Categorical(cell_type), "lineage": pd.Categorical([lineage[c] for c in cell_type]),
                        "annotation_confidence": "high", "cell_area": rng.uniform(40, 120, len(xy))},
                       index=[f"c{i}" for i in range(len(xy))])
    a = ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = xy
    return a

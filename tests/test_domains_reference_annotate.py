import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ist_analysis import annotate, domains, reference


def test_name_domains_orders_by_epithelial_share():
    lineage = pd.Series(["Epithelial"] * 9 + ["T / NK"] + ["Stromal"] * 6 + ["Endothelial"] * 4)
    labels = pd.Series(["b"] * 10 + ["a"] * 10)
    named = domains.name_domains(labels, lineage, ["Epithelial", "T / NK", "Stromal", "Endothelial"])
    assert named.iloc[0] == "D1: Epithelial 90%"
    assert named.iloc[-1] == "D2: Stromal 60% + Endothelial 40%"


def test_compartment_majority():
    lineage = pd.Series(["T / NK", "B / plasma", "Epithelial", "Stromal", "Stromal", "Endothelial"])
    labels = pd.Series(["x", "x", "x", "y", "y", "y"])
    assert list(domains.compartment(labels, lineage)) == ["immune"] * 3 + ["stroma / vessels"] * 3


def test_foreign_genes_flags_other_lineage_markers():
    X = np.array([[5, 0, 1], [6, 0, 1], [0, 5, 1], [0, 6, 1]], dtype=float)   # gene0 = myeloid, gene1 = epithelial, gene2 = shared
    a = ad.AnnData(sp.csr_matrix(X), var=pd.DataFrame(index=["CD68", "EPCAM", "ACTB"]))
    a.layers["lognorm"] = sp.csr_matrix(np.log1p(X))
    lineage = pd.Series(["Myeloid", "Myeloid", "Epithelial", "Epithelial"], index=a.obs_names)
    f = domains.foreign_genes(a, lineage)
    assert f["EPCAM"] == "Epithelial"
    assert pd.isna(f["CD68"]) and pd.isna(f["ACTB"])


def test_call_genes_decision_rules():
    r10 = pd.DataFrame({"baseMean": 10, "log2FoldChange": [2, 2, 2, 0.1], "padj": [1e-4, 1e-4, 1e-4, 0.9]}, index=list("ABCD"))
    rP = pd.DataFrame({"log2FoldChange": [1.5, 1.5, -1.0, 0.0], "padj": [1e-3, 1e-3, 1e-3, 0.9]}, index=list("ABCD"))
    foreign = pd.Series({"B": "Stromal"})
    calls = domains.call_genes(r10, rP, foreign)["call"]
    assert calls.to_dict() == {"A": "macrophage-intrinsic", "B": "spillover (foreign gene)", "C": "not confirmed with Proseg", "D": "n.s."}


def test_blocks():
    b = domains.blocks(np.array([[10, 10], [999, 10], [1001, 2500]]), 1000, ["a", "b", "c"])
    assert list(b) == ["b0_0", "b0_0", "b1_2"]


def test_lognorm_scales_to_10k():
    X = sp.csr_matrix(np.array([[1, 3, 0], [0, 0, 0], [10, 10, 20]], dtype=float))
    a = ad.AnnData(X, var=pd.DataFrame(index=["g1", "g2", "g3"]))
    L = reference.lognorm(a, ["g1", "g2", "g3"]).toarray()
    assert np.allclose(np.expm1(L[0]).sum(), 1e4) and np.allclose(np.expm1(L[2]).sum(), 1e4)
    assert np.allclose(L[1], 0)                                         # empty cells stay empty, no division by zero


def test_agreement_uses_confirmed_types():
    manual = pd.DataFrame({"cell_type": ["B cell"] * 4, "lineage": ["B / plasma"] * 4}, index=list("abcd"))
    atlas = pd.DataFrame({"atlas_label": ["B cell", "B cell", "B cell", "Plasma cell"], "atlas_prob": 0.9}, index=list("abcd"))
    out = reference.agreement(atlas, manual, manual.index, ceiling=pd.Series({"B cell": 0.95}))
    assert out.loc["B cell", "% confirmed by atlas"] == 75


def test_apply_labels_guard_fails_loudly():
    sub = ad.AnnData(np.zeros((4, 2)))
    sub.obs["sub"] = pd.Categorical(["0", "0", "1", "1"])
    names = np.array([("GZMK", "FOXP3"), ("CD3E", "IL2RA")] + [("X", "Y")] * 8, dtype=[("0", "O"), ("1", "O")])
    sub.uns["sub_markers"] = {"names": names}
    good = {"0": ["CD8 T", "high", "GZMK"], "1": ["Treg", "high", "FOXP3"]}
    table = annotate.apply_labels(sub, good)
    assert list(table["label"]) == ["CD8 T", "Treg"]
    with pytest.raises(ValueError, match="guard"):
        annotate.apply_labels(sub, {"0": ["CD8 T", "high", "GZMK"], "1": ["Treg", "high", "MKI67"]})

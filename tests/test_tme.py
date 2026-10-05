import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay

from ist_analysis import spatial, tme


def delaunay_adjacency(xy):
    tri = Delaunay(xy)
    rows, cols = [], []
    for s in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    rows.append(s[i]); cols.append(s[j])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    A.data[:] = 1
    return A


def test_positivity_and_rates(tissue):
    pos = tme.Positivity(tissue)
    assert pos("EPCAM").sum() == (tissue.obs["lineage"] == "Epithelial").sum()
    rates = tme.positivity_rates(pos, tissue.obs["cell_type"].astype(str).to_numpy(), {"MS4A1": "CD20"})
    assert rates.loc["B cell", "MS4A1 (CD20)"] == 100
    assert rates.loc["Fibroblast", "MS4A1 (CD20)"] == 0


def test_contact_test_detects_planted_contacts(tissue, rng):
    """CCL19 on B cells and CCR7 on the T-cell rim around them: contacts far above the within-type null."""
    a = tissue
    X = a.layers["counts"].toarray()
    b = (a.obs["cell_type"] == "B cell").to_numpy()
    t = (a.obs["cell_type"] == "CD8 T (GZMK+)").to_numpy()
    # half of each type positive at random, but only those near the other type: sender/receiver in contact
    X[b, list(a.var_names).index("CCL19")] = 1
    X[t, list(a.var_names).index("CCR7")] = 1
    a.layers["counts"] = sp.csr_matrix(X)
    a.obsp["spatial_connectivities"] = delaunay_adjacency(a.obsm["spatial"])
    pairs = [("CCL19", "CCR7", ["B / plasma"], ["T / NK"], "test")]
    lr = tme.contact_test(a, tme.Positivity(a), pairs=pairs, n_perms=50)
    # all senders / receivers of a type are positive, so shuffling within type changes nothing: obs == expected
    assert np.isclose(lr["obs / exp"].iloc[0], 1.0)

    # now only a random half of the rim T cells is CCR7+, and only B cells touching T cells are CCL19+:
    # the shuffle within type breaks the co-location, so observed contacts exceed the null
    X[:, list(a.var_names).index("CCR7")] = 0
    X[np.flatnonzero(t)[rng.random(t.sum()) < 0.5], list(a.var_names).index("CCR7")] = 1
    A = a.obsp["spatial_connectivities"]
    touches_t = np.asarray(A[:, np.flatnonzero(t)].sum(axis=1)).ravel() > 0
    X[:, list(a.var_names).index("CCL19")] = (b & touches_t).astype(float)
    a.layers["counts"] = sp.csr_matrix(X)
    lr = tme.contact_test(a, tme.Positivity(a), pairs=pairs, n_perms=200)
    assert lr["obs / exp"].iloc[0] > 1.2
    assert lr["p (perm)"].iloc[0] < 0.05


def test_classify_tiles():
    tiles = pd.DataFrame({"total_density": [10, 200, 200], "intra_density": [0, 150, 20], "stroma_density": [5, 100, 200]})
    assert list(tme.classify_tiles(tiles, desert=50, ratio=0.5)) == ["desert", "inflamed", "excluded"]


def test_cd8_compartments(tissue):
    spatial.distance_to_tumour(tissue)
    spatial.find_tls(tissue)
    tme.add_cd8_compartments(tissue)
    cd8 = tissue.obs["cell_type"] == "CD8 T (GZMK+)"
    assert set(tissue.obs.loc[cd8, "cd8_compartment"]) <= {"TLS", "Stroma (> 50 um)"}
    assert (tissue.obs.loc[tissue.obs["lineage"] == "Epithelial", "cd8_compartment"] == "Tumour contact (<= 15 um)").all()

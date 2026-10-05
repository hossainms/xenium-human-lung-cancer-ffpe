import numpy as np
import pandas as pd

from ist_analysis import spatial


def test_distance_to_tumour_is_zero_for_tumour_cells(tissue):
    is_tumour = spatial.distance_to_tumour(tissue)
    d = tissue.obs["dist_to_tumour_um"].to_numpy()
    assert (d[is_tumour] == 0).all()
    assert (d[~is_tumour] > 0).all()


def test_find_tls_recovers_planted_aggregates(tissue):
    table, in_aggregates, n_b = spatial.find_tls(tissue, eps_um=25, min_cells=20)
    assert len(table) == 2
    centres = table[["x (mm)", "y (mm)"]].to_numpy() * 1000
    planted = np.array([[1400, 300], [1700, 750]])
    assert all(np.min(np.linalg.norm(planted - c, axis=1)) < 10 for c in centres)
    assert in_aggregates == n_b                                      # every planted B cell is found
    assert (table["% tumour"] == 0).all()


def test_infiltration_flags_excluded_cells(tissue):
    is_tumour = spatial.distance_to_tumour(tissue)
    infil, _, base_contact = spatial.infiltration_table(tissue, is_tumour, contact_um=15, types=["CD8 T (GZMK+)", "B cell"])
    assert infil.loc["CD8 T (GZMK+)", "% within 15 um"] == 0          # planted far from the tumour
    assert infil.loc["CD8 T (GZMK+)", "vs. baseline"] == "excluded"


def test_niche_name():
    assert spatial.niche_name(pd.Series({"Epithelial": 0.9, "T / NK": 0.1})) == "Epithelial 90%"
    assert spatial.niche_name(pd.Series({"Epithelial": 0.3, "T / NK": 0.6, "B / plasma": 0.1})) == "T / NK 60% + Epithelial 30%"


def test_composition_niches_separates_tumour_and_stroma(tissue):
    centres, _ = spatial.composition_niches(tissue, radius_um=50, n_niches=3)
    assert centres.index[0].startswith("N1: Epithelial")              # niches ordered by epithelial share
    assert centres["cells"].sum() == tissue.n_obs

import numpy as np
import pandas as pd

from ist_analysis import qc
from ist_analysis.utils import display_path, mad_bounds


def test_mad_bounds_are_symmetric_on_log_scale(rng):
    x = np.exp(rng.normal(4, 0.5, 20000)) - 1
    lo, hi = mad_bounds(x, 3)
    med = np.median(np.log1p(x))
    assert np.isclose(np.log1p(hi) - med, med - np.log1p(lo), rtol=1e-6)
    assert np.isclose(np.log1p(hi) - med, 3 * 0.5, rtol=0.05)   # MAD * 1.4826 ~ SD for normal data


def test_qc_flags_each_rule(rng):
    n = 1000
    obs = pd.DataFrame({"transcript_counts": rng.integers(20, 200, n), "n_genes": rng.integers(10, 60, n),
                        "cell_area": np.exp(rng.normal(4, 0.3, n)), "control_frac": np.zeros(n)})
    obs.iloc[0, 0] = 5          # too few counts
    obs.iloc[1, 1] = 2          # too few genes
    obs.iloc[2, 2] = 1e5        # huge area
    obs.iloc[3, 3] = 0.2        # background
    flags, (lo, hi) = qc.qc_flags(obs, min_counts=10, min_genes=5, area_nmads=3, max_control_frac=0.05)
    assert flags.iloc[0, 0] and flags.iloc[1, 1] and flags.iloc[2, 3] and flags.iloc[3, 4]
    assert flags.iloc[4:].to_numpy().sum() <= 0.01 * n                  # almost nothing else removed
    table = qc.rule_table(flags)
    assert table["cells flagged"].sum() == flags.to_numpy().sum()


def test_assign_fov_first_match_wins():
    fov = pd.DataFrame({"x": [0, 90], "y": [0, 0], "width": [100, 100], "height": [100, 100]}, index=["A1", "A2"])
    xy = np.array([[10, 10], [95, 50], [150, 50], [500, 500]])
    assert list(qc.assign_fov(xy, fov)) == ["A1", "A1", "A2", ""]


def test_display_path_hides_home(tmp_path):
    from pathlib import Path

    assert display_path(Path.home() / "data" / "x.h5ad") == "~/data/x.h5ad"
    assert display_path(Path("/not/home/x.h5ad")) == "x.h5ad"


def test_add_qc_metrics_uses_the_platform_control_columns():
    """The platform adapter names its negative-control columns; QC metrics do not hard-code Xenium's."""
    import anndata as ad
    import scipy.sparse as sp

    X = sp.csr_matrix(np.array([[4, 0, 6], [10, 10, 0]], dtype=float))
    obs = pd.DataFrame({"transcript_counts": [10, 20], "cell_area": [50.0, 100.0], "nucleus_area": [20.0, np.nan],
                        "neg_probe": [1, 0], "blank_code": [0, 2]}, index=["a", "b"])
    a = ad.AnnData(X, obs=obs)
    qc.add_qc_metrics(a, control_columns=("neg_probe", "blank_code"))
    assert list(a.obs["control_counts"]) == [1, 2]
    assert np.allclose(a.obs["control_frac"], [0.1, 0.1])        # controls / gene counts
    assert np.allclose(a.obs["density"], [0.2, 0.2])
    assert np.isnan(a.obs["nucleus_ratio"].iloc[1])


def test_project_constants_are_exposed():
    """Notebooks read these from ist_analysis.utils; a lint auto-fix once removed the re-export."""
    from ist_analysis import utils
    from ist_analysis.io import xenium

    assert utils.PIXEL_SIZE_UM == xenium.PIXEL_SIZE_UM == 0.2125
    assert len(utils.TUMOUR_TYPES) == 4 and len(utils.LINEAGE_COLORS) == 7
    assert utils.ANNOTATION_FILE.name == "annotation.yaml"

"""10x Genomics Xenium adapter: everything that depends on the Xenium output bundle.

The analysis modules (qc, cluster, annotate, spatial, tme, domains, ...) work on a standard AnnData:
counts in layers['counts'], cell centroids in um in obsm['spatial'], per-cell areas and control counts in .obs.
This module turns a Xenium `outs/` folder into that, and reads the Xenium-specific side files. Supporting
another imaging-based platform (MERSCOPE, CosMx) means adding a sibling module with the same functions.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

PLATFORM = "xenium"
PIXEL_SIZE_UM = 0.2125                                        # morphology image: um per pixel
CONTROL_COLUMNS = ("control_probe_counts", "control_codeword_counts")   # negative controls in the cell table
NON_GENE_FEATURES = "^(NegControl|Unassigned|Deprecated)"     # non-gene features in transcripts.parquet
MIN_QV = 20                                                   # Xenium's transcript quality cutoff
PROSEG_PRESET = "--xenium"                                    # Proseg's input preset for this platform


def load_cell_table(outs_dir: Path):
    """The cell x gene table (spatialdata-io), without images, transcripts or shapes; raw counts in layers['counts']."""
    from spatialdata_io import xenium

    sdata = xenium(outs_dir, morphology_focus=False, cells_labels=False, nucleus_labels=False,
                   transcripts=False, cells_boundaries=False, nucleus_boundaries=False)
    adata = sdata.tables["table"].copy()
    adata.layers["counts"] = adata.X.copy()
    return adata


def fov_table(outs_dir: Path) -> pd.DataFrame:
    """Imaging tiles (fields of view) with positions in um (aux_outputs), plus the fraction of gene transcripts with qv >= 20."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    with tarfile.open(outs_dir / "aux_outputs.tar.gz") as tf:
        fov_json = json.load(tf.extractfile("aux_outputs/morphology_fov_locations.json"))
    assert fov_json["units"] == "micron"
    fov = pd.DataFrame(fov_json["fov_locations"]).T.rename_axis("fov")
    tx = pq.read_table(outs_dir / "transcripts.parquet", columns=["fov_name", "qv", "feature_name"])
    is_gene = ~pc.match_substring_regex(tx["feature_name"], NON_GENE_FEATURES).to_numpy(zero_copy_only=False)
    q20 = pd.DataFrame({"fov": tx["fov_name"].to_numpy(zero_copy_only=False), "q20": tx["qv"].to_numpy() >= MIN_QV})[is_gene]
    fov["frac_q20"] = q20.groupby("fov")["q20"].mean()
    return fov


def gene_detection(outs_dir: Path, counts=None, nucleus_fraction: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Total counts per gene vs negative-control probes (background); optional % of cells detected and % in nucleus.

    Returns the gene table (weakest first) and the negative-control probe totals.
    """
    import scanpy as sc

    full = sc.read_10x_h5(outs_dir / "cell_feature_matrix.h5", gex_only=False)
    totals = pd.Series(np.asarray(full.X.sum(axis=0)).ravel(), index=full.var_names)
    ftype = full.var["feature_types"]
    neg = totals[ftype == "Negative Control Probe"]
    genes = pd.DataFrame({"total_counts": totals[ftype == "Gene Expression"]})
    if counts is not None:
        genes["pct_cells_detected"] = 100 * np.asarray((counts > 0).mean(axis=0)).ravel()
    genes["fold_over_background"] = genes["total_counts"] / neg.mean()
    if nucleus_fraction:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        tx = pq.read_table(outs_dir / "transcripts.parquet", columns=["feature_name", "qv", "cell_id", "overlaps_nucleus"])
        keep = (pc.greater_equal(tx["qv"], MIN_QV).to_numpy(zero_copy_only=False)
                & pc.not_equal(tx["cell_id"], "UNASSIGNED").to_numpy(zero_copy_only=False))
        nuc = pd.DataFrame({"gene": tx["feature_name"].to_numpy(zero_copy_only=False)[keep],
                            "in_nucleus": tx["overlaps_nucleus"].to_numpy()[keep].astype(bool)})
        genes["pct_in_nucleus"] = 100 * nuc.groupby("gene")["in_nucleus"].mean()
    return genes.sort_values("total_counts"), neg


def vendor_clusters(outs_dir: Path) -> pd.Series:
    """10x's own graph-based clustering (analysis.tar.gz), for benchmarking."""
    with tarfile.open(outs_dir / "analysis.tar.gz") as tf:
        return pd.read_csv(tf.extractfile("analysis/clustering/gene_expression_graphclust/clusters.csv"), index_col="Barcode")["Cluster"]


def gene_count_matrix(outs_dir: Path):
    """The 10x cell x gene matrix (gene features only), e.g. for comparing segmentations."""
    import scanpy as sc

    return sc.read_10x_h5(outs_dir / "cell_feature_matrix.h5")

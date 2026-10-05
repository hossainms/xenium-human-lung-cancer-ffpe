"""Steps 3-4: load the Xenium cell table and apply quality control.

Same metrics and thresholds as the notebook (Step 4). Writes the QC'd AnnData (raw counts in .X and
layers['counts'], QC metrics + imaging tile in .obs, centroids in um in .obsm['spatial']), a
per-rule table, a per-tile table and a per-gene detection table.
"""

import json
import tarfile

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scanpy as sc
from spatialdata_io import xenium

from common import Step, mad_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Steps 3-4: load and quality control", inputs=["marker"], outputs=["output"])
cfg = step.config["qc"]
outs = step.outs_dir

# ---- Load only the cell table (images, transcripts and shapes are not needed here) ---------
sdata = xenium(outs, morphology_focus=False, cells_labels=False, nucleus_labels=False,
               transcripts=False, cells_boundaries=False, nucleus_boundaries=False)
adata = sdata.tables["table"].copy()
adata.layers["counts"] = adata.X.copy()
step.say(f"  loaded {adata.n_obs:,} cells x {adata.n_vars} genes")

# ---- Per-cell metrics --------------------------------------------------------------------------
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
obs = adata.obs
obs["n_genes"] = obs["n_genes_by_counts"]
obs["nucleus_ratio"] = obs["nucleus_area"] / obs["cell_area"]
obs["density"] = obs["transcript_counts"] / obs["cell_area"]
obs["control_counts"] = obs["control_probe_counts"] + obs["control_codeword_counts"]
obs["control_frac"] = obs["control_counts"] / obs["total_counts"].replace(0, np.nan)   # denominator: gene counts

# ---- Rules: fixed floors for counts / genes, MAD bounds for area -------------------------------
area_lo, area_hi = mad_bounds(obs["cell_area"].to_numpy(), cfg["area_nmads"])
flags = pd.DataFrame({
    f"counts < {cfg['min_counts']}": obs["transcript_counts"] < cfg["min_counts"],
    f"genes < {cfg['min_genes']}": obs["n_genes"] < cfg["min_genes"],
    f"area < {area_lo:.1f} um2": obs["cell_area"] < area_lo,
    f"area > {area_hi:.0f} um2": obs["cell_area"] > area_hi,
    f"control frac > {cfg['max_control_frac']:.0%}": obs["control_frac"].fillna(0) > cfg["max_control_frac"],
}, index=obs.index)
obs["qc_pass"] = ~flags.any(axis=1)
rules = pd.DataFrame({"cells flagged": flags.sum(), "% of cells": (flags.mean() * 100).round(2),
                      "flagged only by this rule": (flags & (flags.sum(axis=1) == 1).to_numpy()[:, None]).sum()})
step.save_table(rules, "step04_qc_rules")
step.say(f"  area bounds {area_lo:.1f}-{area_hi:.0f} um2; pass {obs['qc_pass'].sum():,} of {adata.n_obs:,} "
         f"({obs['qc_pass'].mean():.1%})")

# ---- Distributions with the cutoffs --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, (col, label, cuts) in zip(axes, [("transcript_counts", "Q20 transcripts per cell", [cfg["min_counts"]]),
                                         ("n_genes", "Genes detected per cell", [cfg["min_genes"]]),
                                         ("cell_area", "Cell area (um^2)", [area_lo, area_hi])]):
    x = obs[col].to_numpy()
    x = x[x > 0]
    ax.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 80), color="#9aa5b1")
    ax.set_xscale("log")
    for c in cuts:
        ax.axvline(c, color="#d1495b", ls="--")
    ax.set_title(label, loc="left")
    ax.set_yticks([])
step.save_fig(fig, "step04_qc_distributions")

# ---- Imaging tiles: assign each cell to the tile under its centroid --------------------------------
with tarfile.open(outs / "aux_outputs.tar.gz") as tf:
    fov = pd.DataFrame(json.load(tf.extractfile("aux_outputs/morphology_fov_locations.json"))["fov_locations"]).T
tx = pq.read_table(outs / "transcripts.parquet", columns=["fov_name", "qv", "feature_name"])
is_gene = ~pc.match_substring_regex(tx["feature_name"], "^(NegControl|Unassigned|Deprecated)").to_numpy(zero_copy_only=False)
q20 = pd.DataFrame({"fov": tx["fov_name"].to_numpy(zero_copy_only=False), "q20": tx["qv"].to_numpy() >= 20})[is_gene]
fov["frac_q20"] = q20.groupby("fov")["q20"].mean()
del tx, q20
xy = adata.obsm["spatial"]
cell_fov = np.full(adata.n_obs, "", dtype=object)
for name, r in fov.iterrows():
    inside = (xy[:, 0] >= r.x) & (xy[:, 0] < r.x + r.width) & (xy[:, 1] >= r.y) & (xy[:, 1] < r.y + r.height) & (cell_fov == "")
    cell_fov[inside] = name
obs["fov"] = pd.Categorical(cell_fov)
fov = fov.join(obs.groupby("fov", observed=True).agg(cells=("qc_pass", "size"), median_counts=("transcript_counts", "median"),
                                                     pct_removed=("qc_pass", lambda s: 100 * (1 - s.mean()))))
robust_z = lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median())   # noqa: E731
big = fov["cells"] >= 200
fov["outlier"] = False
fov.loc[big, "outlier"] = (robust_z(fov.loc[big, "median_counts"]).abs() > 3) | (robust_z(fov.loc[big, "frac_q20"]).abs() > 3)
step.save_table(fov, "step04_qc_tiles")
step.say(f"  {len(fov)} imaging tiles, {int(fov['outlier'].sum())} outliers: {', '.join(fov.index[fov['outlier']])}")

# ---- Genes vs negative-control background ------------------------------------------------------------
full = sc.read_10x_h5(outs / "cell_feature_matrix.h5", gex_only=False)
totals = pd.Series(np.asarray(full.X.sum(axis=0)).ravel(), index=full.var_names)
neg = totals[full.var["feature_types"] == "Negative Control Probe"]
genes = pd.DataFrame({"total_counts": totals[full.var["feature_types"] == "Gene Expression"]})
genes["fold_over_background"] = genes["total_counts"] / neg.mean()
genes = genes.sort_values("total_counts")
step.save_table(genes, "step04_gene_detection")
step.say(f"  weakest gene {genes.index[0]}: {genes.total_counts.iloc[0]:.0f} counts "
         f"({genes.fold_over_background.iloc[0]:.0f}x background)")

# ---- Filter and save -------------------------------------------------------------------------------------
adata_qc = adata[obs["qc_pass"].to_numpy()].copy()
adata_qc.uns["qc_params"] = {"min_counts": cfg["min_counts"], "min_genes": cfg["min_genes"], "area_nmads": cfg["area_nmads"],
                             "area_bounds_um2": [area_lo, area_hi], "max_control_frac": cfg["max_control_frac"],
                             "count_filter": "gene transcripts with qv >= 20 assigned to a cell (10x default)"}
adata_qc.obs = adata_qc.obs.drop(columns=["qc_pass"])
adata_qc.uns.pop("spatialdata_attrs", None)
out = step.path("output")
adata_qc.write_h5ad(out, compression="gzip")
step.say(f"  kept {adata_qc.n_obs:,} cells")
step.done(out)

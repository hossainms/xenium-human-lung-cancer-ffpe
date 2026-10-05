"""Steps 3-4: load the Xenium cell table and apply quality control (ist_analysis.qc)."""

import matplotlib
import numpy as np

from common import Step
from ist_analysis import qc
from ist_analysis.io import xenium  # platform adapter: everything specific to the Xenium bundle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Steps 3-4: load and quality control", inputs=["marker"], outputs=["output"])
cfg = step.config["qc"]

adata = xenium.load_cell_table(step.outs_dir)
step.say(f"  loaded {adata.n_obs:,} cells x {adata.n_vars} genes")
qc.add_qc_metrics(adata, control_columns=xenium.CONTROL_COLUMNS)
obs = adata.obs
flags, (area_lo, area_hi) = qc.qc_flags(obs, cfg["min_counts"], cfg["min_genes"], cfg["area_nmads"], cfg["max_control_frac"])
obs["qc_pass"] = ~flags.any(axis=1)
step.save_table(qc.rule_table(flags).round(2), "step04_qc_rules")
step.say(f"  area bounds {area_lo:.1f}-{area_hi:.0f} um2; pass {obs['qc_pass'].sum():,} of {adata.n_obs:,} ({obs['qc_pass'].mean():.1%})")

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

fov = xenium.fov_table(step.outs_dir)
obs["fov"] = qc.assign_fov(adata.obsm["spatial"], fov)
obs["fov"] = obs["fov"].astype("category")
fov = qc.fov_outliers(fov, obs)
step.save_table(fov, "step04_qc_tiles")
step.say(f"  {len(fov)} imaging tiles, {int(fov['outlier'].sum())} outliers: {', '.join(fov.index[fov['outlier']])}")

genes, _ = xenium.gene_detection(step.outs_dir)
step.save_table(genes, "step04_gene_detection")
step.say(f"  weakest gene {genes.index[0]}: {genes.total_counts.iloc[0]:.0f} counts ({genes.fold_over_background.iloc[0]:.0f}x background)")

params = {"min_counts": cfg["min_counts"], "min_genes": cfg["min_genes"], "area_nmads": cfg["area_nmads"],
          "area_bounds_um2": [area_lo, area_hi], "max_control_frac": cfg["max_control_frac"],
          "count_filter": "gene transcripts with qv >= 20 assigned to a cell (10x default)"}
adata_qc = qc.filter_cells(adata, obs["qc_pass"].to_numpy(), params)
out = step.path("output")
adata_qc.write_h5ad(out, compression="gzip")
step.say(f"  kept {adata_qc.n_obs:,} cells")
step.done(out)

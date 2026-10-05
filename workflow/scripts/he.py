"""Step 8: H&E alignment check against DAPI, histology windows, per-cell stain features (ist_analysis.he)."""

import matplotlib
import scanpy as sc

from common import Step, xu
from ist_analysis import he

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 8: H&E alignment and per-cell features", inputs=["input"], outputs=["output"])
cfg = step.config["he"]
sample = step.config["sample"]
HE = step.sample_dir / f"{sample}_he_image.ome.tif"
DAPI = step.outs_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
M = he.load_alignment(step.sample_dir / f"{sample}_he_imagealignment.csv")

qc, corr, he_grid, _, shifts = he.alignment_qc(HE, DAPI, M, cfg["qc_res_um"])
step.save_table(qc.round(3).to_frame("value"), "step08_alignment_qc")
step.say(f"  hematoxylin vs DAPI: r = {qc.iloc[0]:.2f} at zero shift, best {qc.iloc[1]:.2f} at ({qc.iloc[2]:+.0f}, {qc.iloc[3]:+.0f}) um")
res = cfg["qc_res_um"]
fig, axes = plt.subplots(1, 2, figsize=(17, 4.3), gridspec_kw={"width_ratios": [3.2, 1]})
axes[0].imshow(he_grid, extent=[0, he_grid.shape[1] * res / 1000, he_grid.shape[0] * res / 1000, 0])
axes[0].set_title("H&E warped into the Xenium frame", loc="left")
lim = shifts[-1] * res
im = axes[1].imshow(corr, cmap="Blues", extent=[-lim, lim, lim, -lim])
fig.colorbar(im, ax=axes[1], label="r")
axes[1].set_title("Hematoxylin vs DAPI by shift (um)", loc="left")
step.save_fig(fig, "step08_alignment")

adata = sc.read_h5ad(step.path("input"))
xy = adata.obsm["spatial"]
W = 400
regions = he.histology_windows(adata, W)
fig, axes = plt.subplots(len(regions), 2, figsize=(12, 6 * len(regions)))
for row, (name, (cx, cy)) in zip(axes, regions.items()):
    img, (x0, y0) = he.he_crop(HE, M, cx, cy, W)
    ext = [x0, x0 + W, y0 + W, y0]
    row[0].imshow(img, extent=ext)
    row[1].imshow(img, extent=ext, alpha=0.55)
    inside = (xy[:, 0] > x0) & (xy[:, 0] < x0 + W) & (xy[:, 1] > y0) & (xy[:, 1] < y0 + W)
    for lin, col in xu.LINEAGE_COLORS.items():
        m = inside & (adata.obs["lineage"] == lin).to_numpy()
        row[1].scatter(xy[m, 0], xy[m, 1], s=9, c=col, edgecolors="white", linewidths=0.3, label=lin)
    row[0].set_title(f"{name.split(' (')[0]}: H&E", loc="left")
    row[1].set_title("H&E + Xenium lineages", loc="left")
    for ax in row:
        ax.set_xlim(x0, x0 + W); ax.set_ylim(y0 + W, y0); ax.set_xticks([]); ax.set_yticks([])
axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
step.save_fig(fig, "step08_histology_windows")

he.stain_features(adata, HE, M, cfg["feature_level"])
feat = adata.obs.groupby("lineage", observed=True)[["he_hematoxylin", "he_eosin"]].median().sort_values("he_hematoxylin")
step.save_table(feat.round(4), "step08_stain_by_lineage")
step.say(f"  hematoxylin highest under {feat.index[-1]}, lowest under {feat.index[0]}")

adata.uns["step8_params"] = {"alignment": "he_imagealignment.csv (H&E px -> Xenium px)", "he_feature_level": cfg["feature_level"],
                             "qc_r_zero_shift": float(qc.iloc[0])}
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

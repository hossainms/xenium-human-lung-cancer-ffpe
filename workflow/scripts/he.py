"""Step 8: H&E alignment check and per-cell H&E stain features.

The 10x matrix (he_imagealignment.csv) maps H&E pixels -> Xenium pixels. Checks it by correlating the
hematoxylin channel (colour deconvolution) with Xenium DAPI at shifts of +/-40 um, shows three
histology windows with the cell lineages on top, and stores hematoxylin / eosin under every cell.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import zarr
from skimage.color import rgb2hed
from skimage.transform import warp

from common import Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 8: H&E alignment and per-cell features", inputs=["input"], outputs=["output"])
cfg = step.config["he"]
sample = step.config["sample"]
HE = step.sample_dir / f"{sample}_he_image.ome.tif"
DAPI = step.outs_dir / "morphology_focus" / "morphology_focus_0000.ome.tif"
M = np.loadtxt(step.sample_dir / f"{sample}_he_imagealignment.csv", delimiter=",")   # H&E px -> Xenium px
PX = xu.PIXEL_SIZE_UM
scale = lambda sx, sy: np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])   # noqa: E731

with tifffile.TiffFile(HE) as tif:
    H0, W0 = tif.series[0].levels[0].shape[:2]


def he_level(level):
    with tifffile.TiffFile(HE) as tif:
        return tif.series[0].levels[level].asarray()


# ---- Alignment QC on a 4 um grid: hematoxylin vs DAPI at shifts of +/-40 um ------------------------
res = cfg["qc_res_um"]
with tifffile.TiffFile(DAPI) as tif:
    Hd0, Wd0 = tif.series[0].levels[0].shape[-2:]
    dapi = tif.series[0].levels[3].asarray()[0].astype(float)
shape = (int(np.ceil(Hd0 * PX / res)), int(np.ceil(Wd0 * PX / res)))
dapi_grid = warp(dapi, scale(dapi.shape[1] / Wd0, dapi.shape[0] / Hd0) @ scale(1 / PX, 1 / PX) @ scale(res, res),
                 output_shape=shape, order=1, preserve_range=True)
img3 = he_level(3)
T = scale(img3.shape[1] / W0, img3.shape[0] / H0) @ np.linalg.inv(M) @ scale(1 / PX, 1 / PX) @ scale(res, res)
he_grid = warp(img3, T, output_shape=shape, order=1, preserve_range=True, cval=255).astype(np.uint8)
hema = rgb2hed(he_grid)[..., 0]
dapi_log = np.log1p(dapi_grid)
tissue = (dapi_grid > np.percentile(dapi_grid, 50)) | (hema > np.percentile(hema, 50))
shifts = np.arange(-10, 11)
corr = np.array([[np.corrcoef(dapi_log[tissue], np.roll(np.roll(hema, dy, 0), dx, 1)[tissue])[0, 1] for dx in shifts] for dy in shifts])
iy, ix = np.unravel_index(corr.argmax(), corr.shape)
qc = pd.Series({"r at zero shift": corr[10, 10], "best r": corr.max(), "best shift x (um)": shifts[ix] * res,
                "best shift y (um)": shifts[iy] * res, "r at 40 um (mean of 4)": np.mean([corr[0, 10], corr[20, 10], corr[10, 0], corr[10, 20]])})
step.save_table(qc.round(3).to_frame("value"), "step08_alignment_qc")
step.say(f"  hematoxylin vs DAPI: r = {qc.iloc[0]:.2f} at zero shift, best {qc.iloc[1]:.2f} at "
         f"({qc.iloc[2]:+.0f}, {qc.iloc[3]:+.0f}) um")
fig, axes = plt.subplots(1, 2, figsize=(17, 4.3), gridspec_kw={"width_ratios": [3.2, 1]})
axes[0].imshow(he_grid, extent=[0, shape[1] * res / 1000, shape[0] * res / 1000, 0])
axes[0].set_title("H&E warped into the Xenium frame", loc="left")
im = axes[1].imshow(corr, cmap="Blues", extent=[-40, 40, 40, -40])
fig.colorbar(im, ax=axes[1], label="r")
axes[1].set_title("Hematoxylin vs DAPI by shift (um)", loc="left")
step.save_fig(fig, "step08_alignment")
del dapi, dapi_grid, img3, he_grid, hema

# ---- Cells: three histology windows chosen from the data ----------------------------------------------
adata = sc.read_h5ad(step.path("input"))
xy = adata.obsm["spatial"]
W, CROP_RES = 400, 0.5


def he_crop(cx, cy, level=1):
    x0, y0 = cx - W / 2, cy - W / 2
    with tifffile.TiffFile(HE) as tif:
        h_l, w_l = tif.series[0].levels[level].shape[:2]
    to_he = scale(w_l / W0, h_l / H0) @ np.linalg.inv(M) @ scale(1 / PX, 1 / PX)
    c = to_he @ np.array([[x0, x0 + W, x0, x0 + W], [y0, y0, y0 + W, y0 + W], [1, 1, 1, 1]])
    lo = np.maximum(np.floor(c[:2].min(axis=1)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(c[:2].max(axis=1)).astype(int) + 2, [w_l, h_l])
    store = tifffile.imread(HE, aszarr=True, level=level)
    patch = np.asarray(zarr.open(store, mode="r")[lo[1]:hi[1], lo[0]:hi[0]])
    store.close()
    Tc = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]]) @ to_he @ np.array([[CROP_RES, 0, x0], [0, CROP_RES, y0], [0, 0, 1]])
    n = int(W / CROP_RES)
    return warp(patch, Tc, output_shape=(n, n), order=1, preserve_range=True, cval=255).astype(np.uint8), (x0, y0)


b = np.floor(xy / W).astype(int)
keys, inv = np.unique(b, axis=0, return_inverse=True)
epi = (adata.obs["lineage"] == "Epithelial").to_numpy()
strom = (adata.obs["lineage"] == "Stromal").to_numpy()
my_niche = adata.obs["niche"].astype(str).str.contains(r"^N\d+: Myeloid").to_numpy()
regions = {"TLS1": xy[(adata.obs["tls_id"] == "TLS1").to_numpy()].mean(axis=0),
           "Myeloid-rich niche": (keys[np.bincount(inv, weights=my_niche).argmax()] + 0.5) * W,
           "Tumour-stroma border": (keys[np.minimum(np.bincount(inv, weights=epi), np.bincount(inv, weights=strom)).argmax()] + 0.5) * W}
fig, axes = plt.subplots(len(regions), 2, figsize=(12, 6 * len(regions)))
for row, (name, (cx, cy)) in zip(axes, regions.items()):
    img, (x0, y0) = he_crop(cx, cy)
    ext = [x0, x0 + W, y0 + W, y0]
    row[0].imshow(img, extent=ext)
    row[1].imshow(img, extent=ext, alpha=0.55)
    inside = (xy[:, 0] > x0) & (xy[:, 0] < x0 + W) & (xy[:, 1] > y0) & (xy[:, 1] < y0 + W)
    for lin, col in xu.LINEAGE_COLORS.items():
        m = inside & (adata.obs["lineage"] == lin).to_numpy()
        row[1].scatter(xy[m, 0], xy[m, 1], s=9, c=col, edgecolors="white", linewidths=0.3, label=lin)
    row[0].set_title(f"{name}: H&E", loc="left")
    row[1].set_title("H&E + Xenium lineages", loc="left")
    for ax in row:
        ax.set_xlim(x0, x0 + W); ax.set_ylim(y0 + W, y0); ax.set_xticks([]); ax.set_yticks([])
axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
step.save_fig(fig, "step08_histology_windows")

# ---- Per-cell stain intensity (H&E level ~1.1 um / px) ------------------------------------------------------
img = he_level(cfg["feature_level"])
hed = rgb2hed(img)
to_he = scale(img.shape[1] / W0, img.shape[0] / H0) @ np.linalg.inv(M) @ scale(1 / PX, 1 / PX)
p = to_he @ np.vstack([xy.T, np.ones(len(xy))])
cols = np.clip(np.round(p[0]).astype(int), 0, img.shape[1] - 1)
rows = np.clip(np.round(p[1]).astype(int), 0, img.shape[0] - 1)
adata.obs["he_hematoxylin"] = hed[rows, cols, 0]
adata.obs["he_eosin"] = hed[rows, cols, 1]
del img, hed
feat = adata.obs.groupby("lineage", observed=True)[["he_hematoxylin", "he_eosin"]].median().sort_values("he_hematoxylin")
step.save_table(feat.round(4), "step08_stain_by_lineage")
step.say(f"  hematoxylin highest under {feat.index[-1]}, lowest under {feat.index[0]}")

adata.uns["step8_params"] = {"alignment": "he_imagealignment.csv (H&E px -> Xenium px)", "he_feature_level": cfg["feature_level"],
                             "qc_r_zero_shift": float(qc.iloc[0])}
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

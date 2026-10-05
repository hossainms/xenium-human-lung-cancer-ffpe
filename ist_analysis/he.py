"""H&E alignment: the 10x affine matrix, alignment check against DAPI, histology windows, per-cell stains (Step 8).

he_imagealignment.csv maps H&E pixels -> Xenium morphology pixels; its inverse maps Xenium -> H&E. Pixels
are read level by level from the H&E pyramid (tifffile), never the full 2 GB image at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io.xenium import PIXEL_SIZE_UM  # default morphology pixel size (Xenium)


def scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])


def load_alignment(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def he_level_shapes(he_path: Path) -> list[tuple]:
    import tifffile

    with tifffile.TiffFile(he_path) as tif:
        return [lvl.shape for lvl in tif.series[0].levels]


def he_level(he_path: Path, level: int) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(he_path) as tif:
        return tif.series[0].levels[level].asarray()


def to_he_matrix(M: np.ndarray, level_shape, full_shape) -> np.ndarray:
    """Xenium um -> H&E pixels of one pyramid level."""
    (h_l, w_l), (H0, W0) = level_shape[:2], full_shape[:2]
    return scale(w_l / W0, h_l / H0) @ np.linalg.inv(M) @ scale(1 / PIXEL_SIZE_UM, 1 / PIXEL_SIZE_UM)


def alignment_qc(he_path: Path, dapi_path: Path, M: np.ndarray, res_um: float = 4.0, he_level_idx: int = 3, max_shift: int = 10):
    """Hematoxylin (colour deconvolution) vs Xenium DAPI on a res_um grid, at shifts of +/- max_shift grid steps.

    A correct alignment peaks at zero shift. Returns (summary Series, correlation matrix, H&E grid, DAPI grid, shifts).
    """
    import tifffile
    from skimage.color import rgb2hed
    from skimage.transform import warp

    full = he_level_shapes(he_path)[0]
    with tifffile.TiffFile(dapi_path) as tif:
        Hd0, Wd0 = tif.series[0].levels[0].shape[-2:]
        dapi = tif.series[0].levels[3].asarray()[0].astype(float)
    shape = (int(np.ceil(Hd0 * PIXEL_SIZE_UM / res_um)), int(np.ceil(Wd0 * PIXEL_SIZE_UM / res_um)))
    T_dapi = scale(dapi.shape[1] / Wd0, dapi.shape[0] / Hd0) @ scale(1 / PIXEL_SIZE_UM, 1 / PIXEL_SIZE_UM) @ scale(res_um, res_um)
    dapi_grid = warp(dapi, T_dapi, output_shape=shape, order=1, preserve_range=True)
    img = he_level(he_path, he_level_idx)
    T = to_he_matrix(M, img.shape, full) @ scale(res_um, res_um)
    he_grid = warp(img, T, output_shape=shape, order=1, preserve_range=True, cval=255).astype(np.uint8)
    hema = rgb2hed(he_grid)[..., 0]
    dapi_log = np.log1p(dapi_grid)
    tissue = (dapi_grid > np.percentile(dapi_grid, 50)) | (hema > np.percentile(hema, 50))
    shifts = np.arange(-max_shift, max_shift + 1)
    corr = np.array([[np.corrcoef(dapi_log[tissue], np.roll(np.roll(hema, dy, 0), dx, 1)[tissue])[0, 1] for dx in shifts] for dy in shifts])
    iy, ix = np.unravel_index(corr.argmax(), corr.shape)
    c = max_shift
    summary = pd.Series({"r at zero shift": corr[c, c], "best r": corr.max(), "best shift x (um)": shifts[ix] * res_um,
                         "best shift y (um)": shifts[iy] * res_um,
                         "r at 40 um (mean of 4)": np.mean([corr[0, c], corr[-1, c], corr[c, 0], corr[c, -1]])})
    return summary, corr, he_grid, dapi_grid, shifts


def he_crop(he_path: Path, M: np.ndarray, cx: float, cy: float, size_um: float = 400, res_um: float = 0.5, level: int = 1):
    """H&E around a Xenium centre (um), warped to an aligned grid; reads only the tiles under the window."""
    import tifffile
    import zarr
    from skimage.transform import warp

    shapes = he_level_shapes(he_path)
    h_l, w_l = shapes[level][:2]
    x0, y0 = cx - size_um / 2, cy - size_um / 2
    to_he = to_he_matrix(M, shapes[level], shapes[0])
    c = to_he @ np.array([[x0, x0 + size_um, x0, x0 + size_um], [y0, y0, y0 + size_um, y0 + size_um], [1, 1, 1, 1]])
    lo = np.maximum(np.floor(c[:2].min(axis=1)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(c[:2].max(axis=1)).astype(int) + 2, [w_l, h_l])
    store = tifffile.imread(he_path, aszarr=True, level=level)
    patch = np.asarray(zarr.open(store, mode="r")[lo[1]:hi[1], lo[0]:hi[0]])
    store.close()
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]]) @ to_he @ np.array([[res_um, 0, x0], [0, res_um, y0], [0, 0, 1]])
    n = int(size_um / res_um)
    return warp(patch, T, output_shape=(n, n), order=1, preserve_range=True, cval=255).astype(np.uint8), (x0, y0)


def histology_windows(adata, window_um: float = 400) -> dict:
    """Three windows chosen from the data: the largest TLS, the densest myeloid-rich niche, the most mixed tumour/stroma tile."""
    xy = adata.obsm["spatial"]
    b = np.floor(xy / window_um).astype(int)
    keys, inv = np.unique(b, axis=0, return_inverse=True)
    epi = (adata.obs["lineage"] == "Epithelial").to_numpy()
    strom = (adata.obs["lineage"] == "Stromal").to_numpy()
    myeloid = adata.obs["niche"].astype(str).str.contains(r"^N\d+: Myeloid").to_numpy()
    return {"TLS1 (largest TLS-like aggregate)": xy[(adata.obs["tls_id"] == "TLS1").to_numpy()].mean(axis=0),
            "Myeloid-rich niche": (keys[np.bincount(inv, weights=myeloid).argmax()] + 0.5) * window_um,
            "Tumour-stroma border": (keys[np.minimum(np.bincount(inv, weights=epi), np.bincount(inv, weights=strom)).argmax()] + 0.5) * window_um}


def stain_features(adata, he_path: Path, M: np.ndarray, level: int = 2) -> None:
    """obs['he_hematoxylin'] and obs['he_eosin']: stain optical density at each cell centroid (H&E level ~1.1 um/px)."""
    from skimage.color import rgb2hed

    shapes = he_level_shapes(he_path)
    img = he_level(he_path, level)
    hed = rgb2hed(img)
    p = to_he_matrix(M, img.shape, shapes[0]) @ np.vstack([adata.obsm["spatial"].T, np.ones(adata.n_obs)])
    cols = np.clip(np.round(p[0]).astype(int), 0, img.shape[1] - 1)
    rows = np.clip(np.round(p[1]).astype(int), 0, img.shape[0] - 1)
    adata.obs["he_hematoxylin"] = hed[rows, cols, 0]
    adata.obs["he_eosin"] = hed[rows, cols, 1]

"""Morphology from the aligned H&E with a pathology foundation model (notebook 05).

The H&E is resampled once onto the Xenium coordinate grid (um), so a patch around any cell is a plain array
slice. Patches are embedded with a self-supervised histology model (CLS token), and the embeddings are used to
predict Xenium cell types and gene expression with cross-validation split by tissue blocks: neighbouring cells
share morphology, so a random split would leak and overstate accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .he import he_level, he_level_shapes, to_he_matrix

# Models that load without a Hugging Face account (CLS token of a DINOv2 / iBOT ViT, ImageNet normalisation).
# Gated models (UNI / UNI2-h, CONCH, Virchow2, H-optimus-0) need an access request and a token, and are not
# included because they were not tested here.
MODELS = {
    "phikon-v2": {"repo": "owkin/phikon-v2", "dim": 1024, "mpp": 0.5,
                  "note": "ViT-L / DINOv2, 460M tiles from 60k slides (Owkin non-commercial licence)"},
    "phikon": {"repo": "owkin/phikon", "dim": 768, "mpp": 0.5,
               "note": "ViT-B / iBOT, 43M tiles from TCGA (Owkin non-commercial licence)"},
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---- H&E on the Xenium grid --------------------------------------------------------------------------------------
def aligned_he(he_path: Path, M: np.ndarray, bounds_um, res_um: float, level: int, strip_px: int = 512) -> np.ndarray:
    """The H&E resampled onto a Xenium-aligned grid: pixel (row, col) covers x = x0 + col * res_um,
    y = y0 + row * res_um, with (x0, y0, x1, y1) = bounds_um. Built strip by strip (only the H&E pixels under
    each strip are converted to float), so memory stays near the size of the output image. uint8 RGB; white
    outside the scanned area."""
    from skimage.transform import warp

    x0, y0, x1, y1 = bounds_um
    w, h = int(np.ceil((x1 - x0) / res_um)), int(np.ceil((y1 - y0) / res_um))
    shapes = he_level_shapes(he_path)
    img = he_level(he_path, level)
    to_he = to_he_matrix(M, img.shape, shapes[0])
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    for r0 in range(0, h, strip_px):
        r1 = min(r0 + strip_px, h)
        grid = np.array([[res_um, 0, x0], [0, res_um, y0 + r0 * res_um], [0, 0, 1]])
        T = to_he @ grid                                     # strip pixel (col, row) -> H&E level pixel
        c = T @ np.array([[0, w, 0, w], [0, 0, r1 - r0, r1 - r0], [1, 1, 1, 1]])
        lo = np.maximum(np.floor(c[:2].min(axis=1)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(c[:2].max(axis=1)).astype(int) + 2, [img.shape[1], img.shape[0]])
        if (hi <= lo).any():
            continue
        src = img[lo[1]:hi[1], lo[0]:hi[0]]
        shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]])
        out[r0:r1] = warp(src, shift @ T, output_shape=(r1 - r0, w), order=1, preserve_range=True, cval=255).astype(np.uint8)
    return out


def patches(img: np.ndarray, xy_um: np.ndarray, origin_um, res_um: float, size_px: int = 224) -> np.ndarray:
    """Square patches of size_px centred on each point (um), from an aligned image; white padding at the edges."""
    half = size_px // 2
    pad = np.pad(img, ((half, half), (half, half), (0, 0)), constant_values=255)
    cols = np.round((xy_um[:, 0] - origin_um[0]) / res_um).astype(int) + half
    rows = np.round((xy_um[:, 1] - origin_um[1]) / res_um).astype(int) + half
    cols = np.clip(cols, half, pad.shape[1] - half)
    rows = np.clip(rows, half, pad.shape[0] - half)
    return np.stack([pad[r - half:r - half + size_px, c - half:c - half + size_px] for r, c in zip(rows, cols)])


def stain_summary(p: np.ndarray) -> np.ndarray:
    """Simple colour baseline per patch: mean and SD of R, G, B and of hematoxylin / eosin / DAB optical density."""
    from skimage.color import rgb2hed

    hed = np.stack([rgb2hed(x) for x in p])
    chans = np.concatenate([p.astype(np.float32) / 255, hed.astype(np.float32)], axis=-1)
    return np.concatenate([chans.mean(axis=(1, 2)), chans.std(axis=(1, 2))], axis=1)


def stain_features(img: np.ndarray, xy_um: np.ndarray, origin_um, res_um: float, size_px: int = 224, stride: int = 4,
                   batch: int = 2048) -> np.ndarray:
    """stain_summary of the patch around each point, in batches, on every stride-th pixel (12 features)."""
    return np.concatenate([stain_summary(patches(img, xy_um[i:i + batch], origin_um, res_um, size_px)[:, ::stride, ::stride])
                           for i in range(0, len(xy_um), batch)])


# ---- Embedding ---------------------------------------------------------------------------------------------------
def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name: str = "phikon-v2", device: str | None = None):
    """A histology foundation model in evaluation mode (half precision on a GPU). Returns (model, device, dtype)."""
    import torch
    from transformers import AutoModel

    device = device or default_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModel.from_pretrained(MODELS[name]["repo"]).eval().to(device, dtype)
    return model, device, dtype


def embed(model, device, dtype, img: np.ndarray, xy_um: np.ndarray, origin_um, res_um: float, size_px: int = 224,
          batch: int = 64, centre: int = 0, progress=None):
    """Embedding of the patch around each point: the CLS token (a summary of the whole patch).

    centre = k > 0 also returns the mean of the k x k patch tokens at the patch centre, i.e. the model's
    representation of the cell itself (each token is informed by the rest of the patch through attention).
    With 16 px tokens at 0.25 um/px, k = 2 covers the central 8 x 8 um. Returns cls, or (cls, centre).
    progress(i, n) is called after each batch."""
    import torch

    cls, cen = [], []
    with torch.inference_mode():
        for i in range(0, len(xy_um), batch):
            p = patches(img, xy_um[i:i + batch], origin_um, res_um, size_px)
            x = (p.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device, dtype)
            h = model(pixel_values=x).last_hidden_state
            cls.append(h[:, 0].float().cpu().numpy())
            if centre:
                g = int(round((h.shape[1] - 1) ** 0.5))                  # token grid (no register tokens in these models)
                tok = h[:, 1:1 + g * g].reshape(len(h), g, g, -1)
                a = g // 2 - centre // 2
                cen.append(tok[:, a:a + centre, a:a + centre].mean(dim=(1, 2)).float().cpu().numpy())
            if progress:
                progress(min(i + batch, len(xy_um)), len(xy_um))
    return (np.concatenate(cls), np.concatenate(cen)) if centre else np.concatenate(cls)


# ---- Sampling and cross-validation -------------------------------------------------------------------------------
def balanced_sample(labels: pd.Series, n_per: int, seed: int = 0) -> np.ndarray:
    """Positions of up to n_per cells of each label (all cells of rarer labels)."""
    rng = np.random.default_rng(seed)
    pos = np.arange(len(labels))
    keep = [rng.choice(idx, min(n_per, len(idx)), replace=False) for idx in pd.Series(pos).groupby(labels.to_numpy()).apply(np.asarray)]
    return np.sort(np.concatenate(keep))


def neighbourhood_composition(xy_all: np.ndarray, lineage_all: pd.Series, xy: np.ndarray, radius_um: float,
                              lineages: list[str]) -> np.ndarray:
    """Fraction of each lineage among the OTHER cells within radius_um of each point (the cell itself excluded)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy_all)
    codes = pd.Categorical(lineage_all, categories=lineages).codes
    out = np.zeros((len(xy), len(lineages)))
    for i, nb in enumerate(tree.query_ball_point(xy, radius_um)):
        c = codes[nb]
        counts = np.bincount(c[c >= 0], minlength=len(lineages)).astype(float)
        d = np.linalg.norm(xy_all[nb] - xy[i], axis=1)
        self_hit = np.flatnonzero(d < 1e-6)
        if len(self_hit) and codes[nb[self_hit[0]]] >= 0:
            counts[codes[nb[self_hit[0]]]] -= 1
        out[i] = counts / max(counts.sum(), 1)
    return out


def cv_classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_splits: int = 5, C: float = 0.1,
                n_components: int | None = None, seed: int = 0):
    """Out-of-fold predictions of a standardised multinomial logistic regression with balanced class weights.
    Folds are split by spatial block (groups); groups=None gives a random stratified split, which leaks between
    neighbouring cells and is only used to show how much that inflates accuracy.
    n_components: PCA before the classifier (fitted on the training folds only), for high-dimensional embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.empty(len(y), dtype=object)
    folds = (GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups) if groups is not None
             else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    for train, test in folds:
        steps = [StandardScaler()] + ([PCA(n_components, random_state=seed)] if n_components and n_components < X.shape[1] else [])
        clf = make_pipeline(*steps, LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        clf.fit(X[train], y[train])
        pred[test] = clf.predict(X[test])
    return pred


def scores(y: np.ndarray, pred: np.ndarray) -> pd.Series:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return pd.Series({"balanced accuracy": balanced_accuracy_score(y, pred), "macro F1": f1_score(y, pred, average="macro"),
                      "chance (balanced)": 1 / len(np.unique(y))})


def recall_by_class(y: np.ndarray, pred: np.ndarray) -> pd.Series:
    return pd.Series(y == pred).groupby(y).mean()


def cv_gene_r(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_splits: int = 5, alphas=(1e2, 1e3, 1e4, 1e5),
              seed: int = 0) -> np.ndarray:
    """Pearson r between out-of-fold ridge predictions and observed values, one per column of Y."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros_like(Y, dtype=float)
    for train, test in GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, Y, groups):
        sc = StandardScaler().fit(X[train])
        model = RidgeCV(alphas=alphas).fit(sc.transform(X[train]), Y[train])
        pred[test] = model.predict(sc.transform(X[test]))
    Yc, Pc = Y - Y.mean(axis=0), pred - pred.mean(axis=0)
    denom = np.sqrt((Yc ** 2).sum(axis=0) * (Pc ** 2).sum(axis=0))
    return np.divide((Yc * Pc).sum(axis=0), denom, out=np.zeros(Y.shape[1]), where=denom > 0)


def neighbourhood_mean(xy: np.ndarray, values, radius_um: float, rows: np.ndarray | None = None) -> np.ndarray:
    """Mean of each column of values (dense or sparse, one row per cell) over all cells within radius_um of each
    cell in rows (the cell included): the regional expression level around those cells."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    rows = np.arange(len(xy)) if rows is None else rows
    nbrs = cKDTree(xy).query_ball_point(xy[rows], radius_um)
    lens = np.array([len(n) for n in nbrs])
    A = sp.csr_matrix((np.repeat(1.0 / lens, lens), np.concatenate(nbrs), np.r_[0, np.cumsum(lens)]), shape=(len(rows), len(xy)))
    out = A @ values
    return out.toarray() if sp.issparse(out) else np.asarray(out)


# ---- Whole-slide tiles -------------------------------------------------------------------------------------------
def tiles(xy: np.ndarray, tile_um: float, min_cells: int = 5):
    """Centres (um) of tile_um x tile_um tiles holding at least min_cells cells, and each cell's tile index (-1 if none)."""
    b = np.floor(xy / tile_um).astype(int)
    keys, inv, counts = np.unique(b, axis=0, return_inverse=True, return_counts=True)
    keep = counts >= min_cells
    remap = np.full(len(keys), -1)
    remap[keep] = np.arange(keep.sum())
    return (keys[keep] + 0.5) * tile_um, remap[inv.ravel()]


def majority(labels: pd.Series, tile_of_cell: np.ndarray, n_tiles: int) -> pd.Series:
    """Most common label among the cells of each tile."""
    s = pd.Series(labels.to_numpy(), index=tile_of_cell)
    s = s[s.index >= 0]
    return s.groupby(level=0).agg(lambda v: v.value_counts().idxmax()).reindex(range(n_tiles))

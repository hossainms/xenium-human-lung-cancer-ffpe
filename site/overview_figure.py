"""Overview figure: one panel from each layer of the analysis (README, project website and portfolio).

    python site/overview_figure.py [--out images/overview_figure.jpg]       # in spatial_env, from the repository root

a  cell lineages across the section (Steps 6-7), with the windows of panels c-e
b  spatial domains (CellCharter, notebook 04)
c  segmentation: 10x vs. Proseg outlines; T cells assigned tumour (EPCAM / MALL) transcripts are filled (notebook 02)
d  spatial neighbourhood graph: Delaunay edges <= 30 um at a tumour-stroma border (Step 7a)
e  TLS1 on the matched H&E with Xenium cells coloured by lineage (Steps 7-8)
"""

import argparse
from pathlib import Path

import anndata as ad
import geopandas as gpd
import matplotlib
import matplotlib.colors  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ist_analysis import he
from ist_analysis import morphology as mo
from ist_analysis import utils as xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.linewidth": 0.6})
LIN = xu.LINEAGE_COLORS
SEG_WINDOW = (5730.0, 2070.0, 60.0)       # centre x, y and side (um): tumour edge where 10x T cells carry tumour transcripts
GRAPH_WINDOW = (3375.0, 2025.0, 150.0)    # most lineage-diverse tumour / stroma / immune border window
TLS_SIDE = 160.0
TUMOUR_GENES = ["EPCAM", "MALL"]


def bbox(cx, cy, side):
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def scale_bar(ax, x0, y0, length, label, color="black", lw=2.0):
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=lw, solid_capstyle="butt", zorder=5)
    lo, hi = ax.get_ylim()
    up = -1 if lo > hi else 1                                   # "above" on screen, also for inverted y axes
    box = {"boxstyle": "round,pad=0.15", "facecolor": "white" if color == "black" else "black", "edgecolor": "none", "alpha": 0.75}
    ax.text(x0 + length / 2, y0 + up * 0.045 * abs(hi - lo), label, color=color, ha="center", va="bottom", fontsize=8, bbox=box)


def tidy(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def letter(ax, s):
    ax.text(-0.01, 1.02, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


def tenx_polygons(outs: Path, box):
    x0, y0, x1, y1 = box
    f = outs / "cell_boundaries.parquet"
    near = pq.read_table(f, columns=["cell_id"], filters=[("vertex_x", ">", x0 - 15), ("vertex_x", "<", x1 + 15),
                                                          ("vertex_y", ">", y0 - 15), ("vertex_y", "<", y1 + 15)])
    ids = sorted(set(near["cell_id"].to_pylist()))
    v = pq.read_table(f, filters=[("cell_id", "in", ids)]).to_pandas()
    return {c: g[["vertex_x", "vertex_y"]].to_numpy() for c, g in v.groupby("cell_id", sort=False)}


def proseg_polygons(proseg_dir: Path, box):
    g = gpd.read_parquet(proseg_dir / "proseg-output.zarr" / "shapes" / "cell_boundaries" / "shapes.parquet")
    g = g.cx[box[0] - 15:box[2] + 15, box[1] - 15:box[3] + 15]
    meta = pd.read_csv(proseg_dir / "cell-metadata.csv.gz", usecols=["cell", "original_cell_id"]).set_index("cell")
    out = {}
    for cell, geom in zip(g["cell"], g.geometry):
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        out[cell] = (meta["original_cell_id"].get(cell), [np.asarray(p.exterior.coords) for p in parts])
    return out


def tumour_rna(adata, genes=TUMOUR_GENES) -> dict:
    """EPCAM + MALL counts per cell, from a method's own transcript-to-cell assignment (its count matrix)."""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    cols = [list(adata.var_names).index(g) for g in genes]
    return dict(zip(adata.obs_names, np.asarray(X[:, cols].sum(axis=1)).ravel()))


def segmentation_panel(ax, polys, lineage_of, tumour_of, box, title):
    """Outlines coloured by lineage; T cells assigned tumour (EPCAM / MALL) transcripts are filled.
    Returns (T cells with tumour transcripts, T cells) among outlines centred in the window."""
    keys = list(polys)
    edge = [LIN.get(lineage_of.get(k), "#9a9a9a") for k in keys]
    is_t = [lineage_of.get(k) == "T / NK" for k in keys]
    hit = [t and tumour_of.get(k, 0) > 0 for k, t in zip(keys, is_t)]
    face = [(*matplotlib.colors.to_rgb(LIN["Epithelial"]), 0.45) if h else "none" for h in hit]
    ax.add_collection(PolyCollection([polys[k] for k in keys], facecolors=face, edgecolors=edge, linewidths=1.1))
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.set_aspect("equal"); tidy(ax)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#bbbbbb")
    centred = [box[0] < polys[k].mean(axis=0)[0] < box[2] and box[1] < polys[k].mean(axis=0)[1] < box[3] for k in keys]
    n_t = sum(t and c for t, c in zip(is_t, centred)); n_hit = sum(h and c for h, c in zip(hit, centred))
    ax.set_title(title.format(hit=n_hit, n=n_t), fontsize=9, fontweight="normal")
    return n_hit, n_t


def main(out: Path):
    a = xu.load_step("time")
    xy = a.obsm["spatial"]
    lineage = a.obs["lineage"].astype(str).to_numpy()
    dom = pd.read_csv(xu.PROCESSED_DIR / "spatial_domains" / "domains.csv.gz", index_col="cell_id")["cellcharter"].reindex(a.obs_names)
    ann = ad.read_h5ad(xu.STEP_FILES["annotated"], backed="r")
    lineage_10x = dict(zip(ann.obs_names, ann.obs["lineage"].astype(str)))   # all QC cells, for the segmentation panel

    fig = plt.figure(figsize=(16, 8.3))
    gs = fig.add_gridspec(3, 12, height_ratios=[2.55, 0.3, 4.3], hspace=0.22, wspace=0.35, left=0.03, right=0.99, top=0.95, bottom=0.04)

    # ---- a: lineages, with the windows of c-e -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :6])
    order = np.random.default_rng(0).permutation(len(xy))
    ax.scatter(xy[order, 0] / 1000, xy[order, 1] / 1000, s=0.2, c=[LIN[x] for x in lineage[order]], linewidths=0, rasterized=True)
    tls1 = xy[(a.obs["tls_id"] == "TLS1").to_numpy()].mean(axis=0)
    for (cx, cy, side), lab in [(SEG_WINDOW, "c"), (GRAPH_WINDOW, "d"), ((tls1[0], tls1[1], TLS_SIDE), "e")]:
        side_box = max(side, 120.0)    # tiny windows drawn at least 120 um wide so they stay visible
        ax.add_patch(Rectangle(((cx - side_box / 2) / 1000, (cy - side_box / 2) / 1000), side_box / 1000, side_box / 1000,
                               fill=False, lw=1.3, color="black"))
        ax.text((cx + side_box / 2) / 1000 + 0.04, (cy - side_box / 2) / 1000, lab, fontsize=10, fontweight="bold", va="top")
    ax.set_xlim(0.05, 10.95); ax.set_ylim(3.7, 0.38); ax.set_aspect("equal"); tidy(ax)
    scale_bar(ax, 0.2, 3.55, 1.0, "1 mm")
    ax.set_title("Cell lineages (139,129 high-confidence cells)"); letter(ax, "a")

    # ---- b: CellCharter domains ----------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 6:])
    names = sorted(dom.dropna().unique(), key=lambda s: int(s.split(":")[0].lstrip("D")))
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(names):
        sel = (dom == d).to_numpy()
        ax.scatter(xy[sel, 0] / 1000, xy[sel, 1] / 1000, s=0.2, color=cmap(i), linewidths=0, rasterized=True)
    ax.set_xlim(0.05, 10.95); ax.set_ylim(3.7, 0.38); ax.set_aspect("equal"); tidy(ax)
    scale_bar(ax, 0.2, 3.55, 1.0, "1 mm")
    ax.set_title("Spatial domains (CellCharter)"); letter(ax, "b")
    ax.legend([Line2D([], [], marker="s", ls="", color=cmap(i), markersize=6) for i in range(len(names))], names,
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7, frameon=False, handletextpad=0.2, columnspacing=1.0)

    # ---- lineage legend ------------------------------------------------------------------------------------------------
    lax = fig.add_subplot(gs[1, :6]); lax.axis("off")
    lax.legend([Line2D([], [], marker="o", ls="", color=c, markersize=6) for c in LIN.values()], list(LIN), loc="center",
               ncol=7, fontsize=8, frameon=False, handletextpad=0.2, columnspacing=1.2)

    # ---- c: segmentation, 10x vs Proseg (each method's own transcript assignment) --------------------------------------
    box = bbox(*SEG_WINDOW)
    c1, c2 = fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6])
    tenx = tenx_polygons(xu.OUTS_DIR, box)
    h10, n10 = segmentation_panel(c1, tenx, lineage_10x, tumour_rna(ann.to_memory()), box,
                                  "10x segmentation: {hit} of {n} T cells carry tumour RNA")
    pro = proseg_polygons(xu.PROCESSED_DIR / "resegmentation" / "proseg", box)
    pro_counts = tumour_rna(ad.read_h5ad(xu.PROCESSED_DIR / "resegmentation" / f"{xu.SAMPLE}_proseg_counts.h5ad"))
    polys = {orig: max(parts, key=len) for _, (orig, parts) in pro.items() if isinstance(orig, str)}   # main outline per cell
    hp, npro = segmentation_panel(c2, polys, lineage_10x, pro_counts, box, "Proseg: {hit} of {n} T cells carry tumour RNA")
    print(f"  panel c: T cells assigned EPCAM/MALL transcripts: 10x {h10}/{n10}, Proseg {hp}/{npro}")
    scale_bar(c2, box[0] + 4, box[3] - 4, 20, "20 µm")
    c1.text(0.0, 1.10, "Re-segmentation removes tumour transcripts from T cells", transform=c1.transAxes, fontsize=10, fontweight="bold")
    c1.text(-0.01, 1.10, "c", transform=c1.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")
    c2.legend([Rectangle((0, 0), 1, 1, facecolor=(*matplotlib.colors.to_rgb(LIN["Epithelial"]), 0.45), edgecolor=LIN["T / NK"])],
              ["T cell assigned tumour transcripts (EPCAM / MALL); outlines coloured by lineage"], loc="upper center",
              bbox_to_anchor=(-0.05, -0.02), fontsize=7.5, frameon=False, handlelength=1.2)

    # ---- d: spatial neighbourhood graph -------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 6:9])
    box = bbox(*GRAPH_WINDOW)
    inside = (xy[:, 0] > box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] > box[1]) & (xy[:, 1] < box[3])
    G = a.obsp["spatial_connectivities"].tocoo()
    keep = inside[G.row] & inside[G.col] & (G.row < G.col)
    ax.add_collection(LineCollection(np.stack([xy[G.row[keep]], xy[G.col[keep]]], axis=1), colors="#9a9a9a", linewidths=0.6, zorder=1))
    ax.scatter(xy[inside, 0], xy[inside, 1], s=16, c=[LIN[x] for x in lineage[inside]], edgecolors="white", linewidths=0.4, zorder=2)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.set_aspect("equal"); tidy(ax)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#bbbbbb")
    scale_bar(ax, box[0] + 6, box[3] - 6, 50, "50 µm")
    ax.set_title(f"Spatial graph: {int(keep.sum())} Delaunay edges ≤ 30 µm"); letter(ax, "d")

    # ---- e: TLS1 on the H&E -------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 9:12])
    box = bbox(tls1[0], tls1[1], TLS_SIDE)
    img = mo.aligned_he(xu.HE_PATH, he.load_alignment(xu.HE_ALIGNMENT_PATH), box, 0.25, 0)
    ax.imshow(img, extent=[box[0], box[2], box[3], box[1]])
    inside = (xy[:, 0] > box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] > box[1]) & (xy[:, 1] < box[3])
    ax.scatter(xy[inside, 0], xy[inside, 1], s=7, c=[LIN[x] for x in lineage[inside]], edgecolors="white", linewidths=0.3)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); tidy(ax)
    scale_bar(ax, box[0] + 6, box[3] - 6, 50, "50 µm", color="white")
    ax.set_title("TLS1: B-cell core, T-cell rim on H&E"); letter(ax, "e")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        fig.savefig(out, dpi=200, facecolor="white", pil_kwargs={"quality": 88, "optimize": True})
    else:
        fig.savefig(out, dpi=200, facecolor="white")
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="images/overview_figure.jpg")
    main(Path(p.parse_args().out))

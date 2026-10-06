"""Overview figure: one panel from each layer of the analysis (README, project website and portfolio).

    python site/overview_figure.py [--out images/overview_figure.jpg]       # in spatial_env, from the repository root
    python site/overview_figure.py --thumbnail                             # 2 x 2 image tile for web cards
    python site/overview_figure.py --robustness                            # re-segmentation robustness figure (finding 4)

a  cell lineages across the section (Steps 6-7), with the windows of panels c-e
b  spatial domains (CellCharter, notebook 04)
c  segmentation on DAPI: 10x vs. Proseg outlines with tumour and T-cell transcripts; T cells assigned tumour RNA outlined (notebook 02)
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
SEG_WINDOW = (4260.0, 1980.0, 120.0)      # notebook 02's window: the 120 um tile where tumour and T-cell transcripts meet most densely
GRAPH_WINDOW = (5925.0, 1875.0, 150.0)    # most lineage-diverse tumour / stroma / immune 150 um tile between windows c and e,
                                          # so the boxes on panel a read c, d, e from left to right
TLS_SIDE = 160.0
TUMOUR_GENES, T_GENES = ["EPCAM", "MALL"], ["CD3E", "TRAC"]
# Transcript dots on the dark DAPI image: lighter tints of the lineage colours (tumour = blue, T cell = orange)
TX_COLOURS = {"tumour": "#6fb3ff", "T": "#ff9a52"}
FLAG = "#ff3fd2"                           # outline of a T cell assigned tumour transcripts


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


def dapi_crop(box):
    """DAPI (morphology_focus channel 0) for the window, read tile by tile, contrast-stretched to 0-1."""
    import tifffile
    import zarr

    px = xu.PIXEL_SIZE_UM
    store = tifffile.imread(xu.OUTS_DIR / "morphology_focus" / "morphology_focus_0000.ome.tif", aszarr=True, level=0)
    img = zarr.open(store, mode="r")
    r0, c0, r1, c1 = int(box[1] / px), int(box[0] / px), int(box[3] / px), int(box[2] / px)
    d = np.asarray(img[0, r0:r1, c0:c1] if img.ndim == 3 else img[r0:r1, c0:c1]).astype(float)
    store.close()
    lo, hi = np.percentile(d, [1, 99.7])
    return np.clip((d - lo) / (hi - lo), 0, 1)


def transcripts(box, genes):
    x0, y0, x1, y1 = box
    t = pq.read_table(xu.OUTS_DIR / "transcripts.parquet", columns=["feature_name", "x_location", "y_location", "qv"],
                      filters=[("x_location", ">", x0), ("x_location", "<", x1), ("y_location", ">", y0), ("y_location", "<", y1),
                               ("qv", ">=", 20)]).to_pandas()
    t["feature_name"] = t["feature_name"].astype(str)
    return t[t["feature_name"].isin(genes)]


def segmentation_panel(ax, polys, lineage_of, tumour_of, dapi, tx, box, title):
    """DAPI with cell outlines (yellow) and tumour / T-cell transcripts; T cells that the method assigned tumour
    transcripts (its own count matrix) get a bold magenta outline. Returns (flagged T cells, T cells) centred in the window."""
    ax.imshow(dapi, cmap="gray", extent=[box[0], box[2], box[3], box[1]])
    keys = list(polys)
    is_t = [lineage_of.get(k) == "T / NK" for k in keys]
    hit = [t and tumour_of.get(k, 0) > 0 for k, t in zip(keys, is_t)]
    ax.add_collection(PolyCollection([polys[k] for k in keys], facecolors="none", edgecolors="#f5d547", linewidths=0.6, alpha=0.9))
    ax.add_collection(PolyCollection([polys[k] for k, h in zip(keys, hit) if h], facecolors="none", edgecolors=FLAG, linewidths=1.8))
    for genes, colour in [(TUMOUR_GENES, TX_COLOURS["tumour"]), (T_GENES, TX_COLOURS["T"])]:
        t = tx[tx["feature_name"].isin(genes)]
        ax.scatter(t["x_location"], t["y_location"], s=3, color=colour, linewidths=0, zorder=3)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); tidy(ax)
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

    # ---- c: segmentation on DAPI, 10x vs Proseg (each method's own transcript-to-cell assignment) ------------------
    box = bbox(*SEG_WINDOW)
    dapi, tx = dapi_crop(box), transcripts(box, TUMOUR_GENES + T_GENES)
    c1, c2 = fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6])
    h10, n10 = segmentation_panel(c1, tenx_polygons(xu.OUTS_DIR, box), lineage_10x, tumour_rna(ann.to_memory()), dapi, tx, box,
                                  "10x segmentation: {hit} of {n} T cells carry tumour RNA")
    pro = proseg_polygons(xu.PROCESSED_DIR / "resegmentation" / "proseg", box)
    pro_counts = tumour_rna(ad.read_h5ad(xu.PROCESSED_DIR / "resegmentation" / f"{xu.SAMPLE}_proseg_counts.h5ad"))
    polys = {orig: max(parts, key=len) for _, (orig, parts) in pro.items() if isinstance(orig, str)}   # main outline per cell
    hp, npro = segmentation_panel(c2, polys, lineage_10x, pro_counts, dapi, tx, box, "Proseg: {hit} of {n} T cells carry tumour RNA")
    print(f"  panel c: T cells assigned EPCAM/MALL transcripts: 10x {h10}/{n10}, Proseg {hp}/{npro}")
    scale_bar(c1, box[0] + 6, box[3] - 6, 20, "20 µm", color="white")
    c1.text(0.0, 1.10, "Re-segmentation removes tumour transcripts from T cells", transform=c1.transAxes, fontsize=10, fontweight="bold")
    c1.text(-0.01, 1.10, "c", transform=c1.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")
    handles = [Line2D([], [], marker="o", ls="", color=TX_COLOURS[k], markersize=4) for k in ("tumour", "T")]
    handles += [Line2D([], [], color="#f5d547", lw=1), Line2D([], [], color=FLAG, lw=2)]
    c2.legend(handles, ["tumour RNA (EPCAM, MALL)", "T-cell RNA (CD3E, TRAC)", "cell outline", "T cell assigned tumour RNA"],
              loc="upper center", bbox_to_anchor=(-0.05, -0.02), ncol=4, fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.0)

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


def thumbnail(out: Path):
    """2 x 2 image tile without text for web cards: TLS1 on H&E, the same with Xenium lineages, the 10x segmentation on
    DAPI (window c) and the neighbourhood graph (window d)."""
    a = xu.load_step("time")
    xy = a.obsm["spatial"]
    lineage = a.obs["lineage"].astype(str).to_numpy()
    tls1 = xy[(a.obs["tls_id"] == "TLS1").to_numpy()].mean(axis=0)
    ann = ad.read_h5ad(xu.STEP_FILES["annotated"], backed="r")
    lineage_10x = dict(zip(ann.obs_names, ann.obs["lineage"].astype(str)))

    fig = plt.figure(figsize=(12, 12), dpi=100)
    gs = fig.add_gridspec(2, 2, wspace=0.012, hspace=0.012, left=0, right=1, top=1, bottom=0)
    inside = lambda b: (xy[:, 0] > b[0]) & (xy[:, 0] < b[2]) & (xy[:, 1] > b[1]) & (xy[:, 1] < b[3])  # noqa: E731

    box = bbox(tls1[0], tls1[1], TLS_SIDE)
    img = mo.aligned_he(xu.HE_PATH, he.load_alignment(xu.HE_ALIGNMENT_PATH), box, 0.25, 0)
    for k in range(2):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, extent=[box[0], box[2], box[3], box[1]])
        if k == 1:
            m = inside(box)
            ax.scatter(xy[m, 0], xy[m, 1], s=22, c=[LIN[x] for x in lineage[m]], edgecolors="white", linewidths=0.5)
        ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.axis("off")

    box = bbox(*SEG_WINDOW)
    ax = fig.add_subplot(gs[1, 0])
    segmentation_panel(ax, tenx_polygons(xu.OUTS_DIR, box), lineage_10x, tumour_rna(ann.to_memory()), dapi_crop(box),
                       transcripts(box, TUMOUR_GENES + T_GENES), box, "")
    ax.axis("off")

    box = bbox(*GRAPH_WINDOW)
    ax = fig.add_subplot(gs[1, 1])
    m = inside(box)
    G = a.obsp["spatial_connectivities"].tocoo()
    keep = m[G.row] & m[G.col] & (G.row < G.col)
    ax.add_collection(LineCollection(np.stack([xy[G.row[keep]], xy[G.col[keep]]], axis=1), colors="#9a9a9a", linewidths=0.9, zorder=1))
    ax.scatter(xy[m, 0], xy[m, 1], s=46, c=[LIN[x] for x in lineage[m]], edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.set_aspect("equal"); ax.axis("off")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "optimize": True})
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


def robustness_figure(out: Path, tables: Path):
    """Finding 4: (a) cross-lineage marker spillover, 10x vs Proseg (notebook 02, paired same cells); (b) CCR7 in CD8+ T
    cells by compartment under both segmentations, with the tumour-marker spillover control."""
    sp = pd.read_csv(tables / "nb02_paired_spillover.csv")
    st = pd.read_csv(tables / "nb02_cd8_states_10x_vs_proseg.csv")
    comps = ["TLS", "Stroma (> 50 um)", "Border (15-50 um)", "Tumour contact (<= 15 um)"]
    labels = ["TLS", "Stroma\n(> 50 µm)", "Border\n(15-50 µm)", "Tumour contact\n(≤ 15 µm)"]
    C10, CPS = "#9a9a9a", "#0f6e78"

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 9.2), gridspec_kw={"hspace": 0.42})   # stacked: fits a narrow web column
    sp = sp.sort_values("10x: % positive")
    y = np.arange(len(sp))
    for i, r in enumerate(sp.itertuples(index=False)):
        a1.plot([r[3], r[4]], [i, i], color="#cccccc", lw=2.5, zorder=1)
    a1.scatter(sp["10x: % positive"], y, color=C10, s=60, zorder=2, label="10x segmentation")
    a1.scatter(sp["Proseg: % positive"], y, color=CPS, s=60, zorder=3, label="Proseg")
    a1.set_yticks(y, [f"{c} cells with {m}" for c, m in zip(sp["cells (Step 6 lineage)"], sp["foreign markers"])], fontsize=8.5)
    a1.set_xlabel("% of cells with markers of another lineage")
    a1.set_title("Cross-lineage spillover, same cells", loc="left"); letter(a1, "a")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    for s_ in ("top", "right"):
        a1.spines[s_].set_visible(False)

    x = np.arange(len(comps))
    for gene, ls, name in [("CCR7", "-", "CCR7 (biology)"), ("EPCAM", ":", "EPCAM (spillover control)")]:
        for seg, col in [("10x", C10), ("Proseg", CPS)]:
            v = st[(st["gene"] == gene) & (st["segmentation"] == seg)][comps].to_numpy().ravel()
            a2.plot(x, v, ls=ls, marker="o", color=col, lw=2, label=f"{name}, {seg}")
    a2.set_xticks(x, labels, fontsize=8.5)
    a2.set_ylabel("% of CD8+ T cells positive")
    a2.set_title("CCR7 gradient persists; tumour-marker control disappears", loc="left"); letter(a2, "b")
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    for s_ in ("top", "right"):
        a2.spines[s_].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 90, "optimize": True})
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="images/overview_figure.jpg")
    p.add_argument("--thumbnail", action="store_true", help="write the 2 x 2 image tile for web cards instead")
    p.add_argument("--robustness", action="store_true", help="write the re-segmentation robustness figure (finding 4) instead")
    p.add_argument("--tables", default="~/data/xenium_lung/pipeline/tables", help="pipeline tables (for --robustness)")
    args = p.parse_args()
    if args.robustness:
        robustness_figure(Path(args.out if args.out != "images/overview_figure.jpg" else "images/resegmentation_robustness.jpg"),
                          Path(args.tables).expanduser())
    elif args.thumbnail:
        thumbnail(Path(args.out if args.out != "images/overview_figure.jpg" else "images/overview_thumbnail.jpg"))
    else:
        main(Path(args.out))

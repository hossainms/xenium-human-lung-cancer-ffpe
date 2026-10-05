"""Cell re-segmentation with Proseg: run, compare with the 10x segmentation, export, robustness (notebook 02)."""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SPILL_TESTS = [   # (Step 6 lineage, foreign genes that lineage should NOT express)
    ("T / NK", ["EPCAM", "MALL"]), ("T / NK", ["CD163", "CD68"]), ("Myeloid", ["EPCAM", "MALL"]),
    ("Myeloid", ["CD3E", "TRAC"]), ("Epithelial", ["CD3E", "TRAC"]), ("Stromal", ["EPCAM", "MALL"]),
]
MIXED_PARTNERS = {"T / epithelial mixed": ["EPCAM", "MALL"], "T / myeloid mixed": ["CD163", "MS4A6A"],
                  "T / fibroblast mixed": ["FBN1", "PDGFRA"], "Myeloid / epithelial mixed": ["EPCAM", "MALL"],
                  "Myeloid / T mixed": ["CD3E", "TRAC"], "Myeloid / fibroblast mixed": ["FBN1", "PDGFRA"], "CD4 T": ["VWF", "CAVIN1"]}
TNK_PROBES = {"CD4": ["CD4"], "CD8A": ["CD8A"], "FOXP3": ["FOXP3"], "T (CD3E/TRAC)": ["CD3E", "TRAC"], "epithelial": ["EPCAM", "MALL"],
              "macrophage": ["CD163", "CD68"], "endothelial": ["VWF", "CAVIN1"], "fibroblast": ["PDGFRA", "FBN1"], "mregDC": ["LAMP3"]}
FOREIGN = ["epithelial", "macrophage", "endothelial", "fibroblast", "mregDC"]


def proseg_args(transcripts: Path, zarr_name: str = "proseg-output.zarr", threads: int | None = None, preset: str = "--xenium") -> list[str]:
    """Proseg command-line arguments; preset = the platform input preset (--xenium, --merscope, --cosmx)."""
    args = [preset, str(transcripts), "--output-counts", "counts.csv.gz", "--output-cell-metadata", "cell-metadata.csv.gz",
            "--output-cell-polygons", "cell-polygons.geojson.gz", "--exclude-spatialdata-transcripts",
            "--output-spatialdata", zarr_name, "--overwrite"]
    return args + (["--nthreads", str(threads)] if threads else [])


def run_proseg(binary: Path, transcripts: Path, workdir: Path, threads: int | None = None, log_name: str = "run.log") -> None:
    """Run Proseg in workdir; caffeinate keeps macOS awake and /usr/bin/time -l records runtime and peak memory."""
    cmd = [str(binary), *proseg_args(transcripts, threads=threads)]
    if Path("/usr/bin/caffeinate").exists():
        cmd = ["caffeinate", "-i", "/usr/bin/time", "-l", *cmd]
    with open(workdir / log_name, "w") as log:
        subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=True)


def read_proseg(zarr_path: Path):
    """The Proseg SpatialData output: (count table, 2D cell-outline areas in um^2)."""
    import spatialdata as sd

    with warnings.catch_warnings():   # Proseg writes an older SpatialData zarr format; reading it is fine
        warnings.simplefilter("ignore")
        pro_sd = sd.read_zarr(zarr_path)
    return pro_sd.tables["table"], pro_sd.shapes["cell_boundaries"].geometry.area.to_numpy(), pro_sd


def summarise(adata, area) -> dict:
    X = sp.csr_matrix(adata.X)
    tot = np.asarray(X.sum(axis=1)).ravel()
    return {"cells": adata.n_obs, "cells with >= 10 transcripts": int((tot >= 10).sum()), "transcripts in cells": int(tot.sum()),
            "median transcripts / cell": float(np.median(tot)), "median genes / cell": float(np.median(np.asarray((X > 0).sum(axis=1)))),
            "median cell area (um^2)": float(np.nanmedian(area))}


def positive(adata, genes, cells) -> np.ndarray:
    """Per cell: any transcript of any of genes."""
    return np.asarray((sp.csr_matrix(adata[cells, genes].X) > 0).sum(axis=1)).ravel() > 0


def paired_spillover(tenx, pro, labels: pd.DataFrame, paired) -> pd.DataFrame:
    """The same cells before / after: % with transcripts of a lineage they should not express."""
    rows = []
    for lineage, genes in SPILL_TESTS:
        cells = paired[labels.loc[paired, "lineage"].to_numpy() == lineage]
        b, a = positive(tenx, genes, cells).mean(), positive(pro, genes, cells).mean()
        rows.append({"cells (Step 6 lineage)": lineage, "foreign markers": " / ".join(genes), "n": len(cells),
                     "10x: % positive": 100 * b, "Proseg: % positive": 100 * a, "reduction": b / max(a, 1e-9)})
    return pd.DataFrame(rows)


def mixed_spillover(tenx, pro, labels: pd.DataFrame, paired, top: int = 6) -> pd.DataFrame:
    """Step 6 'mixed' groups: their contaminant genes before / after re-segmentation."""
    mixed = paired[labels.loc[paired, "annotation_confidence"].to_numpy() == "mixed"]
    rows = []
    for t in labels.loc[mixed, "cell_type"].value_counts().head(top).index:
        partner = MIXED_PARTNERS.get(t)
        if partner:
            cells = paired[labels.loc[paired, "cell_type"].to_numpy() == t]
            rows.append({"Step 6 'mixed' group": t, "n": len(cells), "contaminant genes": " / ".join(partner),
                         "10x: % positive": 100 * positive(tenx, partner, cells).mean(), "Proseg: % positive": 100 * positive(pro, partner, cells).mean()})
    return pd.DataFrame(rows)


def export_counts(pro, area):
    """An AnnData indexed by the 10x cell ids. Proseg's own uns['proseg_run'] (local paths) is not copied."""
    import anndata as ad

    out = ad.AnnData(X=sp.csr_matrix(pro.X).astype(np.float32), var=pd.DataFrame(index=pro.var_names))
    out.obs_names = pro.obs["original_cell_id"].astype(str).to_numpy()
    out.obs["cell_area"] = area
    out.obs["proseg_volume"] = pro.obs["volume"].to_numpy()
    out.obs["transcript_counts"] = np.asarray(out.X.sum(axis=1)).ravel().astype(int)
    out.obsm["spatial"] = pro.obs[["centroid_x", "centroid_y"]].to_numpy()
    out.layers["counts"] = out.X.copy()
    out.uns["segmentation"] = {"method": "Proseg", "version": pro.uns["proseg_run"]["version"], "initialised_from": "10x nuclei",
                               "transcript_filter": "qv >= 20, gene features"}
    return out


def qc_proseg(pc, min_counts: int = 10, min_genes: int = 5, area_nmads: float = 3):
    """Core QC rules on Proseg cells (no control probes in Proseg output, so no control-fraction rule)."""
    from .utils import mad_bounds

    pc.obs["n_genes"] = np.asarray((pc.X > 0).sum(axis=1)).ravel()
    lo, hi = mad_bounds(pc.obs["cell_area"].to_numpy(), area_nmads)
    keep = (pc.obs["transcript_counts"] >= min_counts) & (pc.obs["n_genes"] >= min_genes) & pc.obs["cell_area"].between(lo, hi)
    return pc[keep.to_numpy()].copy(), (lo, hi)


def tnk_probe_rates(tnk) -> pd.DataFrame:
    """% of cells per T/NK subcluster with own-lineage and foreign-lineage transcripts."""
    cnt = sp.csr_matrix(tnk.layers["counts"])
    return pd.DataFrame({n: pd.Series(np.asarray((cnt[:, [tnk.var_names.get_loc(g) for g in gs]] > 0).sum(axis=1)).ravel() > 0,
                                      index=tnk.obs_names).groupby(tnk.obs["sub"].to_numpy()).mean() * 100 for n, gs in TNK_PROBES.items()}).round(0)


def clean_cd4_treg(rates: pd.DataFrame):
    """Clean CD4 T / Treg subclusters: CD4 >= 30%, T >= 60%, CD8A < 20% and every foreign lineage < 25%; Treg = FOXP3 >= 25%."""
    is_cd4 = (rates["CD4"] >= 30) & (rates["T (CD3E/TRAC)"] >= 60) & (rates["CD8A"] < 20)
    clean = rates[FOREIGN].max(axis=1) < 25
    treg = rates["FOXP3"] >= 25
    return rates[is_cd4 & clean & ~treg], rates[is_cd4 & clean & treg]


def cd8_by_segmentation(tenx, pro, time_obs: pd.DataFrame, genes, cell_type: str = "CD8 T (GZMK+)") -> pd.DataFrame:
    """The same CD8 T cells (Step 6 labels, Step 9 compartments), % positive per gene and location, with 10x vs Proseg counts."""
    cd8 = time_obs.index[time_obs["cell_type"] == cell_type].intersection(pro.obs_names)
    comps = list(time_obs["cd8_compartment"].cat.categories)
    rows = []
    for seg, a in [("10x", tenx), ("Proseg", pro)]:
        df = pd.DataFrame((sp.csr_matrix(a[cd8, list(genes)].X) > 0).toarray(), columns=list(genes), index=cd8)
        df["compartment"] = time_obs.loc[cd8, "cd8_compartment"].astype(str).to_numpy()
        for g in genes:
            r = df.groupby("compartment")[g].mean().reindex(comps) * 100
            rows.append({"segmentation": seg, "gene": g, **r.round(1).to_dict()})
    return pd.DataFrame(rows).set_index(["gene", "segmentation"]).sort_index(level=0, sort_remaining=False), len(cd8)


def positions_high_confidence(pro, labels: pd.DataFrame):
    """Proseg centroids of the cells annotated with high confidence, with their Step 6 cell types."""
    pxy = pd.DataFrame(pro.obsm["spatial"], index=pro.obs_names, columns=["x", "y"])
    hc = labels.index[labels["annotation_confidence"] == "high"].intersection(pxy.index)
    return pxy.loc[hc].to_numpy(), labels.loc[hc, "cell_type"]


def exclusion_on_positions(xy: np.ndarray, types: pd.Series, tumour_types, show, contact_um: float = 15):
    """Distance to the nearest tumour cell on Proseg positions: per type median and % within contact, and the baseline."""
    from scipy.spatial import cKDTree

    is_tum = types.isin(tumour_types).to_numpy()
    d, _ = cKDTree(xy[is_tum]).query(xy, k=1)
    base = float((d[~is_tum] <= contact_um).mean())
    rows = [{"cell type": t, "median distance (um)": np.median(d[(types == t).to_numpy()]),
             f"% within {contact_um} um": 100 * (d[(types == t).to_numpy()] <= contact_um).mean()} for t in show]
    return pd.DataFrame(rows).set_index("cell type"), base


def aggregates_on_positions(xy: np.ndarray, types: pd.Series, eps_um: float = 25, min_cells: int = 20) -> np.ndarray:
    """Centres of B-cell aggregates (DBSCAN) on Proseg positions, largest first."""
    from sklearn.cluster import DBSCAN

    b = (types == "B cell").to_numpy()
    db = DBSCAN(eps=eps_um, min_samples=min_cells).fit_predict(xy[b])
    return np.array([xy[b][db == k].mean(axis=0) for k in pd.Series(db[db >= 0]).value_counts().index])

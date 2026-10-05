"""Build the project website (GitHub Pages) from the analysis outputs.

    python site/build_site.py [--out ~/data/xenium_lung/site]      # in spatial_env, from the repository root

The site is written outside the repository (the viewer data is a zarr store of many small files) and
published to the gh-pages branch by site/deploy.sh. Contents:
  index.html, style.css, images/        landing page (site/ sources + README figures)
  notebooks/*.html                      the five notebooks, rendered with their saved outputs
  report.html                           the Snakemake report (workflow/make_report.py)
  viewer/cells.zarr, viewer/config.json Vitessce viewer data: high-confidence cells, lognormalised
                                        expression of all panel genes, cell types, niches, domains, TLS
The build fails if any file contains the local home folder.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ist_analysis import utils as xu

SITE_URL = "https://hossainms.github.io/xenium-human-lung-cancer-ffpe"
NOTEBOOKS = ["01_core_python", "01_core_R", "02_resegmentation_proseg", "03_reference_mapping", "04_spatial_domains"]

# obs columns shown as cell sets in the viewer: (column, name in the viewer)
CELL_SETS = [("lineage", "Lineage"), ("cell_type", "Cell type"), ("niche", "Niche (Step 7)"),
             ("banksy", "Domain: BANKSY"), ("cellcharter", "Domain: CellCharter"), ("tls_id", "TLS-like aggregate"),
             ("cd8_compartment", "CD8 T compartment"), ("immune_tile_phenotype", "Immune phenotype (400 um tile)"),
             ("leiden_0.5", "Leiden 0.5"), ("tenx_graphclust", "10x graph clustering")]


def viewer_data(out: Path) -> dict:
    """AnnData zarr (format 2, dense X chunked by gene) for Vitessce, and its view configuration."""
    from vitessce import AnnDataWrapper, VitessceConfig, get_initial_coordination_scope_prefix
    from vitessce import Component as cm
    from vitessce import CoordinationLevel as CL

    a = xu.load_step("time")
    domains = pd.read_csv(xu.PROCESSED_DIR / "spatial_domains" / "domains.csv.gz", index_col="cell_id")
    obs = a.obs[[c for c, _ in CELL_SETS if c in a.obs]].copy()
    obs.index = a.obs["cell_id"].astype(str).to_numpy()
    obs = obs.join(domains[["banksy", "cellcharter"]], how="left")
    obs = obs[[c for c, _ in CELL_SETS]]
    for c in obs:
        obs[c] = obs[c].astype(str).replace({"nan": "none", "None": "none"}).astype("category")
    X = a.layers["lognorm"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    v = ad.AnnData(X.astype(np.float32), obs=obs, var=pd.DataFrame(index=a.var_names.astype(str)),
                   obsm={"spatial": a.obsm["spatial"].astype(np.float32), "X_umap": a.obsm["X_umap"].astype(np.float32)})
    store = out / "viewer" / "cells.zarr"
    shutil.rmtree(store, ignore_errors=True)
    ad.settings.zarr_write_format = 2
    v.write_zarr(store, chunks=[v.n_obs, 10])                 # one chunk per 10 genes: the viewer loads genes on demand

    names = dict(CELL_SETS)
    vc = VitessceConfig(schema_version="1.0.16", name="Xenium human lung cancer FFPE: 139,129 cells",
                        description="High-confidence cells; colour by cell set or by gene (lognormalised counts).")
    ds = vc.add_dataset(name="Xenium lung cancer").add_object(AnnDataWrapper(
        adata_url=f"{SITE_URL}/viewer/cells.zarr", obs_spots_path="obsm/spatial",
        obs_embedding_paths=["obsm/X_umap"], obs_embedding_names=["UMAP"],
        obs_set_paths=[f"obs/{c}" for c in obs.columns], obs_set_names=[names[c] for c in obs.columns],
        obs_feature_matrix_path="X",
        coordination_values={"obsType": "cell", "featureType": "gene", "featureValueType": "expression"}))
    # spatialBeta draws cells as a spot layer; the colour state (selected cell sets / gene) is shared by all views
    spatial = vc.add_view("spatialBeta", dataset=ds)
    layers = vc.add_view("layerControllerBeta", dataset=ds)
    umap = vc.add_view(cm.SCATTERPLOT, dataset=ds, mapping="UMAP")
    sets = vc.add_view(cm.OBS_SETS, dataset=ds)
    genes = vc.add_view(cm.FEATURE_LIST, dataset=ds)
    spatial.set_props(title="Tissue (um)")
    colour = vc.add_coordination("obsType", "obsSetSelection", "obsSetColor", "featureSelection", "obsColorEncoding")
    colour[0].set_value("cell")
    colour[4].set_value("cellSetSelection")
    for view in (umap, sets, genes):
        view.use_coordination(*colour)
    xy = v.obsm["spatial"]
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    vc.link_views([spatial, layers], ["spatialTargetX", "spatialTargetY", "spatialZoom"],
                  [float((x0 + x1) / 2), float((y0 + y1) / 2), float(-np.log2(max(x1 - x0, y1 - y0) / 700))])
    vc.link_views_by_dict([spatial, layers], {"spotLayer": CL([{
        "obsType": colour[0], "obsSetSelection": colour[1], "obsSetColor": colour[2], "featureSelection": colour[3],
        "obsColorEncoding": colour[4], "spatialSpotRadius": 4.0, "spatialLayerVisible": True, "spatialLayerOpacity": 1.0,
        "featureValueColormapRange": [0, 1]}])}, scope_prefix=get_initial_coordination_scope_prefix("A", "obsSpots"))
    vc.layout((spatial | (umap / layers)) / (sets | genes))
    config = vc.to_dict(base_url=SITE_URL)
    (out / "viewer" / "config.json").write_text(json.dumps(config, indent=1))
    size = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
    print(f"  viewer: {v.n_obs:,} cells x {v.n_vars} genes, {len(obs.columns)} cell-set columns, {size / 1e6:.0f} MB")
    return config


def notebooks(out: Path) -> None:
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="lab")
    (out / "notebooks").mkdir(parents=True, exist_ok=True)
    for name in NOTEBOOKS:
        html, _ = exporter.from_filename(str(xu.REPO_ROOT / "notebooks" / f"{name}.ipynb"))
        (out / "notebooks" / f"{name}.html").write_text(html, encoding="utf-8")
        print(f"  notebook {name}: {len(html) / 1e6:.1f} MB")


def static(out: Path, report: Path) -> None:
    here = Path(__file__).parent
    for f in ("index.html", "style.css"):
        shutil.copy(here / f, out / f)
    (out / "images").mkdir(exist_ok=True)
    for f in (xu.REPO_ROOT / "images").iterdir():
        shutil.copy(f, out / "images" / f.name)
    shutil.copy(xu.REPO_ROOT / "workflow" / "rulegraph.png", out / "images" / "rulegraph.png")
    shutil.copy(report, out / "report.html")
    (out / ".nojekyll").write_text("")          # Jekyll would skip the zarr metadata files (.zattrs, .zarray)


def check_paths(out: Path) -> None:
    home = os.path.expanduser("~")
    leaks = [f for f in out.rglob("*") if f.is_file() and f.suffix in {".html", ".json", ".css", ".zattrs", ".zgroup"}
             and home in f.read_text(encoding="utf-8", errors="ignore")]
    if leaks:
        raise SystemExit(f"local paths found in: {[xu.display_path(f) for f in leaks]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="~/data/xenium_lung/site")
    p.add_argument("--report", default="~/data/xenium_lung/pipeline/report.html")
    p.add_argument("--skip-viewer", action="store_true", help="keep the existing viewer data")
    args = p.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building the site in {xu.display_path(out)}")
    static(out, Path(args.report).expanduser())
    notebooks(out)
    if not args.skip_viewer:
        viewer_data(out)
    check_paths(out)
    print("Done: no local paths in the site")

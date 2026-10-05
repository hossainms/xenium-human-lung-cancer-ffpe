"""Notebook 02, parts C-D: compare Proseg with the 10x segmentation, then export the Proseg counts.

C1: whole-slide summary. C2: the same cells before / after, grouped by their Step 6 lineage: % of cells
with transcripts of a lineage they should not express (spillover). D: an AnnData indexed by the 10x
cell ids. Proseg's own metadata stores its command line with local paths; it is not copied.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import spatialdata as sd

from common import Step

step = Step("Notebook 02 C-D: compare segmentations and export Proseg counts", inputs=["zarr", "annotated"], outputs=["output"])
with warnings.catch_warnings():   # Proseg writes an older SpatialData zarr format; reading it is fine
    warnings.simplefilter("ignore")
    pro_sd = sd.read_zarr(step.path("zarr"))
pro = pro_sd.tables["table"]
pro_area = pro_sd.shapes["cell_boundaries"].geometry.area.to_numpy()
tenx = sc.read_10x_h5(step.outs_dir / "cell_feature_matrix.h5")
cells10 = pd.read_parquet(step.outs_dir / "cells.parquet").set_index("cell_id")


def summarise(adata, area):
    X = sp.csr_matrix(adata.X)
    tot = np.asarray(X.sum(axis=1)).ravel()
    return {"cells": adata.n_obs, "cells with >= 10 transcripts": int((tot >= 10).sum()), "transcripts in cells": int(tot.sum()),
            "median transcripts / cell": float(np.median(tot)), "median genes / cell": float(np.median(np.asarray((X > 0).sum(axis=1)))),
            "median cell area (um2)": float(np.nanmedian(area))}


compare = pd.DataFrame({"10x segmentation": summarise(tenx, cells10.loc[tenx.obs_names, "cell_area"].to_numpy()),
                        "Proseg": summarise(pro, pro_area)})
compare["Proseg / 10x"] = (compare["Proseg"] / compare["10x segmentation"]).round(2)
step.save_table(compare, "nb02_segmentation_summary")

# ---- C2: paired spillover, same cells, Step 6 lineages ------------------------------------------------
labels = ad.read_h5ad(step.path("annotated"), backed="r").obs[["cell_type", "lineage", "annotation_confidence"]].copy()
pro.obs_names = pro.obs["original_cell_id"].astype(str).to_numpy()
paired = labels.index.intersection(pro.obs_names).intersection(tenx.obs_names)
positive = lambda a, genes, cells: np.asarray((sp.csr_matrix(a[cells, genes].X) > 0).sum(axis=1)).ravel() > 0   # noqa: E731
TESTS = [("T / NK", ["EPCAM", "MALL"]), ("T / NK", ["CD163", "CD68"]), ("Myeloid", ["EPCAM", "MALL"]),
         ("Myeloid", ["CD3E", "TRAC"]), ("Epithelial", ["CD3E", "TRAC"]), ("Stromal", ["EPCAM", "MALL"])]
rows = []
for lineage, genes in TESTS:
    cells = paired[labels.loc[paired, "lineage"].to_numpy() == lineage]
    b, a = positive(tenx, genes, cells).mean(), positive(pro, genes, cells).mean()
    rows.append({"Step 6 lineage": lineage, "foreign markers": " / ".join(genes), "cells": len(cells),
                 "10x % positive": 100 * b, "Proseg % positive": 100 * a, "reduction (x)": b / max(a, 1e-9)})
spill = pd.DataFrame(rows)
step.save_table(spill.round(2), "nb02_paired_spillover", index=False)
step.say(f"  {len(paired):,} paired cells; T/NK with EPCAM/MALL: {spill.iloc[0]['10x % positive']:.1f}% -> "
         f"{spill.iloc[0]['Proseg % positive']:.1f}%")

# ---- D: export --------------------------------------------------------------------------------------------
out_ad = ad.AnnData(X=sp.csr_matrix(pro.X).astype(np.float32), var=pd.DataFrame(index=pro.var_names))
out_ad.obs_names = pro.obs_names
out_ad.obs["cell_area"] = pro_area
out_ad.obs["proseg_volume"] = pro.obs["volume"].to_numpy()
out_ad.obs["transcript_counts"] = np.asarray(out_ad.X.sum(axis=1)).ravel().astype(int)
out_ad.obsm["spatial"] = pro.obs[["centroid_x", "centroid_y"]].to_numpy()
out_ad.layers["counts"] = out_ad.X.copy()
out_ad.uns["segmentation"] = {"method": "Proseg", "version": pro.uns["proseg_run"]["version"], "initialised_from": "10x nuclei",
                              "transcript_filter": "qv >= 20, gene features"}
out = step.path("output")
out_ad.write_h5ad(out, compression="gzip")
step.done(out)

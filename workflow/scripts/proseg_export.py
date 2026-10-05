"""Notebook 02, parts C-D: compare Proseg with the 10x segmentation, then export the counts (ist_analysis.proseg)."""

import anndata as ad
import pandas as pd
import scanpy as sc

from common import Step
from ist_analysis import proseg

step = Step("Notebook 02 C-D: compare segmentations and export Proseg counts", inputs=["zarr", "annotated"], outputs=["output"])
pro, pro_area, _ = proseg.read_proseg(step.path("zarr"))
tenx = sc.read_10x_h5(step.outs_dir / "cell_feature_matrix.h5")
cells10 = pd.read_parquet(step.outs_dir / "cells.parquet").set_index("cell_id")

compare = pd.DataFrame({"10x segmentation": proseg.summarise(tenx, cells10.loc[tenx.obs_names, "cell_area"].to_numpy()),
                        "Proseg": proseg.summarise(pro, pro_area)})
compare["Proseg / 10x"] = (compare["Proseg"] / compare["10x segmentation"]).round(2)
step.save_table(compare, "nb02_segmentation_summary")

out_ad = proseg.export_counts(pro, pro_area)                  # indexed by the 10x cell ids
labels = ad.read_h5ad(step.path("annotated"), backed="r").obs[["cell_type", "lineage", "annotation_confidence"]].copy()
paired = labels.index.intersection(out_ad.obs_names).intersection(tenx.obs_names)
spill = proseg.paired_spillover(tenx, out_ad, labels, paired)
step.save_table(spill.round(2), "nb02_paired_spillover", index=False)
step.say(f"  {len(paired):,} paired cells; T/NK with EPCAM/MALL: {spill.iloc[0]['10x: % positive']:.1f}% -> {spill.iloc[0]['Proseg: % positive']:.1f}%")

out = step.path("output")
out_ad.write_h5ad(out, compression="gzip")
step.done(out)

"""Notebook 03: validate the manual cell types against the Lung Cancer Atlas (ist_analysis.reference).

A1: atlas restricted to the panel genes (cached). B: CellTypist's model on a balanced sample, ceiling on
held-out donors. C: annotate the 10x and Proseg cells. D: agreement by lineage and by cell type.
"""

import pickle

import anndata as ad
import pandas as pd
import scanpy as sc

from common import Step
from ist_analysis import proseg, reference

step = Step("Notebook 03: reference mapping to LuCA", inputs=["atlas", "annotated", "proseg"], outputs=["marker"])
cfg = step.config["reference"]
atlas_path = step.path("atlas")
cache = atlas_path.with_name("luca_core_panel_genes.h5ad")
ann = sc.read_h5ad(step.path("annotated"))

ref = ad.read_h5ad(cache) if cache.exists() else reference.build_panel_reference(atlas_path, ann.var["gene_ids"], cache)
genes = ref.var_names
step.say(f"  atlas: {ref.n_obs:,} cells, {ref.obs['donor_id'].nunique()} donors; {ref.n_vars} of {ann.n_vars} panel genes shared")

o = ref.obs
train_idx, test_idx, test_donors, donors = reference.donor_split(o, reference.eligible(o), cfg["test_donor_fraction"], cfg["max_cells_per_type"])
model = reference.train_model(reference.lognorm(ref[train_idx], genes), o.loc[train_idx, "cell_type_major"].astype(str).to_numpy())
with open(step.results / "reference_mapping_model.pkl", "wb") as fh:
    pickle.dump({"model": model, "genes": list(genes)}, fh)
truth = o.loc[test_idx, "cell_type_major"].astype(str).to_numpy()
held = reference.predict(model, reference.lognorm(ref[test_idx], genes), test_idx)["atlas_label"].to_numpy()
ceiling = pd.Series(held == truth).groupby(truth).mean()
step.save_table(ceiling.rename("held-out accuracy").round(3).to_frame(), "nb03_ceiling")
step.say(f"  ceiling on {len(test_donors)} held-out donors: {(held == truth).mean():.1%}")

atlas10 = reference.predict(model, reference.lognorm(ann, genes), ann.obs_names)
pro, _ = proseg.qc_proseg(sc.read_h5ad(step.path("proseg")), area_nmads=step.config["qc"]["area_nmads"])
atlasP = reference.predict(model, reference.lognorm(pro, genes), pro.obs_names)

manual = ann.obs[["cell_type", "lineage", "annotation_confidence"]].astype(str)
high = manual.index[manual["annotation_confidence"] == "high"]
atlas_lin = atlas10.loc[high, "atlas_label"].map(reference.ATLAS_LINEAGE)
step.save_table((pd.crosstab(manual.loc[high, "lineage"], atlas_lin, normalize="index") * 100).round(1), "nb03_lineage_agreement")
step.say(f"  lineage agreement on {len(high):,} high-confidence cells: {(manual.loc[high, 'lineage'] == atlas_lin).mean():.1%}")
step.save_table(reference.agreement(atlas10, manual, high, ceiling).round(2), "nb03_agreement_by_cell_type")
paired = manual.index.intersection(atlasP.index).intersection(high)
seg = pd.DataFrame({"% confirmed, 10x": reference.agreement(atlas10, manual, paired, ceiling)["% confirmed by atlas"],
                    "% confirmed, Proseg": reference.agreement(atlasP, manual, paired, ceiling)["% confirmed by atlas"]})
step.save_table(seg.round(1), "nb03_agreement_10x_vs_proseg")

export = manual.join(atlas10.add_prefix("tenx_"), how="left").join(atlasP.add_prefix("proseg_"), how="left")
export.index.name = "cell_id"
export.to_csv(step.tables / "nb03_atlas_labels.csv.gz")
marker = step.path("marker")
marker.write_text("notebook 03 complete\n")
step.done(marker)

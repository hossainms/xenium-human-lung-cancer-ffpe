"""Notebook 02, part E: which core findings survive Proseg re-segmentation? (ist_analysis.proseg, cluster, annotate)

E1: re-cluster Proseg cells with the core recipe; E1b: clean CD4 T / Treg subclusters under explicit spillover
thresholds; E2: the same CD8 T cells with 10x vs Proseg counts; E3: CD8 exclusion and TLS on Proseg positions.
"""

import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

from common import Step, xu
from ist_analysis import annotate, cluster, proseg

step = Step("Notebook 02 E: robustness of findings to re-segmentation", inputs=["proseg", "annotated", "clustered", "time"], outputs=["marker"])
qc_cfg, sp_cfg = step.config["qc"], step.config["spatial"]
contact = sp_cfg["contact_um"]
labels = ad.read_h5ad(step.path("annotated"), backed="r").obs[["cell_type", "lineage", "annotation_confidence"]].copy()
tenx = sc.read_10x_h5(step.outs_dir / "cell_feature_matrix.h5")
pro = sc.read_h5ad(step.path("proseg"))

# ---- E1: QC + clustering with the core recipe ---------------------------------------------------
pc, _ = proseg.qc_proseg(pro.copy(), qc_cfg["min_counts"], qc_cfg["min_genes"], qc_cfg["area_nmads"])
cluster.normalise_and_pca(pc)
cluster.cluster(pc, n_pcs=30, n_neighbors=15, resolutions=[0.5], seed=0, umap=False)
tenx_clusters = ad.read_h5ad(step.path("clustered"), backed="r").obs[step.config["cluster"]["working"]]
shared = pc.obs_names.intersection(tenx_clusters.index)
ari = adjusted_rand_score(tenx_clusters[shared], pc.obs.loc[shared, "leiden_0.5"])
step.say(f"  E1: {pc.n_obs:,} Proseg cells after QC, {pc.obs['leiden_0.5'].nunique()} clusters, ARI vs 10x-based clusters {ari:.2f}")

# ---- E1b: clean CD4 T / Treg? ---------------------------------------------------------------------
step6 = labels.reindex(pc.obs_names)
lin_of_cluster = pd.crosstab(pc.obs["leiden_0.5"], step6["lineage"]).idxmax(axis=1)
tnk = annotate.subcluster(pc, "leiden_0.5", list(lin_of_cluster.index[lin_of_cluster == "T / NK"]), resolution=0.6)
rates = proseg.tnk_probe_rates(tnk)
rates["cells"] = tnk.obs["sub"].value_counts()
step.save_table(rates, "nb02_tnk_subclusters_proseg")
clean_cd4, clean_treg = proseg.clean_cd4_treg(rates)

# ---- E2: same CD8 T cells, 10x vs Proseg counts ------------------------------------------------------
time_obs = ad.read_h5ad(step.path("time"), backed="r").obs[["cell_type", "cd8_compartment"]]
e2, _ = proseg.cd8_by_segmentation(tenx, pro, time_obs, ["CCR7", "IL7R", "GZMA", "CTLA4", "HAVCR2", "EPCAM", "MALL"])
step.save_table(e2, "nb02_cd8_states_10x_vs_proseg")
COMP = list(time_obs["cd8_compartment"].cat.categories)

# ---- E3: exclusion and TLS on Proseg positions ------------------------------------------------------------
xy, types = proseg.positions_high_confidence(pro, labels)
excl, base = proseg.exclusion_on_positions(xy, types, xu.TUMOUR_TYPES, ["CD8 T (GZMK+)"], contact)
cd8_contact = excl.iloc[0, 1]
n_agg = len(proseg.aggregates_on_positions(xy, types, sp_cfg["tls_eps_um"], sp_cfg["tls_min_cells"]))

ccr7 = e2.loc[("CCR7", "Proseg")]
verdict = pd.DataFrame([
    ["CD8 T cells excluded from tumour", f"{cd8_contact:.0f}% within {contact} um vs {100 * base:.0f}% baseline",
     "holds" if cd8_contact < 80 * base else "changes"],
    ["TLS-like B-cell aggregates", f"{n_agg} aggregates on Proseg positions", "holds" if n_agg >= 5 else "changes"],
    ["CD8 T cells lose CCR7 toward tumour", f"{ccr7[COMP[0]]:.0f}% -> {ccr7[COMP[-1]]:.0f}%", "holds" if ccr7[COMP[0]] > 2 * ccr7[COMP[-1]] else "weakens"],
    ["Clean conventional CD4 T subcluster", f"{len(clean_cd4)}", "newly visible" if len(clean_cd4) else "still not resolved"],
    ["Clean Treg subcluster", f"{len(clean_treg)}", "resolved" if len(clean_treg) else "not resolved"],
], columns=["finding", "Proseg result", "verdict"]).set_index("finding")
step.save_table(verdict, "nb02_verdict")
step.say(verdict.to_string())

marker = step.path("marker")
marker.write_text("notebook 02 robustness complete\n")
step.done(marker)

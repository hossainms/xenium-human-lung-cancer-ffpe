"""Notebook 02, part E: which core findings survive Proseg re-segmentation?

E1: re-cluster Proseg cells with the core recipe; E1b: look for clean CD4 T / Treg subclusters with explicit
spillover thresholds; E2: the same CD8 T cells and checkpoint rates with 10x vs Proseg counts; E3: CD8
exclusion and TLS-like aggregates on Proseg positions. Ends with a verdict table.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from common import Step, mad_bounds, xu

step = Step("Notebook 02 E: robustness of findings to re-segmentation",
            inputs=["proseg", "annotated", "clustered", "time"], outputs=["marker"])
sp_cfg = step.config["spatial"]
contact = sp_cfg["contact_um"]
labels = ad.read_h5ad(step.path("annotated"), backed="r").obs[["cell_type", "lineage", "annotation_confidence"]].copy()
tenx = sc.read_10x_h5(step.outs_dir / "cell_feature_matrix.h5")
pro = sc.read_h5ad(step.path("proseg"))

# ---- E1: QC + clustering with the core recipe ----------------------------------------------------------
pc = pro.copy()
pc.obs["n_genes"] = np.asarray((pc.X > 0).sum(axis=1)).ravel()
lo, hi = mad_bounds(pc.obs["cell_area"].to_numpy(), step.config["qc"]["area_nmads"])
pc = pc[((pc.obs["transcript_counts"] >= step.config["qc"]["min_counts"]) & (pc.obs["n_genes"] >= step.config["qc"]["min_genes"])
         & pc.obs["cell_area"].between(lo, hi)).to_numpy()].copy()
sc.pp.normalize_total(pc)
sc.pp.log1p(pc)
pc.layers["lognorm"] = pc.X.copy()
sc.pp.scale(pc, max_value=10)
sc.pp.pca(pc, n_comps=50, svd_solver="arpack", random_state=0)
pc.X = pc.layers["lognorm"]
sc.pp.neighbors(pc, n_neighbors=15, n_pcs=30, random_state=0)
sc.tl.leiden(pc, resolution=0.5, key_added="leiden", flavor="igraph", n_iterations=2, directed=False, random_state=0)
tenx_clusters = ad.read_h5ad(step.path("clustered"), backed="r").obs[step.config["cluster"]["working"]]
shared = pc.obs_names.intersection(tenx_clusters.index)
ari = adjusted_rand_score(tenx_clusters[shared], pc.obs.loc[shared, "leiden"])
step6 = labels.reindex(pc.obs_names)
step.say(f"  E1: {pc.n_obs:,} Proseg cells after QC, {pc.obs['leiden'].nunique()} clusters, ARI vs 10x-based clusters {ari:.2f}")

# ---- E1b: clean CD4 T / Treg? ------------------------------------------------------------------------------
lin_of_cluster = pd.crosstab(pc.obs["leiden"], step6["lineage"]).idxmax(axis=1)
tnk = pc[pc.obs["leiden"].isin(lin_of_cluster.index[lin_of_cluster == "T / NK"]).to_numpy()].copy()
tnk.X = tnk.layers["lognorm"].copy()
sc.pp.scale(tnk, max_value=10)
sc.pp.pca(tnk, n_comps=30, svd_solver="arpack", random_state=0)
tnk.X = tnk.layers["lognorm"]
sc.pp.neighbors(tnk, n_neighbors=15, n_pcs=20, random_state=0)
sc.tl.leiden(tnk, resolution=0.6, key_added="sub", flavor="igraph", n_iterations=2, directed=False, random_state=0)
PROBES = {"CD4": ["CD4"], "CD8A": ["CD8A"], "FOXP3": ["FOXP3"], "T (CD3E/TRAC)": ["CD3E", "TRAC"], "epithelial": ["EPCAM", "MALL"],
          "macrophage": ["CD163", "CD68"], "endothelial": ["VWF", "CAVIN1"], "fibroblast": ["PDGFRA", "FBN1"], "mregDC": ["LAMP3"]}
FOREIGN = ["epithelial", "macrophage", "endothelial", "fibroblast", "mregDC"]
cnt = sp.csr_matrix(tnk.layers["counts"])
rates = pd.DataFrame({n: pd.Series(np.asarray((cnt[:, [tnk.var_names.get_loc(g) for g in gs]] > 0).sum(axis=1)).ravel() > 0,
                                   index=tnk.obs_names).groupby(tnk.obs["sub"].to_numpy()).mean() * 100 for n, gs in PROBES.items()})
rates["cells"] = tnk.obs["sub"].value_counts()
step.save_table(rates.round(1), "nb02_tnk_subclusters_proseg")
is_cd4 = (rates["CD4"] >= 30) & (rates["T (CD3E/TRAC)"] >= 60) & (rates["CD8A"] < 20)
clean = rates[FOREIGN].max(axis=1) < 25
n_cd4, n_treg = int((is_cd4 & clean & (rates["FOXP3"] < 25)).sum()), int((is_cd4 & clean & (rates["FOXP3"] >= 25)).sum())

# ---- E2: same CD8 T cells, 10x vs Proseg counts ------------------------------------------------------------------
pro.obs_names = pro.obs_names.astype(str)
time_obs = ad.read_h5ad(step.path("time"), backed="r").obs[["cell_type", "cd8_compartment"]]
cd8 = time_obs.index[time_obs["cell_type"] == "CD8 T (GZMK+)"].intersection(pro.obs_names)
COMP = list(time_obs["cd8_compartment"].cat.categories)
GENES = ["CCR7", "IL7R", "GZMA", "CTLA4", "HAVCR2", "EPCAM", "MALL"]
rows = []
for seg, a in [("10x", tenx), ("Proseg", pro)]:
    df = pd.DataFrame((sp.csr_matrix(a[cd8, GENES].X) > 0).toarray(), columns=GENES, index=cd8)
    df["c"] = time_obs.loc[cd8, "cd8_compartment"].astype(str).to_numpy()
    for g in GENES:
        rows.append({"segmentation": seg, "gene": g, **(df.groupby("c")[g].mean().reindex(COMP) * 100).round(1).to_dict()})
e2 = pd.DataFrame(rows).set_index(["gene", "segmentation"])
step.save_table(e2, "nb02_cd8_states_10x_vs_proseg")

# ---- E3: exclusion and TLS on Proseg positions ---------------------------------------------------------------------
pxy = pd.DataFrame(pro.obsm["spatial"], index=pro.obs_names, columns=["x", "y"])
hc = labels.index[labels["annotation_confidence"] == "high"].intersection(pxy.index)
types = labels.loc[hc, "cell_type"]
is_tum = types.isin(xu.TUMOUR_TYPES).to_numpy()
xy = pxy.loc[hc].to_numpy()
d, _ = cKDTree(xy[is_tum]).query(xy, k=1)
base = 100 * (d[~is_tum] <= contact).mean()
cd8_contact = 100 * (d[(types == "CD8 T (GZMK+)").to_numpy()] <= contact).mean()
b = (types == "B cell").to_numpy()
db = DBSCAN(eps=sp_cfg["tls_eps_um"], min_samples=sp_cfg["tls_min_cells"]).fit_predict(xy[b])
n_agg = int(db.max() + 1)

ccr7 = e2.loc[("CCR7", "Proseg")]
verdict = pd.DataFrame([
    ["Tumour-to-immune spillover removed", "see nb02_paired_spillover", "", ""],
    ["CD8 T cells excluded from tumour", f"{cd8_contact:.0f}% within {contact} um vs {base:.0f}% baseline", "",
     "holds" if cd8_contact < 0.8 * base else "changes"],
    ["TLS-like B-cell aggregates", f"{n_agg} aggregates on Proseg positions", "", "holds" if n_agg >= 5 else "changes"],
    ["CD8 T cells lose CCR7 toward tumour", f"{ccr7[COMP[0]]:.0f}% -> {ccr7[COMP[-1]]:.0f}%", "",
     "holds" if ccr7[COMP[0]] > 2 * ccr7[COMP[-1]] else "weakens"],
    ["Clean conventional CD4 T subcluster", f"{n_cd4}", "", "newly visible" if n_cd4 else "still not resolved"],
    ["Clean Treg subcluster", f"{n_treg}", "", "resolved" if n_treg else "not resolved"],
], columns=["finding", "Proseg result", "", "verdict"]).drop(columns="").set_index("finding")
step.save_table(verdict, "nb02_verdict")
step.say(verdict.to_string())

marker = step.path("marker")
marker.write_text("notebook 02 robustness complete\n")
step.done(marker)

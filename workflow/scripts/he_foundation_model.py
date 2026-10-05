"""Notebook 05: what a pathology foundation model sees in the H&E (ist_analysis.morphology).

A: H&E resampled onto the Xenium grid. B: cell-centred patches (56 and 112 um; CLS and centre
tokens) and whole-slide tiles embedded with the model (cached). C: cell type from morphology vs. stain colour and Xenium neighbours, block-split CV.
D: gene predictability (own vs. regional expression) against Moran's I. E: morphology domains vs. CellCharter / BANKSY.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

from common import Step, xu
from ist_analysis import domains as dm
from ist_analysis import he
from ist_analysis import morphology as mo

step = Step("Notebook 05: H&E foundation-model features", inputs=["input", "domains", "moran"], outputs=["marker"])
cfg = step.config["he_model"]
adata = sc.read_h5ad(step.path("input"))
xy = adata.obsm["spatial"]
cell_type, lineage = adata.obs["cell_type"].astype(str), adata.obs["lineage"].astype(str)
LINEAGES = list(xu.LINEAGE_COLORS)
he_path = step.sample_dir / f"{step.config['sample']}_he_image.ome.tif"
M = he.load_alignment(step.sample_dir / f"{step.config['sample']}_he_imagealignment.csv")
bounds = (np.floor(xy[:, 0].min()) - 100, np.floor(xy[:, 1].min()) - 100, np.ceil(xy[:, 0].max()) + 100, np.ceil(xy[:, 1].max()) + 100)
FOV = {"cell_56um": (0.25, 0), "context_112um": (0.5, 1)}

idx = mo.balanced_sample(cell_type, cfg["n_per_type"], seed=0)
xy_s, ct_s, lin_s = xy[idx], cell_type.iloc[idx].to_numpy(), lineage.iloc[idx].to_numpy()
centres, tile_of_cell = mo.tiles(xy, cfg["tile_um"])
images = {name: mo.aligned_he(he_path, M, bounds, res, level) for name, (res, level) in FOV.items()}

cache = step.results / f"he_embeddings_{cfg['model']}.npz"            # resumable: the embedding is the slow part
if cache.exists() and np.array_equal(np.load(cache)["cell_ids"], adata.obs_names[idx].to_numpy().astype(str)):
    z = np.load(cache)
    emb = {k: z[k] for k in ["cell_56um", "cell_56um_centre", "context_112um", "tiles_112um"]}
    step.say(f"  cached embeddings: {xu.display_path(cache)}")
else:
    model, device, dtype = mo.load_model(cfg["model"])
    step.say(f"  {cfg['model']} on {device}: {len(idx):,} cells x 2 fields of view + {len(centres):,} tiles")
    emb = {name: mo.embed(model, device, dtype, images[fov], pts, bounds[:2], FOV[fov][0])
           for name, pts, fov in [("context_112um", xy_s, "context_112um"), ("tiles_112um", centres, "context_112um")]}
    emb["cell_56um"], emb["cell_56um_centre"] = mo.embed(model, device, dtype, images["cell_56um"], xy_s, bounds[:2], 0.25, centre=2)
    np.savez(cache, cell_ids=adata.obs_names[idx].to_numpy().astype(str), tile_centres=centres, **emb)

# ---- C. Cell identity from morphology ----------------------------------------------------------------------------
blocks = dm.blocks(xy_s, cfg["block_um"], adata.obs_names[idx]).to_numpy()
FEATURES = {"Stain colour (H&E)": (np.hstack([mo.stain_features(images[f], xy_s, bounds[:2], FOV[f][0]) for f in FOV]), None),
            "H&E model, 112 um patch": (emb["context_112um"], 128), "H&E model, 56 um patch": (emb["cell_56um"], 128),
            "H&E model, cell (centre tokens)": (emb["cell_56um_centre"], 128),
            "Xenium neighbours (50 um)": (mo.neighbourhood_composition(xy, lineage, xy_s, 50, LINEAGES), None)}
rows, recall = [], {}
for level, y in [("lineage", lin_s), ("cell type", ct_s)]:
    for name, (X, ncomp) in FEATURES.items():
        pred = mo.cv_classify(X, y, blocks, n_components=ncomp)
        rows.append({"label": level, "features": name, **mo.scores(y, pred).round(3).to_dict()})
        if level == "cell type":
            recall[name] = mo.recall_by_class(y, pred)
results = pd.DataFrame(rows).set_index(["label", "features"])
step.save_table(results, "nb05_classification_scores")
step.save_table((pd.DataFrame(recall) * 100).round(1), "nb05_recall_by_cell_type")
step.say(results.to_string())

# ---- D. Gene predictability --------------------------------------------------------------------------------------
lognorm = adata.layers["lognorm"]
Y_cell = lognorm[idx].toarray()
Y_region = mo.neighbourhood_mean(xy, lognorm, 50, rows=idx)
genes = pd.DataFrame({"r, own expression (cell tokens)": mo.cv_gene_r(emb["cell_56um_centre"], Y_cell, blocks),
                      "r, regional expression (112 um)": mo.cv_gene_r(emb["context_112um"], Y_region, blocks)}, index=adata.var_names)
genes["Moran's I"] = pd.read_csv(step.path("moran"), index_col=0)["I"].reindex(genes.index)
step.save_table(genes.round(4), "nb05_gene_predictability")
step.say(f"  regional predictability vs Moran's I: Spearman {spearmanr(genes.iloc[:, 1], genes.iloc[:, 2], nan_policy='omit')[0]:.2f}")

# ---- E. Morphology domains ---------------------------------------------------------------------------------------
Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(emb["tiles_112um"]))
morph = KMeans(cfg["n_domains"], n_init=10, random_state=0).fit_predict(Z)
dom = pd.read_csv(step.path("domains"), index_col="cell_id")
cell_ids = adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs else adata.obs_names.to_numpy()
agree = {m: ari(mo.majority(dom[m].reindex(cell_ids).fillna("none").reset_index(drop=True), tile_of_cell, len(centres)), morph)
         for m in ["banksy", "cellcharter"]}
step.save_table(pd.Series(agree, name="ARI vs morphology domains").round(3).to_frame(), "nb05_morphology_domain_agreement")
step.say(f"  morphology domains (K = {cfg['n_domains']}) ARI: {agree}")

marker = step.path("marker")
marker.write_text("notebook 05 complete\n")
step.done(marker)

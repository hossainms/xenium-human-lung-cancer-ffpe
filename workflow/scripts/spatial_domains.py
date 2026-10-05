"""Notebook 04: spatial domains (BANKSY, CellCharter) and location-dependent macrophage states.

B: BANKSY on a lambda x resolution grid (stability = ARI with the chosen setting). C: CellCharter with
stability-chosen K (PyTorch seeded explicitly: its random_state argument does not seed it). D: compare
with the Step 7 niches (raw ARI and tissue-compartment agreement). E: paired pseudobulk DESeq2 of one
macrophage type between domains, with a foreign-gene spillover filter and Proseg confirmation.
F: Xenium Explorer cell-group CSVs.
"""

import contextlib
import io
import logging

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score as ari

from common import Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Notebook 04: spatial domains and domain-dependent expression", inputs=["input", "proseg"], outputs=["marker"])
cfg = step.config["domains"]
adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"].copy()
xy = adata.obsm["spatial"]
adata.obs["x"], adata.obs["y"] = xy[:, 0], xy[:, 1]
adata.obsm["coord_xy"] = xy
lineage = adata.obs["lineage"].astype(str)
domains = pd.DataFrame({"niche": adata.obs["niche"].astype(str)}, index=adata.obs_names)

# ---- B. BANKSY ------------------------------------------------------------------------------------------
from banksy.embed_banksy import generate_banksy_matrix  # noqa: E402
from banksy.initialize_banksy import initialize_banksy  # noqa: E402

LAMBDAS, RESOLUTIONS = [0.5, cfg["banksy_lambda"], 0.9], [0.1, 0.15, cfg["banksy_resolution"], 0.3]
with contextlib.redirect_stdout(io.StringIO()):
    bdict = initialize_banksy(adata, ("x", "y", "coord_xy"), num_neighbours=15, max_m=0, plt_edge_hist=False, plt_nbr_weights=False, plt_theta=False)
    bdict, _ = generate_banksy_matrix(adata, bdict, LAMBDAS, max_m=0, verbose=False)
grid = {}
for lam in LAMBDAS:
    bm = bdict["scaled_gaussian"][lam]["adata"]
    sc.pp.pca(bm, n_comps=20, random_state=0)
    sc.pp.neighbors(bm, n_neighbors=15, use_rep="X_pca", random_state=0)
    for res in RESOLUTIONS:
        sc.tl.leiden(bm, resolution=res, flavor="igraph", n_iterations=2, directed=False, random_state=0, key_added="leiden")
        grid[(lam, res)] = bm.obs["leiden"].to_numpy()
del bdict
chosen = grid[(cfg["banksy_lambda"], cfg["banksy_resolution"])]
stab = pd.DataFrame([[ari(chosen, grid[(l, r)]) for r in RESOLUTIONS] for l in LAMBDAS],
                    index=[f"lambda {l}" for l in LAMBDAS], columns=[f"res {r}" for r in RESOLUTIONS])
step.save_table(stab.round(3), "nb04_banksy_stability")
domains["banksy"] = chosen

# ---- C. CellCharter ------------------------------------------------------------------------------------------
import cellcharter as cc  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import squidpy as sq  # noqa: E402

# Lightning (CellCharter's GMM trainer) sets its own log levels on import, so silence it afterwards
for name in ["pytorch_lightning", "lightning", "lightning.pytorch", "lightning.fabric", "torchgmm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, key_added="cc")
cc.gr.remove_long_links(adata, connectivity_key="cc_connectivities", distances_key="cc_distances")
adata.obsm["X_pca30"] = sc.pp.pca(sc.pp.scale(adata.layers["lognorm"], max_value=10, copy=True), n_comps=30, random_state=0)
cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep="X_pca30", connectivity_key="cc_connectivities", out_key="X_cellcharter")
pl.seed_everything(0, verbose=False)
autok = cc.tl.ClusterAutoK(n_clusters=tuple(cfg["cellcharter_k_range"]), max_runs=cfg["cellcharter_runs"],
                           model_params={"random_state": 0, "trainer_params": {"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False}})
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    autok.fit(adata, use_rep="X_cellcharter")
    domains["cellcharter"] = np.asarray(autok.predict(adata, use_rep="X_cellcharter")).astype(str)
stability = pd.Series(np.asarray(autok.stability).mean(axis=1), index=autok.n_clusters[1:-1], name="stability")
step.save_table(stability.round(3).to_frame(), "nb04_cellcharter_stability")
step.say(f"  BANKSY {domains['banksy'].nunique()} domains; CellCharter most stable K = {domains['cellcharter'].nunique()}")

# ---- D. Name domains by composition, compare --------------------------------------------------------------------
LINEAGES = list(xu.LINEAGE_COLORS)


def name_domains(labels):
    comp = pd.crosstab(labels, lineage, normalize="index").reindex(columns=LINEAGES).fillna(0) * 100
    names = {}
    for d, row in comp.iterrows():
        top = row.sort_values(ascending=False)
        names[d] = f"{top.index[0]} {top.iloc[0]:.0f}%" + (f" + {top.index[1]} {top.iloc[1]:.0f}%" if top.iloc[1] >= 15 else "")
    prefix = {d: f"D{i + 1}" for i, d in enumerate(comp.sort_values("Epithelial", ascending=False).index)}
    return labels.map(lambda d: f"{prefix[d]}: {names[d]}")


for m in ["banksy", "cellcharter"]:
    domains[m] = name_domains(domains[m])
    comp = pd.crosstab(domains[m], lineage, normalize="index").reindex(columns=LINEAGES).fillna(0) * 100
    comp["cells"] = domains[m].value_counts()
    step.save_table(comp.round(1), f"nb04_composition_{m}")
COMPARTMENT = {"Epithelial": "tumour", "T / NK": "immune", "B / plasma": "immune", "Myeloid": "immune", "Mast": "immune",
               "Stromal": "stroma / vessels", "Endothelial": "stroma / vessels"}
compartment = lambda labels: labels.map(pd.crosstab(labels, lineage.map(COMPARTMENT), normalize="index").idxmax(axis=1))  # noqa: E731
comps = {m: compartment(domains[m]) for m in ["niche", "banksy", "cellcharter"]}
agree = pd.DataFrame([{"pair": f"{a} vs {b}", "ARI": ari(domains[a], domains[b]), "same compartment (%)": 100 * (comps[a] == comps[b]).mean()}
                      for a, b in [("niche", "banksy"), ("niche", "cellcharter"), ("banksy", "cellcharter")]]).set_index("pair")
step.save_table(agree.round(3), "nb04_method_agreement")
step.say(agree.round(2).to_string())

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
for ax, m in zip(axes, ["niche", "banksy", "cellcharter"]):
    order = sorted(domains[m].unique(), key=lambda s: int(s.split(":")[0].lstrip("ND")))
    cmap = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    for i, d in enumerate(order):
        sel = (domains[m] == d).to_numpy()
        ax.scatter(xy[sel, 0] / 1000, xy[sel, 1] / 1000, s=0.15, color=cmap(i), label=d, rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(m, loc="left")
    ax.legend(markerscale=25, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
step.save_fig(fig, "nb04_domain_maps")

# ---- E. Paired pseudobulk DE of one macrophage type between domains ------------------------------------------------
from pydeseq2.dds import DeseqDataSet  # noqa: E402
from pydeseq2.default_inference import DefaultInference  # noqa: E402
from pydeseq2.ds import DeseqStats  # noqa: E402

pro = sc.read_h5ad(step.path("proseg"))
ccl = domains["cellcharter"]
comp_cc = pd.crosstab(ccl, lineage, normalize="index")
ROLE = {"tumour-myeloid": comp_cc.loc[comp_cc.index[comp_cc["Epithelial"] >= 0.5], "Myeloid"].idxmax(),
        "stroma": ccl[comps["cellcharter"] == "stroma / vessels"].value_counts().idxmax(),
        "lymphoid": (comp_cc["T / NK"] + comp_cc["B / plasma"]).idxmax()}
bs = cfg["block_um"]
block = pd.Series([f"b{int(x // bs)}_{int(y // bs)}" for x, y in xy], index=adata.obs_names)
norm = adata.layers["lognorm"]
lin_mean = pd.DataFrame({l: np.asarray(np.expm1(norm[(lineage == l).to_numpy()]).mean(axis=0)).ravel() for l in LINEAGES}, index=adata.var_names)
other = lin_mean.drop(columns="Myeloid")
foreign_of = other.idxmax(axis=1).where(other.max(axis=1) >= 4 * lin_mean["Myeloid"].clip(lower=1e-3))


def pseudobulk_de(cell_type, dom_a, dom_b, counts="10x", min_cells=10):
    cells = adata.obs_names[(adata.obs["cell_type"] == cell_type).to_numpy() & ccl.isin([dom_a, dom_b]).to_numpy()]
    if counts == "proseg":
        cells = cells.intersection(pro.obs_names)
        X, genes = sp.csr_matrix(pro[cells].X), pro.var_names
    else:
        X, genes = sp.csr_matrix(adata[cells].layers["counts"]), adata.var_names
    key = (block[cells] + "|" + ccl[cells]).to_numpy()
    groups = pd.Series(key).value_counts()
    groups = groups[groups >= min_cells].index
    meta = pd.DataFrame([g.split("|") for g in groups], columns=["block", "domain"], index=groups)
    paired = meta.groupby("block")["domain"].nunique().pipe(lambda s: s[s == 2].index)
    meta = meta[meta["block"].isin(paired)]
    if len(paired) < 5:
        raise ValueError(f"{cell_type}: only {len(paired)} paired blocks; too few to test")
    bulk = pd.DataFrame(np.vstack([np.asarray(X[key == g].sum(axis=0)).ravel() for g in meta.index]).astype(int), index=meta.index, columns=genes)
    bulk = bulk.loc[:, bulk.sum() >= 10]
    meta["domain"] = pd.Categorical(np.where(meta["domain"] == dom_a, "A", "B"), categories=["B", "A"])
    dds = DeseqDataSet(counts=bulk, metadata=meta, design="~ block + domain", inference=DefaultInference(n_cpus=8), quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["domain", "A", "B"], quiet=True)
    with contextlib.redirect_stdout(io.StringIO()):
        stats.summary()
    return stats.results_df[["baseMean", "log2FoldChange", "padj"]], len(paired)


for cell_type, a_role, b_role in [("FCGR1A+ macrophage", "tumour-myeloid", "stroma"), ("CD163+ macrophage", "lymphoid", "stroma")]:
    r10, n_blocks = pseudobulk_de(cell_type, ROLE[a_role], ROLE[b_role])
    rP, _ = pseudobulk_de(cell_type, ROLE[a_role], ROLE[b_role], counts="proseg")
    r = r10.join(rP[["log2FoldChange", "padj"]], rsuffix="_proseg")
    r["foreign (lineage)"] = foreign_of.reindex(r.index)
    sig = r["padj"] < 0.05
    same = (np.sign(r["log2FoldChange"]) == np.sign(r["log2FoldChange_proseg"])) & (r["padj_proseg"] < 0.05)
    r["call"] = np.select([~sig, r["foreign (lineage)"].notna(), ~same], ["n.s.", "spillover (foreign gene)", "not confirmed with Proseg"], "macrophage-intrinsic")
    step.save_table(r.sort_values("padj"), f"nb04_pseudobulk_{cell_type.split('+')[0].lower()}_{a_role}_vs_{b_role}")
    intrinsic = r[r["call"] == "macrophage-intrinsic"].sort_values("log2FoldChange")
    step.say(f"  {cell_type} {a_role} vs {b_role} ({n_blocks} paired blocks): {sig.sum()} significant, "
             f"{(r['call'] == 'spillover (foreign gene)').sum()} spillover; intrinsic up: {', '.join(intrinsic.index[intrinsic.log2FoldChange > 0][-5:])}; "
             f"down: {', '.join(intrinsic.index[intrinsic.log2FoldChange < 0][:5])}")

# ---- F. Xenium Explorer cell groups ----------------------------------------------------------------------------------------
explorer = step.results / "xenium_explorer"
explorer.mkdir(exist_ok=True)
cmap = plt.get_cmap("tab20")
for name, labels in {"cell_types": adata.obs["cell_type"].astype(str), "lineages": lineage, "niches_kmeans": domains["niche"],
                     "domains_banksy": domains["banksy"], "domains_cellcharter": domains["cellcharter"]}.items():
    cats = sorted(labels.unique())
    colors = xu.LINEAGE_COLORS if name == "lineages" else {c: "#%02x%02x%02x" % tuple(int(255 * v) for v in cmap(i % 20)[:3]) for i, c in enumerate(cats)}
    pd.DataFrame({"cell_id": adata.obs_names, "group": labels.to_numpy(), "color": labels.map(colors).to_numpy()}).to_csv(explorer / f"{name}.csv", index=False)
domains.to_csv(step.tables / "nb04_domains.csv.gz", index_label="cell_id")
step.say(f"  Xenium Explorer cell groups in {xu.display_path(explorer)}")
marker = step.path("marker")
marker.write_text("notebook 04 complete\n")
step.done(marker)

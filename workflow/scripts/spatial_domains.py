"""Notebook 04: spatial domains and location-dependent macrophage states (ist_analysis.domains).

B: BANKSY on a lambda x resolution grid. C: CellCharter with stability-chosen K. D: comparison with the
Step 7 niches. E: paired pseudobulk DESeq2 of one macrophage type between domains, with the spillover
filter and Proseg confirmation. F: Xenium Explorer cell groups.
"""

import matplotlib
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score as ari

from common import Step, xu
from ist_analysis import domains as dm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Notebook 04: spatial domains and domain-dependent expression", inputs=["input", "proseg"], outputs=["marker"])
cfg = step.config["domains"]
adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"].copy()
xy = adata.obsm["spatial"]
lineage = adata.obs["lineage"].astype(str)
LINEAGES = list(xu.LINEAGE_COLORS)
labels = pd.DataFrame({"niche": adata.obs["niche"].astype(str)}, index=adata.obs_names)

lambdas, resolutions = [0.5, cfg["banksy_lambda"], 0.9], [0.1, 0.15, cfg["banksy_resolution"], 0.3]
grid = dm.banksy_grid(adata, lambdas, resolutions)
chosen = (cfg["banksy_lambda"], cfg["banksy_resolution"])
stab, _ = dm.grid_stability(grid, chosen, lambdas, resolutions)
step.save_table(stab.round(3), "nb04_banksy_stability")
labels["banksy"] = grid[chosen]

dm.cellcharter_embedding(adata)
labels["cellcharter"], stability = dm.cellcharter_autok(adata, cfg["cellcharter_k_range"], cfg["cellcharter_runs"])
step.save_table(stability.round(3).to_frame(), "nb04_cellcharter_stability")
step.say(f"  BANKSY {labels['banksy'].nunique()} domains; CellCharter most stable K = {labels['cellcharter'].nunique()}")

for m in ["banksy", "cellcharter"]:
    labels[m] = dm.name_domains(labels[m], lineage, LINEAGES)
    comp = pd.crosstab(labels[m], lineage, normalize="index").reindex(columns=LINEAGES).fillna(0) * 100
    comp["cells"] = labels[m].value_counts()
    step.save_table(comp.round(1), f"nb04_composition_{m}")
comps = {m: dm.compartment(labels[m], lineage) for m in ["niche", "banksy", "cellcharter"]}
agree = pd.DataFrame([{"pair": f"{a} vs {b}", "ARI": ari(labels[a], labels[b]), "same compartment (%)": 100 * (comps[a] == comps[b]).mean()}
                      for a, b in [("niche", "banksy"), ("niche", "cellcharter"), ("banksy", "cellcharter")]]).set_index("pair")
step.save_table(agree.round(3), "nb04_method_agreement")
step.say(agree.round(2).to_string())

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
for ax, m in zip(axes, ["niche", "banksy", "cellcharter"]):
    order = sorted(labels[m].unique(), key=lambda s: int(s.split(":")[0].lstrip("ND")))
    cmap = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    for i, d in enumerate(order):
        sel = (labels[m] == d).to_numpy()
        ax.scatter(xy[sel, 0] / 1000, xy[sel, 1] / 1000, s=0.15, color=cmap(i), label=d, rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(m, loc="left")
    ax.legend(markerscale=25, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
step.save_fig(fig, "nb04_domain_maps")

# ---- E. Paired pseudobulk DE of one macrophage type between domains ----------------------------------------------
pro = sc.read_h5ad(step.path("proseg"))
ccl = labels["cellcharter"]
comp_cc = pd.crosstab(ccl, lineage, normalize="index")
ROLE = {"tumour-myeloid": comp_cc.loc[comp_cc.index[comp_cc["Epithelial"] >= 0.5], "Myeloid"].idxmax(),
        "stroma": ccl[comps["cellcharter"] == "stroma / vessels"].value_counts().idxmax(),
        "lymphoid": (comp_cc["T / NK"] + comp_cc["B / plasma"]).idxmax()}
block = dm.blocks(xy, cfg["block_um"], adata.obs_names)
foreign = dm.foreign_genes(adata, lineage)
for cell_type, a_role, b_role in [("FCGR1A+ macrophage", "tumour-myeloid", "stroma"), ("CD163+ macrophage", "lymphoid", "stroma")]:
    cells = adata.obs_names[(adata.obs["cell_type"] == cell_type).to_numpy() & ccl.isin([ROLE[a_role], ROLE[b_role]]).to_numpy()]
    r10, info = dm.pseudobulk_de(adata[cells].layers["counts"], adata.var_names, cells, block, ccl, ROLE[a_role], ROLE[b_role])
    pcells = cells.intersection(pro.obs_names)
    rP, _ = dm.pseudobulk_de(pro[pcells].X, pro.var_names, pcells, block, ccl, ROLE[a_role], ROLE[b_role])
    r = dm.call_genes(r10, rP, foreign)
    step.save_table(r.sort_values("padj"), f"nb04_pseudobulk_{cell_type.split('+')[0].lower()}_{a_role}_vs_{b_role}")
    intrinsic = r[r["call"] == "macrophage-intrinsic"].sort_values("log2FoldChange")
    step.say(f"  {cell_type} {a_role} vs {b_role} ({info['blocks']} paired blocks, {info['cells']:,} cells): {(r['padj'] < 0.05).sum()} significant, "
             f"{(r['call'] == 'spillover (foreign gene)').sum()} spillover; intrinsic up: {', '.join(intrinsic.index[intrinsic.log2FoldChange > 0][-5:])}; "
             f"down: {', '.join(intrinsic.index[intrinsic.log2FoldChange < 0][:5])}")

# ---- F. Xenium Explorer cell groups -------------------------------------------------------------------------------------
explorer = step.results / "xenium_explorer"
explorer.mkdir(exist_ok=True)
for name, lab in {"cell_types": adata.obs["cell_type"].astype(str), "lineages": lineage, "niches_kmeans": labels["niche"],
                  "domains_banksy": labels["banksy"], "domains_cellcharter": labels["cellcharter"]}.items():
    dm.explorer_groups(lab, adata.obs_names, xu.LINEAGE_COLORS if name == "lineages" else None).to_csv(explorer / f"{name}.csv", index=False)
labels.to_csv(step.tables / "nb04_domains.csv.gz", index_label="cell_id")
step.say(f"  Xenium Explorer cell groups in {xu.display_path(explorer)}")
marker = step.path("marker")
marker.write_text("notebook 04 complete\n")
step.done(marker)

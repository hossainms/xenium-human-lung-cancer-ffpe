"""Step 9: tumour immune microenvironment (TIME) deep-dive.

Checkpoint positivity rates within cell types (against the tumour noise floor), CD8 T-cell states by
location, a spatial ligand-receptor contact test (null: expression shuffled within cell type), immune
phenotype per tumour tile (inflamed / excluded / desert) and immunosuppressive ratios by distance.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from common import Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

step = Step("Step 9: tumour immune microenvironment", inputs=["input"], outputs=["output"])
cfg = step.config["time"]
contact = step.config["spatial"]["contact_um"]

adata = sc.read_h5ad(step.path("input"))
xy = adata.obsm["spatial"]
cell_type = adata.obs["cell_type"].astype(str)
is_tumour = cell_type.isin(xu.TUMOUR_TYPES).to_numpy()
counts = adata.layers["counts"].tocsc()
positive = lambda g: np.asarray(counts[:, adata.var_names.get_loc(g)].todense()).ravel() > 0   # noqa: E731

# ---- 9a. Checkpoint positivity rate within each cell type -----------------------------------------
CHECKPOINTS = {"PDCD1": "PD-1", "CD274": "PD-L1", "CTLA4": "CTLA-4", "HAVCR2": "TIM-3", "LAG3": "LAG-3",
               "TNFRSF9": "4-1BB", "CD86": "B7-2", "CD28": "CD28"}
SHOW = ["Tumour (all)", "CD8 T (GZMK+)", "Treg", "Proliferating T", "NK", "B cell", "CD163+ macrophage", "FCGR1A+ macrophage",
        "CXCL9+ macrophage", "Alveolar macrophage", "cDC2", "mregDC (LAMP3+)", "Fibroblast", "Endothelial"]
group = np.where(is_tumour, "Tumour (all)", cell_type)
rates = pd.DataFrame({f"{g} ({p})": pd.Series(positive(g)).groupby(group).mean() * 100 for g, p in CHECKPOINTS.items()})
rates = rates.reindex([t for t in SHOW if t in rates.index])
step.save_table(rates.round(2), "step09_checkpoint_positivity")
step.say(f"  PD-L1 positive: tumour {rates.loc['Tumour (all)', 'CD274 (PD-L1)']:.1f}%, CXCL9+ macrophage "
         f"{rates.loc['CXCL9+ macrophage', 'CD274 (PD-L1)']:.0f}%, mregDC {rates.loc['mregDC (LAMP3+)', 'CD274 (PD-L1)']:.0f}%")

# ---- 9b. CD8 T-cell states by location ------------------------------------------------------------------
d_tumour = adata.obs["dist_to_tumour_um"].to_numpy()
in_tls = (adata.obs["tls_id"] != "none").to_numpy()
COMP = ["TLS", "Stroma (> 50 um)", "Border (15-50 um)", f"Tumour contact (<= {contact} um)"]
adata.obs["cd8_compartment"] = pd.Categorical(np.select([in_tls, d_tumour <= contact, d_tumour <= 50], [COMP[0], COMP[3], COMP[2]], COMP[1]),
                                              categories=COMP)
cd8 = (cell_type == "CD8 T (GZMK+)").to_numpy()
comp = adata.obs["cd8_compartment"].to_numpy()[cd8]
STATES = {"Cytotoxic": ["PRF1", "GZMA", "GZMB", "NKG7"], "Checkpoint / exhaustion": ["CTLA4", "HAVCR2", "LAG3", "TNFRSF9", "PDCD1"],
          "Activation": ["CD69"], "Naive / memory / homing": ["CCR7", "IL7R", "SELL"], "CONTROL: epithelial spillover": ["EPCAM", "MALL"]}
rows, pvals = [], []
for state, genes in STATES.items():
    for g in genes:
        p = positive(g)[cd8]
        table = pd.crosstab(comp, p).reindex(COMP).fillna(0)
        pvals.append(chi2_contingency(table.to_numpy())[1] if table.shape[1] == 2 else 1.0)
        rows.append({"state": state, "gene": g, **{c: 100 * p[comp == c].mean() for c in COMP}})
states = pd.DataFrame(rows)
states["FDR q"] = multipletests(pvals, method="fdr_bh")[1]
step.save_table(states.round(4), "step09_cd8_states", index=False)
g = states.set_index("gene")
step.say(f"  CD8 T, TLS -> tumour contact: CCR7 {g.loc['CCR7', COMP[0]]:.0f} -> {g.loc['CCR7', COMP[3]]:.0f}%, "
         f"GZMA {g.loc['GZMA', COMP[0]]:.0f} -> {g.loc['GZMA', COMP[3]]:.0f}%, EPCAM control "
         f"{g.loc['EPCAM', COMP[0]]:.1f} -> {g.loc['EPCAM', COMP[3]]:.1f}%")

# ---- 9c. Spatial ligand-receptor contact test --------------------------------------------------------------
LINEAGE = adata.obs["lineage"].astype(str).to_numpy()
PAIRS = [("CD274", "PDCD1", ["Epithelial", "Myeloid"], ["T / NK"], "checkpoint inhibition"),
         ("CD86", "CTLA4", ["Myeloid", "B / plasma"], ["T / NK"], "checkpoint inhibition"),
         ("CD86", "CD28", ["Myeloid", "B / plasma"], ["T / NK"], "co-stimulation"),
         ("CCL19", "CCR7", ["Stromal", "Myeloid", "T / NK", "Endothelial"], ["T / NK", "B / plasma", "Myeloid"], "lymphoid homing"),
         ("HLA-DQB2", "CD4", ["Myeloid", "B / plasma"], ["T / NK", "Myeloid"], "antigen presentation"),
         ("EDN1", "EDNRB", ["Epithelial", "Endothelial"], ["Endothelial", "Stromal"], "vascular")]
conn = adata.obsp["spatial_connectivities"].tocoo()
src, dst = conn.row, conn.col
codes = pd.factorize(cell_type)[0]
groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
rng = np.random.default_rng(0)


def shuffle_within_types(v):
    out = v.copy()
    for ix in groups:
        out[ix] = v[rng.permutation(ix)]
    return out


res = []
for lig, rec, send, recv, bio in PAIRS:
    L = positive(lig) & np.isin(LINEAGE, send)
    R = positive(rec) & np.isin(LINEAGE, recv)
    obs = int((L[src] & R[dst]).sum())
    null = np.array([(shuffle_within_types(L)[src] & shuffle_within_types(R)[dst]).sum() for _ in range(cfg["n_perms"])])
    res.append({"pair": f"{lig} -> {rec}", "biology": bio, "senders +": int(L.sum()), "receivers +": int(R.sum()),
                "observed contacts": obs, "expected": null.mean(), "obs / exp": obs / max(null.mean(), 1e-9),
                "z": (obs - null.mean()) / max(null.std(), 1e-9), "p (perm)": (1 + (null >= obs).sum()) / (cfg["n_perms"] + 1)})
lr = pd.DataFrame(res).set_index("pair")
lr["FDR q"] = multipletests(lr["p (perm)"], method="fdr_bh")[1]
step.save_table(lr.round(4), "step09_ligand_receptor_contacts")
step.say("  ligand-receptor contacts (obs/exp, q):  " + "; ".join(f"{p} {r['obs / exp']:.2f} ({r['FDR q']:.3f})" for p, r in lr.iterrows()))

# ---- 9d. Immune phenotype per tumour tile --------------------------------------------------------------------
area = adata.obs["cell_area"].to_numpy()
tile_key = pd.Series(list(map(tuple, np.floor(xy / cfg["tile_um"]).astype(int))), index=adata.obs_names)
df = pd.DataFrame({"tile": tile_key.to_numpy(), "tumour": is_tumour, "cd8_intra": cd8 & (d_tumour <= contact), "cd8_stroma": cd8 & (d_tumour > contact),
                   "area_tumour": area * is_tumour, "area_stroma": area * ~is_tumour, "programme": np.where(is_tumour, cell_type, "")})
tiles = df.groupby("tile").agg(tumour=("tumour", "sum"), cd8_intra=("cd8_intra", "sum"), cd8_stroma=("cd8_stroma", "sum"),
                               area_tumour=("area_tumour", "sum"), area_stroma=("area_stroma", "sum"))
tiles["programme"] = df[df.tumour].groupby("tile")["programme"].agg(lambda s: s.value_counts().index[0])
tiles = tiles[tiles["tumour"] >= 50].copy()
tiles["intra_density"] = tiles["cd8_intra"] / (tiles["area_tumour"] / 1e6)
tiles["stroma_density"] = tiles["cd8_stroma"] / (tiles["area_stroma"] / 1e6).clip(lower=1e-9)
tiles["total_density"] = (tiles["cd8_intra"] + tiles["cd8_stroma"]) / ((tiles["area_tumour"] + tiles["area_stroma"]) / 1e6)
classify = lambda t, d, r: np.select([t["total_density"] < d, t["intra_density"] >= r * t["stroma_density"]], ["desert", "inflamed"], "excluded")  # noqa: E731
tiles["phenotype"] = classify(tiles, cfg["desert_cd8_per_mm2"], cfg["inflamed_ratio"])
PHENO = ["inflamed", "excluded", "desert"]
by_prog = (pd.crosstab(tiles["programme"], tiles["phenotype"], normalize="index") * 100).reindex(columns=PHENO).fillna(0)
by_prog.loc["All tumour tiles"] = tiles["phenotype"].value_counts(normalize=True).reindex(PHENO).fillna(0) * 100
step.save_table(by_prog.round(1), "step09_immune_phenotypes")
sens = pd.DataFrame({f"desert<{d}, ratio>={r}": pd.Series(classify(tiles, d, r)).value_counts(normalize=True).reindex(PHENO).fillna(0) * 100
                     for d in (25, 50, 100) for r in (0.3, 0.5, 0.7)}).T
step.save_table(sens.round(1), "step09_immune_phenotypes_sensitivity")
step.say("  tumour tiles: " + ", ".join(f"{p} {by_prog.loc['All tumour tiles', p]:.0f}%" for p in PHENO))
COLORS = {"inflamed": "#eb6834", "excluded": "#2a78d6", "desert": "#c3c2b7"}
fig, ax = plt.subplots(figsize=(15, 4.8))
ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=0.03, c="#e3e6ea", linewidths=0, rasterized=True)
t_um = cfg["tile_um"] / 1000
for (tx, ty), r in tiles.iterrows():
    ax.add_patch(Rectangle((tx * t_um, ty * t_um), t_um, t_um, facecolor=COLORS[r.phenotype], alpha=0.75, edgecolor="white", lw=0.8))
for p in PHENO:
    ax.scatter([], [], marker="s", s=60, c=COLORS[p], label=f"{p} ({(tiles.phenotype == p).sum()} tiles)")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1))
ax.set_title(f"CD8 immune phenotype per {cfg['tile_um']} um tumour tile", loc="left")
step.save_fig(fig, "step09_immune_phenotypes")

# ---- 9e. Immunosuppressive balance with distance ----------------------------------------------------------------
BANDS = ["TLS", f"<= {contact} um", f"{contact}-50 um", "50-100 um", "> 100 um"]
band = np.select([in_tls, d_tumour <= contact, d_tumour <= 50, d_tumour <= 100], BANDS[:4], BANDS[4])
RATIOS = {"Treg : CD8 T": ("Treg", "CD8 T (GZMK+)"), "CD163+ : CXCL9+ macrophages": ("CD163+ macrophage", "CXCL9+ macrophage")}
bal = pd.DataFrame({name: [((cell_type == a).to_numpy() & (band == b)).sum() / max(((cell_type == d).to_numpy() & (band == b)).sum(), 1)
                           for b in BANDS] for name, (a, d) in RATIOS.items()}, index=BANDS)
step.save_table(bal.round(3), "step09_suppressive_ratios")
step.say(f"  Treg:CD8 TLS {bal.iloc[0, 0]:.2f} -> tumour contact {bal.iloc[1, 0]:.2f}")

adata.obs["immune_tile_phenotype"] = pd.Categorical(tile_key.map(tiles["phenotype"].to_dict()).fillna("not scored"), categories=PHENO + ["not scored"])
adata.uns["step9_params"] = {k: v for k, v in cfg.items()}
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

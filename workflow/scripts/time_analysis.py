"""Step 9: tumour immune microenvironment deep-dive (ist_analysis.tme).

Checkpoint positivity rates within cell types, CD8 T-cell states by location, the spatial ligand-receptor
contact test, immune phenotype per tumour tile, and immunosuppressive ratios by distance.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

from common import Step, xu
from ist_analysis import tme

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

step = Step("Step 9: tumour immune microenvironment", inputs=["input"], outputs=["output"])
cfg = step.config["time"]
contact = step.config["spatial"]["contact_um"]

adata = sc.read_h5ad(step.path("input"))
positive = tme.Positivity(adata)
cell_type = adata.obs["cell_type"].astype(str)
is_tumour = cell_type.isin(xu.TUMOUR_TYPES).to_numpy()

SHOW = ["Tumour (all)", "CD8 T (GZMK+)", "Treg", "Proliferating T", "NK", "B cell", "CD163+ macrophage", "FCGR1A+ macrophage",
        "CXCL9+ macrophage", "Alveolar macrophage", "cDC2", "mregDC (LAMP3+)", "Fibroblast", "Endothelial"]
rates = tme.positivity_rates(positive, np.where(is_tumour, "Tumour (all)", cell_type), tme.CHECKPOINTS, SHOW)
step.save_table(rates.round(2), "step09_checkpoint_positivity")
step.say(f"  PD-L1 positive: tumour {rates.loc['Tumour (all)', 'CD274 (PD-L1)']:.1f}%, CXCL9+ macrophage "
         f"{rates.loc['CXCL9+ macrophage', 'CD274 (PD-L1)']:.0f}%, mregDC {rates.loc['mregDC (LAMP3+)', 'CD274 (PD-L1)']:.0f}%")

tme.add_cd8_compartments(adata, contact)
states = tme.cd8_states(adata, positive)
step.save_table(states.round(4), "step09_cd8_states", index=False)
g = states.set_index("gene")
first, last = tme.COMPARTMENTS[0], tme.COMPARTMENTS[-1]
step.say(f"  CD8 T, TLS -> tumour contact: CCR7 {g.loc['CCR7', first]:.0f} -> {g.loc['CCR7', last]:.0f}%, "
         f"GZMA {g.loc['GZMA', first]:.0f} -> {g.loc['GZMA', last]:.0f}%, EPCAM control {g.loc['EPCAM', first]:.1f} -> {g.loc['EPCAM', last]:.1f}%")

lr = tme.contact_test(adata, positive, n_perms=cfg["n_perms"])
step.save_table(lr.round(4), "step09_ligand_receptor_contacts")
step.say("  ligand-receptor contacts (obs/exp, q):  " + "; ".join(f"{p} {r['obs / exp']:.2f} ({r['FDR q']:.3f})" for p, r in lr.iterrows()))

tiles, tile_key = tme.tile_phenotypes(adata, is_tumour, cfg["tile_um"], contact_um=contact)
tiles["phenotype"] = tme.classify_tiles(tiles, cfg["desert_cd8_per_mm2"], cfg["inflamed_ratio"])
by_prog = tme.phenotype_by_programme(tiles)
step.save_table(by_prog.round(1), "step09_immune_phenotypes")
step.save_table(tme.sensitivity(tiles).round(1), "step09_immune_phenotypes_sensitivity")
step.say("  tumour tiles: " + ", ".join(f"{p} {by_prog.loc['All tumour tiles', p]:.0f}%" for p in tme.PHENOTYPES))
COLORS = {"inflamed": "#eb6834", "excluded": "#2a78d6", "desert": "#c3c2b7"}
xy = adata.obsm["spatial"]
fig, ax = plt.subplots(figsize=(15, 4.8))
ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=0.03, c="#e3e6ea", linewidths=0, rasterized=True)
t_um = cfg["tile_um"] / 1000
for (tx, ty), r in tiles.iterrows():
    ax.add_patch(Rectangle((tx * t_um, ty * t_um), t_um, t_um, facecolor=COLORS[r.phenotype], alpha=0.75, edgecolor="white", lw=0.8))
for p in tme.PHENOTYPES:
    ax.scatter([], [], marker="s", s=60, c=COLORS[p], label=f"{p} ({(tiles.phenotype == p).sum()} tiles)")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1))
ax.set_title(f"CD8 immune phenotype per {cfg['tile_um']} um tumour tile", loc="left")
step.save_fig(fig, "step09_immune_phenotypes")

bal = pd.concat({name: tme.ratio_by_band(adata, a, d, contact).set_index("location")["value"] for name, (a, d) in
                 {"Treg : CD8 T": ("Treg", "CD8 T (GZMK+)"), "CD163+ : CXCL9+ macrophages": ("CD163+ macrophage", "CXCL9+ macrophage")}.items()}, axis=1)
step.save_table(bal.round(3), "step09_suppressive_ratios")
step.say(f"  Treg:CD8 TLS {bal.iloc[0, 0]:.2f} -> tumour contact {bal.iloc[1, 0]:.2f}")

adata.obs["immune_tile_phenotype"] = pd.Categorical(tile_key.map(tiles["phenotype"].to_dict()).fillna("not scored"),
                                                    categories=tme.PHENOTYPES + ["not scored"])
adata.uns["step9_params"] = dict(cfg)
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

"""Step 7: spatial analysis of the tumour immune microenvironment (ist_analysis.spatial).

Delaunay neighbour graph, neighbourhood enrichment, composition niches, distance to the nearest tumour
cell, and TLS-like B-cell aggregates, on high-confidence cells.
"""

import matplotlib
import scanpy as sc
from matplotlib.patches import Circle

from common import Step, xu
from ist_analysis import spatial

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 7: spatial analysis", inputs=["input"], outputs=["output"])
cfg = step.config["spatial"]

full = sc.read_h5ad(step.path("input"))
adata = spatial.high_confidence(full)
step.say(f"  {adata.n_obs:,} high-confidence cells of {full.n_obs:,}")
g = spatial.delaunay_graph(adata, cfg["max_edge_um"])
step.say(f"  Delaunay: {g['frac_removed']:.1%} of edges > {cfg['max_edge_um']} um removed; median {g['median_degree']:.0f} neighbours per cell")

order = spatial.add_time_types(adata)
for key, o, clip in [("lineage", None, 50), ("time_type", order, 30)]:
    z = spatial.nhood_zscores(adata, key, cfg["n_perms"], order=o)
    step.save_table(z.round(2), f"step07_nhood_enrichment_{key}")
    fig, ax = plt.subplots(figsize=(10, 8.5) if o else (7.5, 6))
    im = ax.imshow(z.clip(-clip, clip).to_numpy(), cmap="RdBu_r", vmin=-clip, vmax=clip)
    ax.set_xticks(range(len(z)), z.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(z)), z.index, fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label=f"z-score (clipped at +/-{clip})")
    ax.set_title(f"Neighbourhood enrichment ({key})", loc="left")
    step.save_fig(fig, f"step07_nhood_enrichment_{key}")

centres, _ = spatial.composition_niches(adata, cfg["niche_radius_um"], cfg["n_niches"])
step.save_table(centres.round(3), "step07_niches")

is_tumour = spatial.distance_to_tumour(adata)
infil, _, base_contact = spatial.infiltration_table(adata, is_tumour, cfg["contact_um"])
step.save_table(infil.round(1), "step07_infiltration")
col = f"% within {cfg['contact_um']} um"
step.say(f"  baseline: {base_contact:.0%} of non-tumour cells within {cfg['contact_um']} um; CD8 T: "
         f"{infil.loc['CD8 T (GZMK+)', col]:.0f}% ({infil.loc['CD8 T (GZMK+)', 'vs. baseline']})")

tls, _, _ = spatial.find_tls(adata, cfg["tls_eps_um"], cfg["tls_min_cells"])
step.save_table(tls.round(2), "step07_tls")
step.say(f"  {len(tls)} TLS-like aggregates; largest at ({tls['x (mm)'].iloc[0]:.2f}, {tls['y (mm)'].iloc[0]:.2f}) mm")

xy = adata.obsm["spatial"]
fig, ax = plt.subplots(figsize=(15, 5.2))
for lin, colr in xu.LINEAGE_COLORS.items():
    m = (adata.obs["lineage"] == lin).to_numpy()
    ax.scatter(xy[m, 0] / 1000, xy[m, 1] / 1000, s=0.15, c=colr, linewidths=0, rasterized=True, label=lin)
for name, r in tls.iterrows():
    ax.add_patch(Circle((r["x (mm)"], r["y (mm)"]), r["radius (um)"] / 1000 + 0.05, fill=False, lw=1.2))
    ax.annotate(name, (r["x (mm)"] - r["radius (um)"] / 1000 - 0.08, r["y (mm)"]), ha="right", fontsize=8.5, weight="bold")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.legend(markerscale=30, frameon=False, ncol=7, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
ax.set_title("TLS-like aggregates (circled) on the lineage map", loc="left")
step.save_fig(fig, "step07_tls")

adata.uns["step7_params"] = dict(cfg)
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

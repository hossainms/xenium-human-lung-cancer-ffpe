"""Step 6: cell-type annotation from config/annotation.yaml (ist_analysis.annotate).

Each subcluster label carries a guard marker that must be among the subcluster's top-10 markers, so a
changed clustering stops the pipeline instead of mislabelling cells.
"""

import matplotlib
import scanpy as sc

from common import REPO_ROOT, Step, xu
from ist_analysis import annotate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 6: annotate cell types", inputs=["input"], outputs=["output"])
key = step.config["cluster"]["working"]
ann = annotate.load_annotation(REPO_ROOT / step.config["annotation_file"])

adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"]
z, subs, tables = annotate.annotate(adata, ann, key, seed=step.config["cluster"]["seed"])
step.save_table(z.round(2), "step06_marker_scores_by_cluster")
for name, table in tables.items():
    step.save_table(table, f"step06_subclusters_{name}")
    step.say(f"  {name}: {subs[name].n_obs:,} cells in {subs[name].obs['sub'].nunique()} subclusters, labels verified")
step.save_table(annotate.composition(adata), "step06_cell_types", index=False)
step.say(f"  {adata.obs['cell_type'].nunique()} cell types; {(adata.obs['annotation_confidence'] == 'mixed').mean():.1%} flagged mixed")

xy = adata.obsm["spatial"] / 1000
fig, ax = plt.subplots(figsize=(15, 5.6))
for lin, col in xu.LINEAGE_COLORS.items():
    m = (adata.obs["lineage"] == lin).to_numpy()
    ax.scatter(xy[m, 0], xy[m, 1], s=0.25, c=col, linewidths=0, rasterized=True, label=f"{lin} ({m.sum():,})")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.legend(markerscale=12, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
ax.set_title("Cell lineages in the tissue", loc="left")
step.save_fig(fig, "step06_lineages")

adata.uns["annotation"] = {"coarse_from": key, "subcluster_resolution": ann["subcluster_resolution"], "label_tables": "config/annotation.yaml",
                           "caveats": "Panel lacks SFTPC/NAPSA/KRT5/TP63/SCGB1A1; tumour call inferred, not CNV-confirmed."}
adata.uns["lineage_colors"] = [xu.LINEAGE_COLORS[l] for l in adata.obs["lineage"].cat.categories]
out = step.path("output")
adata.write_h5ad(out, compression="gzip")
step.done(out)

"""Step 10: spatial statistics with squidpy (ist_analysis.stats).

Moran's I (analytic p-values: permutations take >10 min on ~140k cells), co-occurrence by distance and
Ripley's L on seeded subsamples, graph centrality, interaction matrix, and ligand-receptor analysis
(squidpy ligrec, OmniPath) compared with the Step 9 spatial contact test.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from common import Step, xu
from ist_analysis import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 10: spatial statistics (squidpy)", inputs=["input", "contacts"], outputs=["marker"])
cfg = step.config["stats"]
seed = cfg["seed"]
adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"]

sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None, show_progress_bar=False)
moran = adata.uns["moranI"]
step.save_table(moran, "step10_moran_i")
step.say(f"  Moran's I top genes: {', '.join(moran.index[:5])}; CXCL9 I = {moran.loc['CXCL9', 'I']:.2f} "
         f"(rank {moran.index.get_loc('CXCL9') + 1}); PDCD1 I = {moran.loc['PDCD1', 'I']:.3f}")

sub = adata[np.random.default_rng(seed).choice(adata.n_obs, cfg["subsample"], replace=False)].copy()
sq.gr.co_occurrence(sub, cluster_key="lineage", interval=np.linspace(5, 300, 25), show_progress_bar=False)
occ, dist = sub.uns["lineage_co_occurrence"]["occ"], sub.uns["lineage_co_occurrence"]["interval"][1:]
lineages = list(sub.obs["lineage"].cat.categories)
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, anchor in zip(axes, ["T / NK", "B / plasma", "Myeloid"]):
    i = lineages.index(anchor)
    for j, lin in enumerate(lineages):
        ax.plot(dist, occ[i, j], color=xu.LINEAGE_COLORS[lin], lw=2 if lin in (anchor, "Epithelial") else 1.2, label=lin)
    ax.axhline(1, color="#898781", ls="--", lw=1)
    ax.set_title(f"Around {anchor} cells", loc="left")
    ax.set_xlabel("distance (um)")
axes[0].set_ylabel("co-occurrence ratio")
axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
step.save_fig(fig, "step10_co_occurrence")
step.save_table(stats.co_occurrence_epithelium(sub).round(3), "step10_co_occurrence_epithelium")

ripley, _, _ = stats.ripley_balanced(adata, seed=seed)
step.save_table(ripley.round(3).to_frame(), "step10_ripley_L")

sq.gr.centrality_scores(adata, cluster_key="lineage", show_progress_bar=False)
step.save_table(adata.uns["lineage_centrality_scores"].sort_values("closeness_centrality", ascending=False).round(4), "step10_centrality")
sq.gr.interaction_matrix(adata, cluster_key="lineage", normalized=True)
cats = list(adata.obs["lineage"].cat.categories)
step.save_table(pd.DataFrame(adata.uns["lineage_interactions"], index=cats, columns=cats).round(4), "step10_interaction_matrix")

lr = sq.gr.ligrec(adata, cluster_key="time_type", use_raw=False, n_perms=1000, threshold=0.05, seed=seed, copy=True,
                  corr_method="fdr_bh", show_progress_bar=False, n_jobs=4)
means, qvals = stats.ligrec_dense(lr)
long = qvals.stack([0, 1], future_stack=True).rename("q").to_frame().join(means.stack([0, 1], future_stack=True).rename("mean"))
step.save_table(long[long["q"] < 0.05].sort_values("mean", ascending=False), "step10_ligrec_significant")
step.save_table(stats.ligrec_vs_contact(means, qvals, pd.read_csv(step.path("contacts"), index_col=0)), "step10_ligrec_vs_contact_test")
step.say(f"  ligrec: {means.shape[0]} OmniPath pairs testable on the panel")

marker = step.path("marker")
marker.write_text("step 10 complete\n")
step.done(marker)

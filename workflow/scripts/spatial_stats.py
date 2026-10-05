"""Step 10: spatial statistics with squidpy.

Moran's I (analytic p-values: permutations take >10 min on ~140k cells), co-occurrence by distance and
Ripley's L (seeded subsamples), graph centrality, interaction matrix, and ligand-receptor analysis
(squidpy.gr.ligrec with OmniPath), compared against the Step 9 spatial contact test.
"""

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from common import Step, xu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

step = Step("Step 10: spatial statistics (squidpy)", inputs=["input", "contacts"], outputs=["marker"])
cfg = step.config["stats"]
seed = cfg["seed"]
adata = sc.read_h5ad(step.path("input"))
adata.X = adata.layers["lognorm"]
colors = xu.LINEAGE_COLORS

# ---- Moran's I ---------------------------------------------------------------------------------
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None, show_progress_bar=False)
moran = adata.uns["moranI"]
step.save_table(moran, "step10_moran_i")
step.say(f"  Moran's I top genes: {', '.join(moran.index[:5])}; CXCL9 I = {moran.loc['CXCL9', 'I']:.2f} "
         f"(rank {moran.index.get_loc('CXCL9') + 1}); PDCD1 I = {moran.loc['PDCD1', 'I']:.3f}")

# ---- Co-occurrence of lineages by distance (subsample) ---------------------------------------------
sub = adata[np.random.default_rng(seed).choice(adata.n_obs, cfg["subsample"], replace=False)].copy()
sq.gr.co_occurrence(sub, cluster_key="lineage", interval=np.linspace(5, 300, 25), show_progress_bar=False)
occ, dist = sub.uns["lineage_co_occurrence"]["occ"], sub.uns["lineage_co_occurrence"]["interval"][1:]
lineages = list(sub.obs["lineage"].cat.categories)
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, anchor in zip(axes, ["T / NK", "B / plasma", "Myeloid"]):
    i = lineages.index(anchor)
    for j, lin in enumerate(lineages):
        ax.plot(dist, occ[i, j], color=colors[lin], lw=2 if lin in (anchor, "Epithelial") else 1.2, label=lin)
    ax.axhline(1, color="#898781", ls="--", lw=1)
    ax.set_title(f"Around {anchor} cells", loc="left")
    ax.set_xlabel("distance (um)")
axes[0].set_ylabel("co-occurrence ratio")
axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
step.save_fig(fig, "step10_co_occurrence")
e = lineages.index("Epithelial")
at = [np.argmin(np.abs(dist - r)) for r in (20, 100, 300)]
co = pd.DataFrame({a: occ[lineages.index(a), e, at] for a in ["T / NK", "B / plasma", "Myeloid"]},
                  index=[f"epithelium at ~{dist[i]:.0f} um" for i in at]).T
step.save_table(co.round(3), "step10_co_occurrence_epithelium")

# ---- Ripley's L on a balanced subsample (equal cells per lineage) --------------------------------------
n = 2000
rng = np.random.default_rng(seed)
idx = np.concatenate([rng.choice(np.flatnonzero(adata.obs["lineage"].to_numpy() == lin), n, replace=False)
                      for lin in adata.obs["lineage"].cat.categories])
bal = adata[idx].copy()
sq.gr.ripley(bal, cluster_key="lineage", mode="L", max_dist=300, n_steps=31, n_simulations=100, n_observations=n, seed=seed)
L, sims = bal.uns["lineage_ripley_L"]["L_stat"], bal.uns["lineage_ripley_L"]["sims_stat"]
hi = sims.groupby("bins")["stats"].quantile(0.975)
at100 = L[L["bins"].sub(100).abs() == L["bins"].sub(100).abs().min()].set_index("lineage")["stats"]
ripley = (at100 / hi.iloc[np.argmin(np.abs(hi.index - 100))]).sort_values(ascending=False).rename("L(100 um) / random upper band")
step.save_table(ripley.round(3).to_frame(), "step10_ripley_L")

# ---- Graph centrality and interaction matrix -------------------------------------------------------------
sq.gr.centrality_scores(adata, cluster_key="lineage", show_progress_bar=False)
step.save_table(adata.uns["lineage_centrality_scores"].sort_values("closeness_centrality", ascending=False).round(4), "step10_centrality")
sq.gr.interaction_matrix(adata, cluster_key="lineage", normalized=True)
cats = list(adata.obs["lineage"].cat.categories)
step.save_table(pd.DataFrame(adata.uns["lineage_interactions"], index=cats, columns=cats).round(4), "step10_interaction_matrix")

# ---- Ligand-receptor analysis (OmniPath), compared with the Step 9 contact test --------------------------
lr = sq.gr.ligrec(adata, cluster_key="time_type", use_raw=False, n_perms=1000, threshold=0.05, seed=seed, copy=True,
                  corr_method="fdr_bh", show_progress_bar=False, n_jobs=4)
means = lr["means"].sparse.to_dense().astype(float)    # ligrec returns sparse frames: densify before indexing
qvals = lr["pvalues"].sparse.to_dense().astype(float)
long = qvals.stack([0, 1], future_stack=True).rename("q").to_frame().join(means.stack([0, 1], future_stack=True).rename("mean"))
step.save_table(long[long["q"] < 0.05].sort_values("mean", ascending=False), "step10_ligrec_significant")
contacts = pd.read_csv(step.path("contacts"), index_col=0)
rows = []
for pair, r in contacts.iterrows():
    lig, rec = pair.split(" -> ")
    if (lig, rec) in means.index:
        best = pd.DataFrame({"mean": means.loc[(lig, rec)], "q": qvals.loc[(lig, rec)]}).dropna()
        call = f"significant in {(best['q'] < 0.05).sum()} of {len(best)} type pairs" if len(best) else "below expression threshold"
    else:
        call = "pair not in OmniPath"
    rows.append({"pair": pair, "ligrec (expression)": call, "contact obs/exp": r["obs / exp"], "contact FDR q": r["FDR q"]})
step.save_table(pd.DataFrame(rows).set_index("pair"), "step10_ligrec_vs_contact_test")
step.say(f"  ligrec: {means.shape[0]} OmniPath pairs testable on the panel")

marker = step.path("marker")
marker.write_text("step 10 complete\n")
step.done(marker)

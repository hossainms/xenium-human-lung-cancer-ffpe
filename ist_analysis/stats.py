"""Spatial statistics helpers around squidpy (Step 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def co_occurrence_epithelium(sub, anchors=("T / NK", "B / plasma", "Myeloid"), distances=(20, 100, 300), target: str = "Epithelial"):
    """Co-occurrence ratio of `target` around each anchor lineage at chosen distances (after sq.gr.co_occurrence)."""
    occ = sub.uns["lineage_co_occurrence"]["occ"]
    dist = sub.uns["lineage_co_occurrence"]["interval"][1:]
    lineages = list(sub.obs["lineage"].cat.categories)
    e = lineages.index(target)
    at = [int(np.argmin(np.abs(dist - r))) for r in distances]
    return pd.DataFrame({a: occ[lineages.index(a), e, at] for a in anchors},
                        index=[f"epithelium within ~{dist[i]:.0f} um" for i in at]).T


def ripley_balanced(adata, n_per_lineage: int = 2000, max_dist: float = 300, n_simulations: int = 100, seed: int = 0):
    """Ripley's L per lineage on equal cell numbers (squidpy normalises L by the TOTAL cell count, so on raw
    data L tracks abundance, not clustering). Returns (L(100 um) / upper 95% random band, L table, simulations)."""
    import squidpy as sq

    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(adata.obs["lineage"].to_numpy() == lin), n_per_lineage, replace=False)
                          for lin in adata.obs["lineage"].cat.categories])
    bal = adata[idx].copy()
    sq.gr.ripley(bal, cluster_key="lineage", mode="L", max_dist=max_dist, n_steps=31, n_simulations=n_simulations,
                 n_observations=n_per_lineage, seed=seed)
    L, sims = bal.uns["lineage_ripley_L"]["L_stat"], bal.uns["lineage_ripley_L"]["sims_stat"]
    hi = sims.groupby("bins")["stats"].quantile(0.975)
    at100 = L[L["bins"].sub(100).abs() == L["bins"].sub(100).abs().min()].set_index("lineage")["stats"]
    ratio = (at100 / hi.iloc[np.argmin(np.abs(hi.index - 100))]).sort_values(ascending=False).rename("L(100 um) / upper random band")
    return ratio, L, sims


def ligrec_dense(lr: dict):
    """squidpy.gr.ligrec returns SPARSE DataFrames: densify before indexing, otherwise lookups silently return nothing."""
    return lr["means"].sparse.to_dense().astype(float), lr["pvalues"].sparse.to_dense().astype(float)


def ligrec_vs_contact(means: pd.DataFrame, qvals: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    """For each pair of the spatial contact test: what expression-based ligrec says, next to the contact result."""
    rows = []
    for pair, r in contacts.iterrows():
        lig, rec = pair.split(" -> ")
        if (lig, rec) in means.index:
            best = pd.DataFrame({"mean": means.loc[(lig, rec)], "q": qvals.loc[(lig, rec)]}).dropna().sort_values("mean", ascending=False)
            sig = best[best["q"] < 0.05]
            top = f"{sig.index[0][0]} -> {sig.index[0][1]}" if len(sig) else "-"
            call = f"significant in {len(sig)} of {len(best)} type pairs" if len(best) else "in OmniPath, below expression threshold"
        else:
            top, call = "-", "pair not in the OmniPath ligand-receptor list"
        rows.append({"pair": pair, "ligrec (expression)": call, "ligrec top sender -> receiver": top,
                     "contact test obs/exp": round(r["obs / exp"], 2), "contact test FDR q": round(r["FDR q"], 4)})
    return pd.DataFrame(rows).set_index("pair")

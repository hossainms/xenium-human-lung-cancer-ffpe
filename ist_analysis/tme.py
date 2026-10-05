"""Tumour immune microenvironment: positivity rates, CD8 T states by location, spatial ligand-receptor
contact test, immune phenotypes per tumour tile, immunosuppressive ratios (Step 9).

Rule used throughout: compare the % of positive cells WITHIN a cell type against the tumour noise floor,
never counts of positive cells (tumour cells are ~60% of all cells).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHECKPOINTS = {"PDCD1": "PD-1", "CD274": "PD-L1", "CTLA4": "CTLA-4", "HAVCR2": "TIM-3", "LAG3": "LAG-3",
               "TNFRSF9": "4-1BB", "CD86": "B7-2", "CD28": "CD28"}
CD8_STATES = {"Cytotoxic": ["PRF1", "GZMA", "GZMB", "NKG7"], "Checkpoint / exhaustion": ["CTLA4", "HAVCR2", "LAG3", "TNFRSF9", "PDCD1"],
              "Activation": ["CD69"], "Naive / memory / homing": ["CCR7", "IL7R", "SELL"],
              "CONTROL: epithelial spillover": ["EPCAM", "MALL"]}
COMPARTMENTS = ["TLS", "Stroma (> 50 um)", "Border (15-50 um)", "Tumour contact (<= 15 um)"]
LR_PAIRS = [  # (ligand, receptor, sender lineages, receiver lineages, biology)
    ("CD274", "PDCD1", ["Epithelial", "Myeloid"], ["T / NK"], "PD-L1 -> PD-1: checkpoint inhibition"),
    ("CD86", "CTLA4", ["Myeloid", "B / plasma"], ["T / NK"], "B7-2 -> CTLA-4: checkpoint inhibition"),
    ("CD86", "CD28", ["Myeloid", "B / plasma"], ["T / NK"], "B7-2 -> CD28: T-cell co-stimulation"),
    ("CCL19", "CCR7", ["Stromal", "Myeloid", "T / NK", "Endothelial"], ["T / NK", "B / plasma", "Myeloid"], "CCL19 -> CCR7: lymphoid homing (TLS)"),
    ("HLA-DQB2", "CD4", ["Myeloid", "B / plasma"], ["T / NK", "Myeloid"], "MHC-II -> CD4: antigen presentation"),
    ("EDN1", "EDNRB", ["Epithelial", "Endothelial"], ["Endothelial", "Stromal"], "Endothelin-1 -> EDNRB: vascular"),
]
PHENOTYPES = ["inflamed", "excluded", "desert"]


class Positivity:
    """>= 1 transcript per cell, for any gene, from raw counts (column-compressed once)."""

    def __init__(self, adata):
        self.counts = adata.layers["counts"].tocsc()
        self.var_names = adata.var_names

    def __call__(self, gene: str) -> np.ndarray:
        return np.asarray(self.counts[:, self.var_names.get_loc(gene)].todense()).ravel() > 0


def positivity_rates(positive: Positivity, group: np.ndarray, genes: dict, rows=None) -> pd.DataFrame:
    """% positive per gene within each group (e.g. cell type, with all tumour cells pooled as the noise floor)."""
    rates = pd.DataFrame({f"{g} ({p})": pd.Series(positive(g)).groupby(group).mean() * 100 for g, p in genes.items()})
    return rates.loc[[r for r in rows if r in rates.index]] if rows else rates


def above_noise_floor(rates: pd.DataFrame, floor_row: str = "Tumour (all)", fold: float = 3, min_pct: float = 5) -> pd.Series:
    """Per gene: groups above fold x the noise floor (floor clipped at 0.5%) and >= min_pct."""
    floor = rates.loc[floor_row].clip(lower=0.5)
    rest = rates.drop(index=floor_row)
    enriched = (rest > fold * floor) & (rest >= min_pct)
    return pd.Series({c: ", ".join(enriched.index[enriched[c]]) or "none above 3x noise floor" for c in rates.columns})


def add_cd8_compartments(adata, contact_um: float = 15) -> None:
    """obs['cd8_compartment']: TLS / stroma (> 50 um) / border (15-50 um) / tumour contact (<= 15 um)."""
    d = adata.obs["dist_to_tumour_um"].to_numpy()
    in_tls = (adata.obs["tls_id"] != "none").to_numpy()
    adata.obs["cd8_compartment"] = pd.Categorical(
        np.select([in_tls, d <= contact_um, d <= 50], [COMPARTMENTS[0], COMPARTMENTS[3], COMPARTMENTS[2]], COMPARTMENTS[1]),
        categories=COMPARTMENTS)


def cd8_states(adata, positive: Positivity, states: dict = CD8_STATES, cell_type: str = "CD8 T (GZMK+)") -> pd.DataFrame:
    """% of CD8 T cells positive per gene and compartment, chi-square across compartments, BH-corrected."""
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    cd8 = (adata.obs["cell_type"] == cell_type).to_numpy()
    comp = adata.obs["cd8_compartment"].to_numpy()[cd8]
    rows, pvals = [], []
    for state, genes in states.items():
        for g in genes:
            p = positive(g)[cd8]
            table = pd.crosstab(comp, p).reindex(COMPARTMENTS).fillna(0)
            pvals.append(chi2_contingency(table.to_numpy())[1] if table.shape[1] == 2 else 1.0)
            rows.append({"state": state, "gene": g, **{c: 100 * p[comp == c].mean() for c in COMPARTMENTS}})
    out = pd.DataFrame(rows)
    out["FDR q"] = multipletests(pvals, method="fdr_bh")[1]
    return out


def contact_test(adata, positive: Positivity, pairs=LR_PAIRS, n_perms: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Spatial ligand-receptor contact test on the neighbour graph.

    Counts graph edges where a ligand+ sender touches a receptor+ receiver. Null: each gene's positivity is
    shuffled WITHIN each cell type (keeps composition and per-type expression rates), n_perms times.
    """
    from statsmodels.stats.multitest import multipletests

    lineage = adata.obs["lineage"].astype(str).to_numpy()
    conn = adata.obsp["spatial_connectivities"].tocoo()
    src, dst = conn.row, conn.col                     # both directions of every undirected edge
    codes = pd.factorize(adata.obs["cell_type"].astype(str))[0]
    groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)

    def shuffle(v):
        out = v.copy()
        for ix in groups:
            out[ix] = v[rng.permutation(ix)]
        return out

    res = []
    for lig, rec, send, recv, bio in pairs:
        L = positive(lig) & np.isin(lineage, send)
        R = positive(rec) & np.isin(lineage, recv)
        obs = int((L[src] & R[dst]).sum())
        null = np.array([(shuffle(L)[src] & shuffle(R)[dst]).sum() for _ in range(n_perms)])
        res.append({"pair": f"{lig} -> {rec}", "biology": bio, "senders +": int(L.sum()), "receivers +": int(R.sum()),
                    "observed contacts": obs, "expected": null.mean(), "obs / exp": obs / max(null.mean(), 1e-9),
                    "z": (obs - null.mean()) / max(null.std(), 1e-9), "p (perm)": (1 + (null >= obs).sum()) / (n_perms + 1)})
    lr = pd.DataFrame(res).set_index("pair")
    lr["FDR q"] = multipletests(lr["p (perm)"], method="fdr_bh")[1]
    return lr


def tile_phenotypes(adata, is_tumour: np.ndarray, tile_um: float = 400, min_tumour: int = 50, contact_um: float = 15,
                    cell_type: str = "CD8 T (GZMK+)") -> tuple[pd.DataFrame, pd.Series]:
    """CD8 densities per tumour tile (cells per mm^2 of cell area), intratumoral (<= contact) vs stromal.

    Returns (tile table, tile key per cell). Classify with classify_tiles().
    """
    xy = adata.obsm["spatial"]
    d = adata.obs["dist_to_tumour_um"].to_numpy()
    cd8 = (adata.obs["cell_type"] == cell_type).to_numpy()
    area = adata.obs["cell_area"].to_numpy()
    cell_type_str = adata.obs["cell_type"].astype(str)
    tile_key = pd.Series(list(map(tuple, np.floor(xy / tile_um).astype(int))), index=adata.obs_names)
    df = pd.DataFrame({"tile": tile_key.to_numpy(), "tumour": is_tumour, "cd8_intra": cd8 & (d <= contact_um), "cd8_stroma": cd8 & (d > contact_um),
                       "area_tumour": area * is_tumour, "area_stroma": area * ~is_tumour, "programme": np.where(is_tumour, cell_type_str, "")})
    tiles = df.groupby("tile").agg(tumour=("tumour", "sum"), cd8_intra=("cd8_intra", "sum"), cd8_stroma=("cd8_stroma", "sum"),
                                   area_tumour=("area_tumour", "sum"), area_stroma=("area_stroma", "sum"))
    tiles["programme"] = df[df.tumour].groupby("tile")["programme"].agg(lambda s: s.value_counts().index[0])
    tiles = tiles[tiles["tumour"] >= min_tumour].copy()
    tiles["intra_density"] = tiles["cd8_intra"] / (tiles["area_tumour"] / 1e6)
    tiles["stroma_density"] = tiles["cd8_stroma"] / (tiles["area_stroma"] / 1e6).clip(lower=1e-9)
    tiles["total_density"] = (tiles["cd8_intra"] + tiles["cd8_stroma"]) / ((tiles["area_tumour"] + tiles["area_stroma"]) / 1e6)
    return tiles, tile_key


def classify_tiles(tiles: pd.DataFrame, desert: float = 50, ratio: float = 0.5) -> np.ndarray:
    """Chen & Mellman (2017): desert = few CD8 anywhere; inflamed = intratumoral >= ratio x stromal; else excluded."""
    return np.select([tiles["total_density"] < desert, tiles["intra_density"] >= ratio * tiles["stroma_density"]],
                     ["desert", "inflamed"], "excluded")


def phenotype_by_programme(tiles: pd.DataFrame) -> pd.DataFrame:
    by = (pd.crosstab(tiles["programme"], tiles["phenotype"], normalize="index") * 100).reindex(columns=PHENOTYPES).fillna(0)
    by["tiles"] = tiles["programme"].value_counts()
    by.loc["All tumour tiles"] = list(tiles["phenotype"].value_counts(normalize=True).reindex(PHENOTYPES).fillna(0) * 100) + [len(tiles)]
    return by


def sensitivity(tiles: pd.DataFrame, deserts=(25, 50, 100), ratios=(0.3, 0.5, 0.7)) -> pd.DataFrame:
    """% of tiles per phenotype under alternative cutoffs."""
    return pd.DataFrame({f"desert<{d}, ratio>={r}": pd.Series(classify_tiles(tiles, d, r)).value_counts(normalize=True).reindex(PHENOTYPES).fillna(0) * 100
                         for d in deserts for r in ratios}).T


def distance_bands(adata, contact_um: float = 15) -> tuple[np.ndarray, list[str]]:
    bands = ["TLS", f"<= {contact_um} um", f"{contact_um}-50 um", "50-100 um", "> 100 um"]
    d = adata.obs["dist_to_tumour_um"].to_numpy()
    in_tls = (adata.obs["tls_id"] != "none").to_numpy()
    return np.select([in_tls, d <= contact_um, d <= 50, d <= 100], bands[:4], bands[4]), bands


def ratio_by_band(adata, numerator: str, denominator: str, contact_um: float = 15) -> pd.DataFrame:
    """Cell-count ratio of two cell types per distance band (with the counts, since small denominators are noisy)."""
    band, bands = distance_bands(adata, contact_um)
    ct = adata.obs["cell_type"].astype(str).to_numpy()
    rows = []
    for b in bands:
        n1, n2 = int(((ct == numerator) & (band == b)).sum()), int(((ct == denominator) & (band == b)).sum())
        rows.append({"location": b, "numerator": n1, "denominator": n2, "value": n1 / n2 if n2 else np.nan})
    return pd.DataFrame(rows)

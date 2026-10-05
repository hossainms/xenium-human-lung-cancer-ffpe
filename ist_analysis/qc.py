"""Quality control: per-cell metrics, QC rules, imaging-tile outliers (Steps 3-4). Platform-independent;
the platform adapter (ist_analysis.io) loads the data and names its control columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import mad_bounds


def add_qc_metrics(adata, control_columns=("control_probe_counts", "control_codeword_counts")) -> None:
    """Per-cell metrics in .obs. control_counts = sum of the platform's negative-control columns (see
    ist_analysis.io.<platform>.CONTROL_COLUMNS); control_frac uses gene counts as the denominator (as scanpy does)."""
    import scanpy as sc

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    obs = adata.obs
    obs["n_genes"] = obs["n_genes_by_counts"]
    obs["nucleus_ratio"] = obs["nucleus_area"] / obs["cell_area"]             # NaN if no nucleus
    obs["density"] = obs["transcript_counts"] / obs["cell_area"]                # transcripts per um^2
    obs["control_counts"] = obs[list(control_columns)].sum(axis=1)
    obs["control_frac"] = obs["control_counts"] / obs["total_counts"].replace(0, np.nan)


def qc_flags(obs: pd.DataFrame, min_counts: int, min_genes: int, area_nmads: float, max_control_frac: float):
    """One boolean column per QC rule (True = fails), and the data-driven cell-area bounds.

    Fixed floors for counts and genes (a MAD bound on the long low tail removes nothing); MAD bounds on
    the log scale for area (roughly log-normal).
    """
    area_lo, area_hi = mad_bounds(obs["cell_area"].to_numpy(), area_nmads)
    flags = pd.DataFrame({
        f"counts < {min_counts}": obs["transcript_counts"] < min_counts,
        f"genes < {min_genes}": obs["n_genes"] < min_genes,
        f"area < {area_lo:.1f} um^2": obs["cell_area"] < area_lo,
        f"area > {area_hi:.0f} um^2": obs["cell_area"] > area_hi,
        f"control frac > {max_control_frac:.0%}": obs["control_frac"].fillna(0) > max_control_frac,
    }, index=obs.index)
    return flags, (area_lo, area_hi)


def rule_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Cells flagged by each rule, and by that rule alone."""
    only = flags & (flags.sum(axis=1) == 1).to_numpy()[:, None]
    return pd.DataFrame({"cells flagged": flags.sum(), "% of cells": flags.mean() * 100, "flagged ONLY by this rule": only.sum()})


def assign_fov(xy: np.ndarray, fov: pd.DataFrame) -> np.ndarray:
    """Tile name under each cell centroid (tiles overlap by ~2 um; the first match wins)."""
    cell_fov = np.full(len(xy), "", dtype=object)
    for name, r in fov.iterrows():
        inside = ((xy[:, 0] >= r.x) & (xy[:, 0] < r.x + r.width) & (xy[:, 1] >= r.y) & (xy[:, 1] < r.y + r.height)
                  & (cell_fov == ""))
        cell_fov[inside] = name
    return cell_fov


def robust_z(s: pd.Series) -> pd.Series:
    return (s - s.median()) / (1.4826 * (s - s.median()).abs().median())


def fov_outliers(fov: pd.DataFrame, obs: pd.DataFrame, min_cells: int = 200, z: float = 3) -> pd.DataFrame:
    """Per-tile cells, median counts and % removed; tiles with |robust z| > z on counts or Q20 are outliers."""
    per = obs.groupby("fov", observed=True).agg(cells=("qc_pass", "size"), median_counts=("transcript_counts", "median"),
                                               pct_removed=("qc_pass", lambda s: 100 * (1 - s.mean())))
    fov = fov.join(per)
    big = fov["cells"] >= min_cells
    fov["outlier"] = False
    fov.loc[big, "outlier"] = (robust_z(fov.loc[big, "median_counts"]).abs() > z) | (robust_z(fov.loc[big, "frac_q20"]).abs() > z)
    return fov


def filter_cells(adata, keep: np.ndarray, params: dict):
    """The QC-passing cells, with the QC parameters stored in .uns['qc_params']."""
    out = adata[keep].copy()
    out.uns["qc_params"] = params
    out.obs = out.obs.drop(columns=[c for c in ("qc_pass", "qc_fail_reason") if c in out.obs])
    out.uns.pop("spatialdata_attrs", None)   # the SpatialData link is not meaningful in a standalone file
    return out

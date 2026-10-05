"""Shared paths, constants and helpers for the Xenium human lung cancer (FFPE) notebooks.

Every notebook imports this module instead of re-declaring paths and colours, so a change
(e.g. a new data location) is made in one place.

Data never lives in this repository: raw and processed files are under ~/data/xenium_lung/.
Paths printed in notebooks use display_path(), which shows them relative to the home folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- Locations ---------------------------------------------------------------------------
DATA_ROOT = Path("~/data/xenium_lung").expanduser()
SAMPLE = "Xenium_V1_humanLung_Cancer_FFPE"
SAMPLE_DIR = DATA_ROOT / SAMPLE
OUTS_DIR = SAMPLE_DIR / "outs"
PROCESSED_DIR = DATA_ROOT / "processed"

HE_PATH = SAMPLE_DIR / f"{SAMPLE}_he_image.ome.tif"
HE_ALIGNMENT_PATH = SAMPLE_DIR / f"{SAMPLE}_he_imagealignment.csv"

# Outputs of the core workflow (human_lung_cancer_workflow.ipynb), by step
STEP_FILES = {
    "qc": PROCESSED_DIR / f"{SAMPLE}_qc.h5ad",                  # Step 4: QC-filtered cells
    "clustered": PROCESSED_DIR / f"{SAMPLE}_clustered.h5ad",    # Step 5: clusters, UMAP
    "annotated": PROCESSED_DIR / f"{SAMPLE}_annotated.h5ad",    # Step 6: cell types (all cells)
    "spatial": PROCESSED_DIR / f"{SAMPLE}_spatial.h5ad",        # Step 7: high-confidence cells + graph
    "he_features": PROCESSED_DIR / f"{SAMPLE}_he_features.h5ad",  # Step 8: + H&E stain features
    "time": PROCESSED_DIR / f"{SAMPLE}_time.h5ad",              # Step 9: + TIME annotations
}

# ---- Constants -----------------------------------------------------------------------------
PIXEL_SIZE_UM = 0.2125  # Xenium morphology image: um per pixel

TUMOUR_TYPES = [
    "Tumour epithelial (MALL/TCIM)",
    "Tumour epithelial (CYP2B6/CFTR)",
    "Tumour epithelial (MYC/CAPN8)",
    "Proliferating tumour",
]

# Colourblind-checked categorical palette, fixed order (same as the core workflow, Step 6e)
LINEAGE_COLORS = {
    "Epithelial": "#2a78d6",
    "T / NK": "#eb6834",
    "B / plasma": "#1baf7a",
    "Myeloid": "#eda100",
    "Mast": "#e87ba4",
    "Stromal": "#008300",
    "Endothelial": "#4a3aa7",
}


# ---- Helpers ---------------------------------------------------------------------------------
def check_kernel(env_name: str = "spatial_env") -> None:
    """Stop early, with a clear message, if the notebook runs in the wrong Jupyter kernel."""
    if env_name not in sys.prefix:
        raise RuntimeError(f"Wrong kernel. Use Kernel > Change Kernel > 'Python ({env_name})'.")


def display_path(path: Path) -> str:
    """Path for printing: relative to the home folder (never exposes the full local path)."""
    path = Path(path)
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return path.name


def load_step(name: str, **kwargs):
    """Read a core-workflow output by step name, e.g. load_step('annotated')."""
    import scanpy as sc

    if name not in STEP_FILES:
        raise KeyError(f"Unknown step '{name}'. Choose from: {', '.join(STEP_FILES)}")
    path = STEP_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{display_path(path)} not found: run the core workflow up to that step first.")
    return sc.read_h5ad(path, **kwargs)


def output_dir(analysis: str) -> Path:
    """Folder for one side analysis's outputs, e.g. output_dir('resegmentation')."""
    path = PROCESSED_DIR / analysis
    path.mkdir(parents=True, exist_ok=True)
    return path

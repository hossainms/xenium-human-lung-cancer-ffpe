# Xenium human lung cancer (FFPE): spatial analysis of the tumour immune microenvironment

**Where do immune cells sit in a lung tumour, what are they doing there, and which findings survive a critical look at the data?** An end-to-end, single-cell-resolution analysis of the 10x Genomics **Xenium In Situ** public dataset *Human Lung Cancer FFPE* (377-gene panel, ~162,000 cells), built three ways: narrative **notebooks** (Python, with an independent **R / Bioconductor** replication), and a reproducible **Snakemake pipeline** that regenerates every result from the raw download.

## Key findings

**1. CD8 T cells are excluded from the tumour.** Only 12% of CD8 T cells sit within 15 µm of a tumour cell, against 34% for non-tumour cells in general; the tumour border is lined by macrophages, dendritic cells and fibroblasts instead. Per 400 µm tumour tile: 65% *excluded*, 26% *desert*, 9% *inflamed* (identical in Python and R).

![Immune phenotype per tumour tile](images/immune_phenotypes.png)

**2. Six tertiary lymphoid structure (TLS)-like aggregates.** B-cell cores with a T-cell rim, confirmed on the matched H&E. CD8 T cells lose the lymphoid-homing receptor CCR7 on the way from TLS to tumour (36% -> 11%), and CCL19 -> CCR7 cell contacts are enriched 2.4-fold in a spatial permutation test.

![TLS-like aggregates](images/tls_lineage_map.jpg)
![TLS1 on the H&E](images/he_tls_window.jpg)

**3. The same macrophage type changes state with location.** FCGR1A+ macrophages switch from a TREM2 / CTSK / LPL programme in stroma to MARCO / MCEMP1 / CXCL2 next to tumour (paired pseudobulk DESeq2 on spatial domains from BANKSY and CellCharter).

![Spatial domains](images/spatial_domains.jpg)

**4. The critical look.** Segmentation spillover is the dominant artefact in imaging spatial data: 19% of T cells carried tumour transcripts. Re-segmentation with Proseg reduces this to 1.5%, and every finding above was re-tested on the cleaner cells (CD8 exclusion, all six TLS and the CCR7 gradient hold). 42 of 60 'differentially expressed' macrophage genes were neighbouring-cell spillover and are filtered out. Cell types were validated against the Lung Cancer Atlas (892k cells; 92% lineage agreement, against an 87% ceiling for this panel).

## The analysis, three ways

| Form | Where | Use it to |
|---|---|---|
| **Notebooks** (Python) | `human_lung_cancer_workflow.ipynb` (Steps 1-10), `notebooks/02-04` | read the analysis with figures and interpretation |
| **Notebook** (R / Bioconductor) | `human_lung_cancer_workflow_by_R.ipynb` | an independent replication in R (Seurat, SpatialExperiment, imcRtools, Banksy, CellChat, spatstat), cross-checked against Python step by step |
| **Snakemake pipeline** | `workflow/`, `config/` | re-run everything reproducibly from the raw download |

## What the analysis does

| Step | Content | Main tools |
|---|---|---|
| 1-2 | Download and inspect the 10x output bundle | `requests`, `pyarrow`, `tifffile` |
| 3-4 | Load, per-cell and per-tile quality control | `spatialdata-io`, `scanpy` |
| 5 | Normalisation, PCA, Leiden clustering, UMAP, markers | `scanpy` |
| 6 | Cell-type annotation (marker scores, lineage subclustering) | `scanpy` |
| 7 | Spatial graph, neighbourhood enrichment, niches, tumour distances, TLS-like aggregates | `squidpy`, `scikit-learn` |
| 8 | H&E alignment check and per-cell stain features | `tifffile`, `scikit-image` |
| 9 | Checkpoints, CD8 T-cell states by location, ligand-receptor contacts, immune phenotypes | permutation tests |
| 10 | Moran's I, co-occurrence, Ripley's L, centrality, `ligrec` | `squidpy`, OmniPath |
| 02 | Cell re-segmentation and robustness of the findings | Proseg |
| 03 | Validation of cell types against the Lung Cancer Atlas (LuCA) | logistic regression (CellTypist model) |
| 04 | Spatial domains and location-dependent macrophage states | BANKSY, CellCharter, PyDESeq2 |

## The pipeline

![Pipeline rule graph](workflow/rulegraph.png)

Each box is a Snakemake rule (one script); arrows are file dependencies. The core chain runs download -> QC -> clustering -> annotation -> spatial analysis, with H&E, TIME and spatial statistics after it; Proseg re-segmentation branches off the raw data and feeds the robustness test, the atlas mapping and the domain analysis. Regenerate the graph with `snakemake --rulegraph | dot -Tpng > workflow/rulegraph.png`.

## Running the pipeline

**1. Environments** (conda; Apple Silicon and Linux tested with conda-forge packages):

```bash
conda env create -f workflow/envs/spatial_env.yml      # analysis environment (Python 3.12, scverse stack)
conda env create -f workflow/envs/snakemake_env.yml    # orchestrator (Snakemake 9)
```

Notebook 02 also needs [Proseg](https://github.com/dcjones/proseg) (`cargo install proseg`), unless an earlier run is reused (`proseg.reuse` in the config).

**2. Configure** `config/config.yaml`: data and results locations (default `~/data/xenium_lung/`, outside the repository), and every analysis parameter. Cell-type label decisions are in `config/annotation.yaml`.

**3. Run** from the repository root:

```bash
conda activate snakemake_env
snakemake -n                     # dry run: list the jobs
snakemake --cores 8 core         # core workflow, Steps 1-10
snakemake --cores 8              # everything, including notebooks 02-04
snakemake --cores 8 annotate     # stop after one step
```

Each rule runs a script in `workflow/scripts/` inside `spatial_env` (via `conda run`). The scripts also run on their own, e.g. `python workflow/scripts/qc.py --marker <flag> --output qc.h5ad`.

**Outputs** (under `results:` in the config): one `.h5ad` per step, `figures/`, `tables/`, `logs/`, and `xenium_explorer/` cell-group CSVs (cell types, lineages, niches, domains) that load into Xenium Explorer.

**Report:** `python workflow/make_report.py report.html` (in `snakemake_env`) builds a self-contained HTML page with the workflow graph, runtimes, and the key figures and tables of every step, each with a caption (`workflow/report/`) and the code and parameters that produced it. It wraps `snakemake --report` and replaces local paths with `~`, so the page can be shared.

**Inspecting the workflow:** `snakemake -n` (what would run, and why), `snakemake --summary` (every output, the rule that made it, whether it is up to date).

**Approximate runtimes** on an 8-core laptop: core workflow ~15 min; Proseg ~30 min (~13 GB RAM); atlas download 12.9 GB, then ~20 min for the first panel-gene cache; spatial domains ~12 min.

## Repository layout

```
human_lung_cancer_workflow.ipynb        core analysis, Python (Steps 1-10)
human_lung_cancer_workflow_by_R.ipynb   core analysis, R / Bioconductor
notebooks/                               side analyses (02 re-segmentation, 03 reference mapping, 04 domains)
xenium_utils.py                          shared paths, constants, helpers
workflow/Snakefile                       pipeline rules
workflow/scripts/                        one script per step
workflow/envs/                           conda environment files
workflow/report/                         report captions
workflow/make_report.py                  shareable HTML report (local paths removed)
workflow/rulegraph.png                   pipeline diagram
config/                                  pipeline parameters and annotation decisions
spatial_transcriptomic_analysis_*_packages.md   environment documentation (Python, R)
```

## Data

10x Genomics public dataset, licensed CC BY 4.0. Raw data and results are never stored in the repository.

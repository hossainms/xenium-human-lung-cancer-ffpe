# R Environment for Spatial Transcriptomic Analysis

This document describes how the R environment `spatialR_env` was built for the R implementation of the Xenium human lung cancer (FFPE) workflow, which packages it contains, and why each one is needed. It is the R counterpart of the Python environment used in the main workflow notebook.

**Environment summary:** R 4.5.3 · Bioconductor 3.22 · conda (conda-forge + bioconda) · Apple Silicon (osx-arm64) · Jupyter kernel `R (spatialR_env)`

---

## 1. Why a dedicated environment

Spatial transcriptomics in R depends on a large, tightly versioned stack: Seurat, Bioconductor's spatial classes, compiled C++ libraries, and geospatial system libraries (GDAL, GEOS, PROJ). Installing it into a shared R installation risks breaking other projects, and other projects risk breaking it. A dedicated conda environment gives:

- **Isolation:** its own R, its own package library, and its own compilers, independent of any system R.
- **Reproducibility:** package versions are recorded and can be exported and rebuilt.
- **Pre-built binaries:** most packages install from conda without compiling, which is faster and avoids system-library conflicts.

## 2. How the environment was built

### Step 1: Create the environment with R and the Jupyter kernel package

```bash
conda create -n spatialR_env -c conda-forge --override-channels r-base=4.5 r-irkernel
```

A quick linear-algebra test (`solve()` on a small matrix) confirmed that R's numerical libraries load correctly before anything else was installed.

### Step 2: Install pre-built packages from conda-forge and bioconda

```bash
conda install -n spatialR_env --solver=libmamba -c conda-forge -c bioconda --override-channels \
  r-seurat r-seuratobject r-arrow r-hdf5r r-matrix r-data.table r-ggplot2 r-patchwork r-sf \
  r-spatstat r-spatstat.explore r-igraph r-dbscan r-rann r-future r-biocmanager r-remotes \
  r-leidenbase r-dplyr r-tidyr r-viridis r-scales \
  bioconductor-spatialexperiment bioconductor-singlecellexperiment bioconductor-scater \
  bioconductor-scran bioconductor-bluster bioconductor-celldex compilers
```

`--solver=libmamba` matters: with conda's older "classic" solver, this same request ran for more than 12 minutes without finishing; libmamba solved and installed it in about 90 seconds.

### Step 3: Install supporting dependencies as binaries

Several packages below are not available pre-built for Apple Silicon and must be compiled. Their dependencies were first installed from conda so that as little as possible is compiled from source, together with the system libraries that compilation needs (`fftw`, `libtiff`, `pkg-config`).

### Step 4: Compile the remaining packages from source

```bash
# the environment's own compilers must be on PATH, otherwise R cannot find them
PATH="<conda>/envs/spatialR_env/bin:$PATH" Rscript -e '
  BiocManager::install(c("SingleR", "Banksy", "imcRtools", "SpatialFeatureExperiment", "Voyager"))
  remotes::install_github("immunogenomics/presto")
  remotes::install_github("jinworks/CellChat")'
```

### Step 5: Register the Jupyter kernel

`IRkernel::installspec()` could not be used because it calls the `jupyter` command, which is not on conda R's path. The kernel was registered by writing its `kernel.json` directly. Its environment block sets `PATH` to include the environment's compilers, so packages can also be installed from inside a notebook, and pins `R_LIBS_USER` to the environment's library so no packages leak in from elsewhere.

## 3. Packages and their roles

**Source** column: `conda` = pre-built binary from conda-forge / bioconda; `Bioc (src)` = compiled from Bioconductor; `GitHub` = compiled from the developer's repository.

### 3.1 Core data structures and input

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **Seurat** | 5.5.1 | conda | The main single-cell and spatial analysis framework in R. `LoadXenium()` reads a Xenium output bundle (counts, cell centroids, boundaries, transcripts) into one object, and Seurat provides normalisation, clustering, markers and spatial plotting. |
| **SeuratObject** | 5.4.0 | conda | The data classes underlying Seurat (assays, layers, spatial fields of view). |
| **SpatialExperiment** | 1.20.0 | conda | Bioconductor's standard container for spatial omics: a count matrix plus spatial coordinates. Required by Bioconductor spatial tools such as imcRtools and Banksy. |
| **SingleCellExperiment** | 1.32.0 | conda | Bioconductor's core single-cell container, on which SpatialExperiment and most Bioconductor methods are built. |
| **SpatialFeatureExperiment** | 1.12.1 | Bioc (src) | Extends SpatialExperiment with geometries (cell polygons, tissue outlines) through `sf`, and reads Xenium output directly. The data structure that Voyager operates on. |
| **arrow** | 25.0.0 | conda | Reads Parquet files. Xenium stores transcripts, cells and cell boundaries as Parquet. |
| **hdf5r** | 1.3.16 | conda | Reads HDF5 files, including the Xenium cell-by-gene matrix (`cell_feature_matrix.h5`). |
| **Matrix** | 1.7.5 | conda | Sparse matrices. Most of a cell-by-gene count matrix is zeros, so sparse storage is essential at this scale. |

### 3.2 Single-cell analysis: QC, normalisation, clustering, markers

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **scater** | 1.38.1 | conda | Per-cell quality-control metrics and standard single-cell plots in the Bioconductor framework. |
| **scran** | 1.38.1 | conda | Normalisation and graph-based clustering helpers for SingleCellExperiment objects. |
| **bluster** | 1.20.0 | conda | A unified interface to clustering algorithms and cluster-quality diagnostics. |
| **leidenbase** | 0.1.36 | conda | Fast Leiden community detection, the clustering algorithm used in the Python workflow, so results can be compared like for like. |
| **igraph** | 2.3.4 | conda | Graph algorithms behind neighbour graphs and community detection. |
| **presto** | 1.1.0 | GitHub | Very fast Wilcoxon tests for marker genes. Seurat uses it automatically when installed, cutting marker detection on ~150,000 cells from minutes to seconds. |

### 3.3 Cell-type annotation

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **SingleR** | 2.12.0 | Bioc (src) | Reference-based annotation: labels each cell by correlating its expression with annotated reference profiles. An objective check on marker-based manual labels. |
| **celldex** | 1.20.0 | conda | Curated reference datasets (for example human immune cell atlases) used by SingleR. |

### 3.4 Spatial statistics and tissue domains

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **spatstat** / **spatstat.explore** | 3.6.2 / 3.8.2 | conda | The reference toolkit for spatial point-pattern statistics. Ripley's K / L functions can use the real tissue outline as the observation window, which avoids treating holes and edges as "empty". |
| **imcRtools** | 1.16.0 | Bioc (src) | Builds spatial neighbour graphs and runs permutation tests of cell-type neighbourhood enrichment and avoidance (the histoCAT method). |
| **Banksy** | 1.6.0 | Bioc (src) | Spatial domain detection: clusters cells using their own expression together with their neighbours', so the resulting groups are tissue regions rather than cell types. |
| **Voyager** | 1.12.0 | Bioc (src) | Exploratory spatial statistics on SpatialFeatureExperiment: global and local Moran's I, Geary's C, and spatially variable genes. |
| **sf** | 1.1.2 | conda | Simple Features: the standard representation of spatial geometries in R (cell polygons, tissue regions, distance calculations). |
| **dbscan** | 1.2.6 | conda | Density-based clustering, used to detect dense immune aggregates such as tertiary lymphoid structures. |
| **RANN** | 2.6.2 | conda | Fast nearest-neighbour search, for example the distance from each immune cell to the nearest tumour cell. |

### 3.5 Cell-cell communication

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **CellChat** | 2.2.0.9001 | GitHub | Infers ligand-receptor signalling between cell types from a curated interaction database. Version 2 adds a spatially aware mode that restricts interactions to cells within signalling distance. |

### 3.6 Visualisation and data handling

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **ggplot2** | 4.0.3 | conda | The grammar-of-graphics plotting system used for all figures. |
| **patchwork** | 1.3.2 | conda | Combines several ggplot2 plots into one multi-panel figure. |
| **viridis** / **scales** | 0.6.5 / 1.4.0 | conda | Perceptually uniform, colour-blind-friendly colour scales and axis formatting. |
| **dplyr** / **tidyr** | 1.2.1 / 1.3.2 | conda | Readable data manipulation for per-cell tables and summaries. |
| **data.table** | 1.18.6.1 | conda | Fast handling of large tables, such as millions of transcripts. |

### 3.7 Infrastructure

| Package | Version | Source | Why it is used |
|---|---|---|---|
| **IRkernel** | 1.3.2 | conda | Runs R inside Jupyter notebooks. |
| **BiocManager** | 1.30.27 | conda | Installs Bioconductor packages with versions matched to the R release (Bioconductor 3.22 for R 4.5). |
| **remotes** | 2.5.0 | conda | Installs packages distributed only on GitHub (presto, CellChat). |
| **future** | 1.76.0 | conda | Parallel processing backend used by Seurat and CellChat for heavy steps. |

### 3.8 Supporting dependencies

These were installed explicitly so that the packages above could be built. They are not called directly in the analysis.

| Package | Version | Needed by | Role |
|---|---|---|---|
| ComplexHeatmap, circlize | 2.26.1, 0.4.18 | CellChat | Annotated heatmaps and circular (chord) plots of signalling networks |
| NMF | 0.28 | CellChat | Non-negative matrix factorisation to find coordinated signalling patterns |
| ggalluvial, ggnetwork, sna, network | 0.12.6, 0.5.14, 2.8, 1.20.0 | CellChat | Flow and network diagrams of cell-cell communication |
| ggpubr, svglite | 1.0.0, 2.2.2 | CellChat | Publication-style plot layouts and SVG export |
| terra, spdep | 1.9.34, 1.4.2 | SpatialFeatureExperiment, Voyager | Raster images and spatial weights for spatial autocorrelation |
| DropletUtils, tiff, fftwtools | 1.30.0, 0.1.12, 0.9.11 | SpatialFeatureExperiment | Reading 10x matrices and microscopy images |
| SparseArray | 1.10.10 | Bioconductor core | Sparse array support in Bioconductor containers |
| RcppHungarian, mclust, pROC | 0.3, 6.1.3, 1.19.1 | Banksy | Cluster label matching and model-based clustering |
| RCurl | 1.98.1.19 | Bioconductor | Downloads of reference data and annotations |

System libraries installed through conda (not R packages): `compilers` (C, C++ and Fortran toolchain), `fftw`, `libtiff`, `pkg-config`.

## 4. Considered but not installed: deconvolution tools

Readers familiar with spatial transcriptomics often expect **cell-type deconvolution** tools. They were deliberately not installed, because this dataset does not need them.

**Why deconvolution does not apply here.** Spot-based technologies such as 10x Visium measure RNA in spots (55 µm on standard Visium) that each contain roughly 1-10 cells, so every measurement is a mixture. Deconvolution methods use a single-cell RNA-seq reference to estimate which cell types, and in what proportions, make up each spot. Xenium is an imaging-based, **single-cell resolution** platform: every transcript has an exact position and is assigned to an individual segmented cell. There is no mixture to unmix, so cell types are identified directly by clustering and annotation instead.

| Package | What it does | When it would be used |
|---|---|---|
| **spacexr (RCTD)** | Robust cell-type decomposition of spots using a single-cell reference; a Visium HD-specific mode also exists | Spot-based data, e.g. the Visium HD section 10x released from the same lung cancer tissue |
| **SPOTlight** | Seeded non-negative matrix factorisation to estimate spot composition | Spot-based data with a well-annotated single-cell reference |
| **CARD** | Deconvolution that borrows information from neighbouring spots (spatial correlation) | Spot-based data where neighbouring spots share composition |

For single-cell resolution data like Xenium, the equivalent step is **reference-based annotation of individual cells**, which this environment covers with SingleR and celldex (section 3.3).

## 5. Practical notes

- **Use the libmamba solver.** Large R and Bioconductor installs can stall for a long time with conda's classic solver.
- **Not everything is pre-built for Apple Silicon.** Banksy, imcRtools, Voyager, SpatialFeatureExperiment and SingleR had missing or outdated conda builds on osx-arm64 and were compiled from Bioconductor instead. presto and CellChat are GitHub-only.
- **Put the environment's compilers on PATH when compiling.** Calling the environment's `Rscript` without activating the environment leaves its compilers invisible, and every compiled package fails with `clang: command not found`.
- **Register the Jupyter kernel by hand if needed.** `IRkernel::installspec()` requires the `jupyter` command on R's path; writing `kernel.json` directly avoids that.

## 6. Reproducing the environment

To record the exact environment for others:

```bash
conda env export -n spatialR_env --no-builds > spatialR_env.yml
Rscript -e 'writeLines(capture.output(sessionInfo()), "spatialR_sessionInfo.txt")'
```

The packages compiled from source (sections 3.3-3.5 marked `Bioc (src)` or `GitHub`) are not captured by `conda env export` and must be reinstalled with Step 4.

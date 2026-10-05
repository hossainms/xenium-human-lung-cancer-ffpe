# Python Environment for Spatial Transcriptomic Analysis

This document describes how the Python environment `spatial_env` was built for the Xenium human lung cancer (FFPE) workflow, which packages it contains, and why each one is needed. The main workflow notebook (Steps 1-10) and the side-analysis notebooks run in this environment. Its R counterpart is described in `spatial_transcriptomic_analysis_R_packages.md`.

**Environment summary:** Python 3.12 · conda environment + pip · scverse ecosystem (scanpy, squidpy, spatialdata) · Apple Silicon (osx-arm64) · Jupyter kernel `Python (spatial_env)`

---

## 1. Why a dedicated environment

The Python spatial stack (spatialdata, scanpy, squidpy) pins specific versions of numpy, dask, xarray, zarr and anndata. Installing it into a general-purpose environment can silently upgrade or downgrade packages that other projects rely on. A first attempt to add it to an existing environment would have downgraded numpy and pulled in about 55 packages, so a clean environment was created instead. This gives:

- **Isolation:** the spatial stack cannot break, or be broken by, other projects.
- **Reproducibility:** versions are fixed and can be exported.
- **A clean dependency resolution:** pip resolves the whole scverse stack together in one step.

## 2. How the environment was built

### Step 1: Create the environment

```bash
conda create -n spatial_env python=3.12 pip
```

### Step 2: Isolate it from user-level Python packages

```bash
conda env config vars set -n spatial_env PYTHONNOUSERSITE=1
```

Packages installed with `pip install --user` live in a user folder that Python searches *before* the environment's own packages, so they can silently override them. Setting `PYTHONNOUSERSITE=1` removes that folder from the search path. This also exposed two dependencies (`inflect`, `more-itertools`) that had only been satisfied by user-level copies; they were then installed properly into the environment.

### Step 3: Install the spatial stack

```bash
pip install spatialdata spatialdata-io spatialdata-plot scanpy squidpy tifffile h5py pyarrow ipykernel
pip install inflect more-itertools ipywidgets      # missing dependencies + notebook progress bars
pip install igraph leidenalg                       # Leiden clustering
pip check                                          # verify: "No broken requirements found"
```

### Step 4: Register the Jupyter kernel

```bash
python -m ipykernel install --user --name spatial_env --display-name "Python (spatial_env)"
```

The kernel definition also sets `PYTHONNOUSERSITE=1`, so the isolation from Step 2 applies inside notebooks too, not only in an activated terminal.

## 3. Packages and their roles

The **Used in** column refers to the steps of the main workflow notebook.

### 3.1 Spatial data structures and input

| Package | Version | Used in | Why it is used |
|---|---|---|---|
| **spatialdata** | 0.8.0 | 3, 7 | The scverse standard container for spatial omics. Holds images, segmentation masks, transcripts, cell shapes and the count table in one object, with coordinate transformations between them. |
| **spatialdata-io** | 0.7.1 | 3, 8 | Readers for commercial platforms. `xenium()` loads a full Xenium output bundle; `xenium_aligned_image()` attaches the H&E image with its alignment matrix. |
| **spatialdata-plot** | 0.4.2 | 3 | Plotting for SpatialData objects (images, shapes and points rendered in a shared coordinate system). |
| **anndata** | 0.13.4 | 4-10 | The annotated cell-by-gene matrix format used by scanpy and squidpy; all intermediate results are saved as `.h5ad` files. |
| **pyarrow** | 25.0.1 | 2, 3, 4, 9 | Reads Parquet files (transcripts, cells, boundaries) column by column, without loading whole tables into memory. |
| **h5py** | 3.16.0 | 2, 4 | Reads HDF5 files such as the Xenium count matrix (`cell_feature_matrix.h5`). |
| **tifffile** | 2026.9.20 | 2, 8 | Reads OME-TIFF microscopy images, including individual levels of the multi-resolution pyramid, so a whole-slide H&E can be viewed without loading 2 GB at full resolution. |
| **zarr**, **ome-zarr** | 3.4.0, 0.19.2 | 3 | Chunked array storage used by SpatialData for large images and tables. |
| **dask**, **xarray** | 2026.8.0, 2026.9.0 | 3 | Lazy, chunked arrays: images and transcripts are read only when needed, so a 12-million-transcript dataset loads in seconds. |
| **multiscale-spatial-image** | 2.0.3 | 3 | Represents image pyramids (several resolutions of one image) inside SpatialData. |

### 3.2 Single-cell analysis: QC, normalisation, clustering, markers

| Package | Version | Used in | Why it is used |
|---|---|---|---|
| **scanpy** | 1.12.4 | 4-6 | The core single-cell toolkit: QC metrics, normalisation, PCA, neighbour graphs, clustering, UMAP, marker-gene tests and gene-set scoring. |
| **igraph**, **leidenalg** | 1.0.0, 0.12.0 | 5, 6 | Leiden community detection, the clustering algorithm that groups cells in the neighbour graph. |
| **umap-learn**, **pynndescent** | 0.5.12, 0.6.0 | 5 | UMAP embeddings and the fast approximate nearest-neighbour search behind them. |
| **numba** | 0.68.0 | 5, 10 | Just-in-time compilation that makes UMAP and squidpy's spatial statistics fast. |

### 3.3 Spatial statistics and cell-cell communication

| Package | Version | Used in | Why it is used |
|---|---|---|---|
| **squidpy** | 1.8.3 | 7, 10 | Spatial analysis for scverse: spatial neighbour graphs, neighbourhood enrichment, co-occurrence, Ripley's statistics, centrality scores, Moran's I and ligand-receptor analysis (`ligrec`). |
| **omnipath** | 1.0.12 | 10 | Client for the OmniPath database of curated ligand-receptor interactions used by `squidpy.gr.ligrec`. |
| **networkx** | 3.7 | 10 | Graph metrics behind squidpy's centrality scores. |
| **geopandas**, **shapely** | 1.2.0, 2.1.2 | 3 | Geometry handling for cell boundary polygons. |

### 3.4 General scientific computing

| Package | Version | Used in | Why it is used |
|---|---|---|---|
| **numpy** | 2.4.6 | all | Numerical arrays. |
| **pandas** | 3.0.6 | all | Tables of per-cell metadata, QC metrics and results. |
| **scipy** | 1.18.1 | 4-10 | Sparse matrices for counts, KD-trees for fast distance queries (distance to tumour, niche neighbourhoods), statistical tests. |
| **scikit-learn** | 1.9.1 | 5, 7 | Agreement between clusterings (ARI, NMI), k-means for cellular niches, DBSCAN for tertiary lymphoid structure detection. |
| **statsmodels** | 0.15.0 | 9 | Multiple-testing correction (Benjamini-Hochberg FDR). |
| **scikit-image** | 0.26.0 | 8 | Image warping for H&E alignment and colour deconvolution (separating hematoxylin and eosin stains). |

### 3.5 Visualisation and notebook support

| Package | Version | Used in | Why it is used |
|---|---|---|---|
| **matplotlib** | 3.11.2 | all | All figures in the workflow. |
| **seaborn** | 0.13.2 | 5 | Statistical plot styling used internally by scanpy plotting functions. |
| **ipykernel** | 7.4.0 | all | Runs this environment as a Jupyter kernel. |
| **ipywidgets** | 8.1.9 | 1 | Live progress bars (`tqdm`) inside notebooks. |
| **tqdm** | 4.70.1 | 1 | Progress bars for the dataset download. |
| **requests** | 2.34.2 | 1 | Resumable download of the dataset from 10x Genomics. |

## 4. External tool: Proseg (cell re-segmentation)

| Tool | Version | Used in | Why it is used |
|---|---|---|---|
| **Proseg** | 3.2.0 | `notebooks/02_resegmentation_proseg` | Re-draws cell boundaries from the transcripts themselves, using a probabilistic model. Reduces transcripts wrongly assigned to neighbouring cells (segmentation spillover). |
| **Rust toolchain** (rustc, cargo) | 1.99.0 | install only | Proseg is written in Rust and is built from source with `cargo install proseg`. |

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
~/.cargo/bin/cargo install proseg --locked
```

`--no-modify-path` installs Rust without editing shell startup files; the tools are called by their full path.

## 5. Considered but not installed: deconvolution tools

Readers familiar with spatial transcriptomics often expect **cell-type deconvolution** tools. They were deliberately not installed, because this dataset does not need them.

**Why deconvolution does not apply here.** Spot-based technologies such as 10x Visium measure RNA in spots (55 µm on standard Visium) that each contain roughly 1-10 cells, so every measurement is a mixture. Deconvolution methods use a single-cell RNA-seq reference to estimate which cell types, and in what proportions, make up each spot. Xenium is an imaging-based, **single-cell resolution** platform: every transcript has an exact position and is assigned to an individual segmented cell. There is no mixture to unmix, so cell types are identified directly by clustering and annotation instead.

| Tool | What it does | When it would be used |
|---|---|---|
| **cell2location** | Bayesian deconvolution of spots into cell-type abundances, trained on a single-cell reference (built on PyTorch / scvi-tools) | Spot-based data, e.g. the Visium HD section 10x released from the same lung cancer tissue, for cross-platform validation of the Xenium cell types |
| **Tangram** | Maps single-cell RNA-seq profiles onto spatial positions | Imputing genes that are absent from the 377-gene panel (e.g. TIGIT, CXCL13), with caution |
| **DestVI / Stereoscope** (scvi-tools) | Deconvolution with continuous within-cell-type variation | Spot-based data where cell states, not only types, vary across tissue |

These tools would run in a **separate environment**: they depend on PyTorch, which is large and pins its own versions of numpy and other core packages, and model training is best done on a GPU (e.g. a computing cluster) rather than a laptop CPU.

## 6. Practical notes

- **Check what an install would change first.** `pip install --dry-run` showed that adding the stack to an existing environment would downgrade numpy, which is how the decision to create `spatial_env` was made.
- **Isolate from user-level packages.** Without `PYTHONNOUSERSITE=1`, packages from `pip install --user` sit ahead of the environment on the search path and can override it without warning.
- **Two small API gotchas found while building the workflow:**
  - `squidpy.gr.ligrec` returns *sparse* pandas DataFrames; convert them with `.sparse.to_dense()` before indexing, otherwise lookups silently return nothing.
  - Keys inside `.uns` cannot contain `/` when writing `.h5ad` files (for example a dictionary key such as `"T / NK"`).
- **Scale matters on ~140,000 cells:** Moran's I with permutations ran for more than 10 minutes, while the analytical p-value (`n_perms=None`) takes 11 seconds; Ripley's statistics are computed on subsamples.

## 7. Reproducing the environment

```bash
conda env export -n spatial_env --no-builds > spatial_env.yml
conda run -n spatial_env pip freeze > spatial_env_requirements.txt
```

Recreate it with `conda env create -f spatial_env.yml`, then re-run Steps 2 and 4 above (the user-site setting and the kernel registration are not part of the exported file).

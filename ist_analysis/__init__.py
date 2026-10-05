"""ist_analysis: analysis functions for imaging-based spatial transcriptomics (iST).

Used here on a 10x Xenium human lung cancer (FFPE) dataset; platform-specific input lives in ist_analysis.io
(io.xenium), and every other module works on a standard AnnData (counts, centroids in um, cell metadata).

The same functions are used by the notebooks (narrative + figures) and by the Snakemake pipeline
(workflow/scripts/), so every result has one implementation.

Modules
-------
io         platform adapters (io.xenium): loading the bundle, imaging tiles, gene detection, vendor clusters
utils      this project's paths and constants (tumour types, lineage colours), small helpers
qc         per-cell metrics, QC rules, imaging-tile outliers
cluster    normalisation, PCA, Leiden clustering, UMAP, marker genes, benchmark against vendor clusters
annotate   marker-set scores, cluster labels with marker guards, lineage subclustering
spatial    neighbour graph, neighbourhood enrichment, niches, distance to tumour, TLS detection
he         H&E alignment check, histology windows, per-cell stain features
tme        tumour immune microenvironment: positivity rates, CD8 states, contact test, phenotypes
stats      spatial statistics helpers around squidpy (co-occurrence, balanced Ripley, ligrec)
proseg     segmentation comparison, Proseg export, robustness of findings
reference  reference mapping to the Lung Cancer Atlas (LuCA)
domains    spatial domains (BANKSY, CellCharter), spillover filter, paired pseudobulk DE
"""

__version__ = "1.1.0"

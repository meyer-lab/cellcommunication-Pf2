# CCC-RISE - Cell-Cell Communication with RISE

CCC-RISE (Cell-Cell Communication with Reduction and Insight in Single-cell Exploration) is an unsupervised, tensor-based computational method designed for the integrative analysis of cell-cell communication in single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions. Built upon the CP (CANDECOMP/PARAFAC) tensor decomposition framework combined with PARAFAC2, CCC-RISE preserves the four-dimensional structure of multi-condition cell-cell communication data—conditions × sender cells × receiver cells × ligand-receptor pairs—instead of flattening it into conventional matrices.

CCC-RISE decomposes variation into distinct, interpretable patterns associated with experimental conditions, sender cells, receiver cells, and specific ligand-receptor interactions, providing a more nuanced and biologically meaningful analysis of intercellular communication dynamics. This approach enables the identification of coordinated signaling programs that change across conditions, offering insights into how cellular communication networks respond to different experimental perturbations.

CCC-RISE does not require prior cell-type labels or clustering, reducing bias and enabling discovery of novel communication patterns, while maintaining high resolution to identify condition-specific signaling subpopulations missed by pseudobulk or clustering-based approaches. Each resulting component is directly linked to specific conditions, sender cells, receiver cells, and ligand-receptor pairs, making the results biologically tractable.

- **Read the documentation** at [CCC-RISE Documentation](https://meyer-lab.github.io/cellcommunication-Pf2/).
- CCC-RISE uses the [AnnData](https://anndata.readthedocs.io/) format for handling single-cell data matrices.

## Installation

To add CCC-RISE to your Python environment, you can install it directly from GitHub:

```bash
pip install git+https://github.com/meyer-lab/cellcommunication-Pf2.git@main
```

Or add the following line to your `requirements.txt`:

```
git+https://github.com/meyer-lab/cellcommunication-Pf2.git@main
```

## Quick Start

CCC-RISE works with preprocessed AnnData objects containing single-cell RNA-seq data:

```python
from cellcommunicationpf2.ccc_rise import run_ccc_rise_workflow
import pandas as pd

# Load your ligand-receptor pairs database
lr_pairs = pd.read_csv("lr_pairs.csv")  # Should contain ligand-receptor interaction information

# Perform CCC-RISE tensor decomposition
adata, r2x = run_ccc_rise_workflow(
    adata=adata,
    rise_rank=20,
    lr_pairs=lr_pairs,
    condition_column="sample",
    doEmbedding=True,
    random_state=42
)

# Results are stored in the AnnData object:
# - adata.uns["weights"]: Component weights
# - adata.uns["A"]: Condition factors
# - adata.uns["B"]: Sender cell factors
# - adata.uns["C"]: Receiver cell factors
# - adata.uns["D"]: Ligand-receptor pair factors
# - adata.uns["lr_pairs"]: Filtered ligand-receptor pairs
# - adata.uns["r2x"]: Variance explained by the decomposition
# - adata.obsm["projections"]: Cell projections
# - adata.obsm["sc_B"]: Sender cell embeddings
# - adata.obsm["rc_C"]: Receiver cell embeddings
# - adata.obsm["PaCMAP"]: PaCMAP embeddings (if doEmbedding=True)
```

See the [tutorial](https://meyer-lab.github.io/cellcommunication-Pf2/tutorial.html) for a complete workflow including preprocessing, rank selection, visualization, and interpretation.

## Key Features

- **4D Tensor decomposition**: Preserves the full structure of cell-cell communication data (conditions × sender cells × receiver cells × ligand-receptor pairs)
- **Unsupervised analysis**: No prior cell-type labels or clustering required
- **Integrated communication analysis**: Simultaneously analyzes sender cells, receiver cells, and specific ligand-receptor interactions
- **High resolution**: Identifies communication patterns and condition-specific signaling subpopulations
- **Interpretable results**: Components directly linked to conditions, sender/receiver cells, and ligand-receptor pairs
- **Condition-aware**: Captures how communication networks change across experimental conditions
- **Integrated workflow**: Built-in preprocessing, visualization, and interpretation tools

## Citation

If you use CCC-RISE in your work, please cite the CCC-RISE publication as follows:

**Integrative tensor-based analysis of cell-cell communication in single-cell RNA sequencing across experimental conditions**

Andrew Ramirez, [...], Aaron Meyer

*In preparation*, 2025.
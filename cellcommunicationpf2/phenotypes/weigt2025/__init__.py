"""
Weigt et al. 2025 - BAL Genomic Classifier for Acute Cellular Rejection

Source: https://data.mendeley.com/datasets/3x49zhc9r6/1
DOI: 10.17632/3x49zhc9r6.1

Paper: "Development and Validation of a Bronchoalveolar Lavage Genomic Classifier
        for Acute Cellular Rejection" (Weigt et al., 2025)

Original files from Mendeley (4 files):
  1. normalized_counts_CTOT_806.txt
  2. normalized_counts_UCLA_219.txt
  3. Phenotype_data_CTOT_Pre-CLAD_BAL-cp_samples.xlsx (2 sheets)
  4. Phenotype_data_UCLA_BAL-cp_samples.xlsx

This directory contains phenotype/metadata:
- CTOT-20 cohort: 806 BAL-cp samples from 181 lung transplant recipients
- UCLA validation cohort: 219 BAL-cp samples

Expression data (normalized counts) is stored separately on Aretha due to size.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent


def load_ctot_phenotype() -> pd.DataFrame:
    """Load CTOT-20 phenotype data (806 samples, 54 columns)."""
    return pd.read_parquet(DATA_DIR / "ctot_phenotype.parquet")


def load_ctot_phenotype_definitions() -> pd.DataFrame:
    """Load CTOT-20 phenotype group definitions."""
    return pd.read_parquet(DATA_DIR / "ctot_phenotype_definitions.parquet")


def load_ucla_phenotype() -> pd.DataFrame:
    """Load UCLA validation phenotype data (219 samples, 34 columns)."""
    return pd.read_parquet(DATA_DIR / "ucla_phenotype.parquet")


__all__ = [
    "load_ctot_phenotype",
    "load_ctot_phenotype_definitions",
    "load_ucla_phenotype",
]

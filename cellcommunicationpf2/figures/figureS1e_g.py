"""
Figure S1e_g: Cell type proportions across conditions in BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    
    df = (X.obs
        .groupby(["celltype", "sample"])
        .size()
        .reset_index(name="count")
    )

    # Add condition information back by merging with sample-condition mapping
    sample_condition_map = X.obs[["sample", "condition"]].drop_duplicates()
    df = df.merge(sample_condition_map, on="sample", how="left")

    df["celltype"] = pd.Categorical(df["celltype"], categories=np.unique(df["celltype"]), ordered=True)
    df["sample"] = pd.Categorical(df["sample"], categories=np.unique(df["sample"]), ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=np.unique(df["condition"]), ordered=True)
    df = df.sort_values(["condition", "sample", "celltype"])
    df["proportion"] = df.groupby("sample")["count"].transform(lambda x: x / x.sum())
    print(df)

    sns.boxplot(data=df, x="condition", y="proportion", hue="celltype", ax=ax[0], palette="Set3")
    ax[0].set_yscale("log")
    ax[0].set_title("Cell Type Proportions by Condition")
    ax[0].set_ylim(.0001, 1.5)

    condition_severity_map = {"Control": 1, "Moderate COVID-19": 2, "Severe COVID-19": 3}
    df["severity"] = df["condition"].map(condition_severity_map)

    correlation_results = []
    for celltype in df["celltype"].unique():
        subset = df[df["celltype"] == celltype]
        if len(subset["severity"].unique()) > 1:  # Ensure there's variability in severity
            corr = pearsonr(subset["proportion"], subset["severity"]).correlation
            pval = pearsonr(subset["proportion"], subset["severity"]).pvalue
            correlation_results.append((celltype, corr, pval))
    corr_df = pd.DataFrame(correlation_results, columns=["celltype", "correlation", "pvalue"]).dropna()
    print(corr_df)
    sns.scatterplot(data=corr_df, x="correlation", y="pvalue", ax=ax[1], hue="celltype", palette="Set3")
    ax[1].set_title("Correlation of Cell Type Proportion with Severity")
    ax[1].set_xlabel("Pearson Correlation")
    ax[1].set_ylabel("P-value")
    ax[1].axhline(0.05, color='red', linestyle='--')
    
    
    sample_totals = df.groupby(["sample"])["count"].sum().reset_index()
    sample_totals.columns = ["sample", "total_cells"]
    sample_totals = sample_totals.merge(sample_condition_map, on="sample", how="left")
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]
    sns.scatterplot(data=sample_totals, x="sample", y="total_cells", hue="condition",  ax=ax[3], palette=pal)
    ax[3].set_title("Total Cells per Sample")
    ax[3].tick_params(axis='x', rotation=45)
    ax[3].set_ylabel("Total Cell Count")

    return f

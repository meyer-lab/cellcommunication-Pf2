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
from scipy.stats import spearmanr


def makeFigure():
    ax, f = getSetup((4, 4), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    sample_id = "dsco_id"
    condition_id = "ALADstatus"
    celltype_id = "broad_cell_type"

    df = X.obs.groupby([celltype_id, sample_id]).size().reset_index(name="count")

    # Add condition information back by merging with sample-condition mapping
    sample_condition_map = X.obs[[sample_id, condition_id]].drop_duplicates()
    df = df.merge(sample_condition_map, on=sample_id, how="left")

    df[celltype_id] = pd.Categorical(
        df[celltype_id], categories=np.unique(df[celltype_id]), ordered=True
    )
    df[sample_id] = pd.Categorical(
        df[sample_id], categories=np.unique(df[sample_id]), ordered=True
    )
    df[condition_id] = pd.Categorical(
        df[condition_id], categories=np.unique(df[condition_id]), ordered=True
    )
    df = df.sort_values([condition_id, sample_id, celltype_id])
    df["proportion"] = df.groupby(sample_id)["count"].transform(lambda x: x / x.sum())
    print(df)

    sns.boxplot(
        data=df, x=condition_id, y="proportion", hue=celltype_id, ax=ax[0], palette="Set3"
    )
    ax[0].set_yscale("log")
    ax[0].set_title("Cell Type Proportions by Condition")
    ax[0].set_ylim(0.0001, 1.5)

    # condition_severity_map = {
    #     "Control": 1,
    #     "Moderate COVID-19": 2,
    #     "Severe COVID-19": 3,
    # }
    # df["severity"] = df[condition_id].map(condition_severity_map)

    # correlation_results = []
    # for celltype in df[celltype_id].unique():
    #     subset = df[df[celltype_id] == celltype]
    #     if (
    #         len(subset["severity"].unique()) > 1
    #     ):  # Ensure there's variability in severity
    #         corr = spearmanr(subset["proportion"], subset["severity"])[0]
    #         pval = spearmanr(subset["proportion"], subset["severity"])[1]
    #         correlation_results.append((celltype, corr, pval))
    # corr_df = pd.DataFrame(
    #     correlation_results, columns=["celltype", "correlation", "pvalue"]
    # ).dropna()
    # print(corr_df)
    # sns.scatterplot(
    #     data=corr_df,
    #     x="correlation",
    #     y="pvalue",
    #     ax=ax[1],
    #     hue="celltype",
    #     palette="Set3",
    # )
    # ax[1].set_title("Correlation of Cell Type Proportion with Severity")
    # ax[1].set_xlabel("Spearman Correlation")
    # ax[1].set_ylabel("P-value")
    # ax[1].axhline(0.05, color="red", linestyle="--")
    # ax[1].set_yscale("log")
    # ax[1].set_xlim(-1, 1)

    # sample_totals = df.groupby(["sample"])["count"].sum().reset_index()
    # sample_totals.columns = ["sample", "total_cells"]
    # sample_totals = sample_totals.merge(sample_condition_map, on="sample", how="left")
    # pal = sns.color_palette("Set2")
    # pal = [pal[0], pal[1], pal[2]]
    # sns.scatterplot(
    #     data=sample_totals,
    #     x="sample",
    #     y="total_cells",
    #     hue="condition",
    #     ax=ax[3],
    #     palette=pal,
    # )
    # ax[3].set_title("Total Cells per Sample")
    # ax[3].tick_params(axis="x", rotation=45)
    # ax[3].set_ylabel("Total Cell Count")

    return f

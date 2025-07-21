import anndata
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def plot_condition_factors(
    data: anndata.AnnData,
    ax: Axes,
    cond: str = "Condition",
    cond_group_labels: pd.Series | None = None,
    color_key=None,
    group_cond=False,
):
    """Plots Pf2 condition factors"""

    yt = pd.Series(np.unique(data.obs[cond]))
    X = np.array(data.uns["Pf2_A"])

    XX = X
    X -= np.median(XX, axis=0)
    X /= np.std(XX, axis=0)
    X /= np.max(np.abs(X))

    ind = reorder_table(X)
    X = X[ind]
    yt = yt.iloc[ind]

    if cond_group_labels is not None:
        # Align cond_group_labels with the reordered yt
        cond_group_labels = cond_group_labels.loc[yt].reset_index(drop=True)
        if group_cond is True:
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]

        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            colors = sns.color_palette(
                "Set2", n_colors=pd.Series(cond_group_labels).nunique()
            ).as_hex()
        else:
            colors = color_key
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.unique(cond_group_labels)):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))

        group_values = list(cond_group_labels)
        row_colors = [lut.get(val, "#cccccc") for val in group_values]
        ROW_RECTANGLE_X_OFFSET = -0.02
        ROW_RECTANGLE_WIDTH = 0.02
        # Add colored rectangles for each row
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(ROW_RECTANGLE_X_OFFSET, iii),
                    width=ROW_RECTANGLE_WIDTH,
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )

        # Create legend outside the plot area
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Position legend to the right of the plot
            frameon=False,  # Remove frame for a cleaner look
            fontsize=10,
            title="Condition",
            title_fontsize=12,
        )

    xticks = np.arange(1, X.shape[1] + 1)
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component")


def plot_eigenstate_factors(data: anndata.AnnData, ax: Axes, factor_type: str):
    """Plots Pf2 eigenstate factors"""
    cp_rank = data.uns["Pf2_B"].shape[1]
    xticks = np.arange(1, cp_rank + 1)
    X = data.uns["Pf2_B"] if factor_type == "Pf2_B" else data.uns["Pf2_C"]
    X = X / np.max(np.abs(np.array(X)))
    rise_rank = data.uns["Pf2_B"].shape[0]
    yt = np.arange(1, rise_rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


def plot_lr_factors(data: anndata.AnnData, ax: Axes, trim=True, weight=0.08):
    """Plots Pf2 lr factors"""
    # Read the LR factor and pair information from .uns
    X = np.array(data.uns["Pf2_D"])
    lr_pairs = data.uns["Pf2_lr_pairs"]
    rank = X.shape[1]

    # Create labels from the ligand and receptor columns
    yt = [f"{row['ligand']}-{row['receptor']}" for _, row in lr_pairs.iterrows()]

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > weight
        X = X[kept_idxs]
        yt = [y for i, y in enumerate(yt) if kept_idxs[i]]

    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = [yt[ii] for ii in ind]
    xticks = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


def plot_gene_factors_partial(
    cmp: int, dataIn: anndata.AnnData, ax: Axes, geneAmount: int = 5, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=dataIn.varm["Pf2_D"][:, cmp - 1], index=dataIn.var_names, columns=[cmpName]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=ax
        )
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)


def reorder_table(projs: np.ndarray):
    """Reorder a table's rows using hierarchical clustering"""

    # Clean non-finite values
    clean_projs = np.nan_to_num(projs, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure no zero vectors (which cause issues with cosine distance)
    zero_rows = np.all(clean_projs == 0, axis=1)
    if np.any(zero_rows):
        # Add a small epsilon to zero rows to avoid cosine distance issues
        clean_projs[zero_rows, :] = 1e-10

    Z = sch.linkage(
        clean_projs, method="complete", metric="cosine", optimal_ordering=True
    )
    return sch.leaves_list(Z)

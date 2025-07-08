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
    pd.set_option("display.max_rows", None)
    yt = pd.Series(np.unique(data.obs[cond]))
    X = np.array(data.uns["Pf2_A"])

    X = X / np.max(np.abs(X))

    ind = reorder_table(X)
    X = X[ind]
    yt = yt.iloc[ind]

    if cond_group_labels is not None:
        # Try/except to handle potential alignment issues robustly
        try:
            # Align cond_group_labels with the reordered yt
            cond_group_labels = cond_group_labels.loc[yt].reset_index(drop=True)
            if group_cond is True:
                ind = cond_group_labels.argsort()
                cond_group_labels = cond_group_labels.iloc[ind]
                X = X[ind]
                yt = yt.iloc[ind]
        except Exception as e:
            print(f"Warning: Could not align condition labels: {e}")
            # Fall back to original ordering if alignment fails
            pass
            
        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            # Use a colorblind-friendly palette with distinct colors
            colors = sns.color_palette(
                "Set2", n_colors=pd.Series(cond_group_labels).nunique()
            )
        else:
            colors = color_key
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.unique(cond_group_labels)):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))
        
        # Convert labels to a simple list and map directly, avoiding MultiIndex issues
        group_values = list(cond_group_labels)  # Extract values as a simple list
        row_colors = [lut.get(val, "#cccccc") for val in group_values]  # Map with fallback color
        
        # Add colored rectangles for each row - MAKE THEM THINNER
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(-0.02, iii),  # Changed from -0.05 to -0.02
                    width=0.02,       # Changed from 0.05 to 0.02
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )
        
        # Create a more visible, well-positioned legend
        ax.legend(
            handles=legend_elements,
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.15),  # Moved from 1.35 to 1.15 to bring it closer to the plot
            ncol=min(len(legend_elements), 3),  # Limit columns for readability
            frameon=True,  # Add frame for better visibility
            fontsize=12,   # Increase font size
            title="Patient Condition",  
            title_fontsize=14  # Larger title font
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
    rank = data.uns["Pf2_B"].shape[1]
    xticks = np.arange(1, rank + 1)
    X = data.uns["Pf2_B"] if factor_type == "Pf2_B" else data.uns["Pf2_C"]
    X = X / np.max(np.abs(np.array(X)))
    yt = np.arange(1, rank + 1)

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
    yt = np.array(yt)

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > weight
        X = X[kept_idxs]
        yt = yt[kept_idxs]

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

    Z = sch.linkage(clean_projs, method="complete", metric="cosine", optimal_ordering=True)
    return sch.leaves_list(Z)

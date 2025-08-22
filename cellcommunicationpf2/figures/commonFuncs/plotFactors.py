import anndata
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

cmap = sns.diverging_palette(240, 10, as_cmap=True)


from typing import Union, Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import seaborn as sns


def plot_condition_factors(
    data: Union[anndata.AnnData, Dict[str, np.ndarray], np.ndarray],
    ax: Axes,
    cond: Optional[str] = "Condition",
    cond_group_labels: Optional[pd.Series] = None,
    condition_labels: Optional[Union[pd.Series, List[str], np.ndarray]] = None,
    color_key=None,
    group_cond=False,
    vmin=0
):
    """Plots condition factors from Pf2 or similar decomposition"""
    
    # Extract data based on input type
    if isinstance(data, anndata.AnnData):
        X = np.array(data.uns["Pf2_A"])
        if condition_labels is None:
            idxs = np.argsort(data.obs["condition_unique_idxs"].unique())
            condition_labels = pd.Series(data.obs[cond].unique())[idxs]
    elif isinstance(data, dict):
        X = np.array(data["A"])
        if condition_labels is None and "condition_labels" in data:
            condition_labels = data["condition_labels"]
    else:
        X = np.array(data)
    
    # Handle condition labels
    if condition_labels is None:
        condition_labels = [f"Condition {i+1}" for i in range(X.shape[0])]
    elif isinstance(condition_labels, (list, np.ndarray)):
        condition_labels = pd.Series(condition_labels)
    
    # Convert to pandas Series for consistent handling
    if not isinstance(condition_labels, pd.Series):
        condition_labels = pd.Series(condition_labels)
    
    # # Normalize data
    # XX = X.copy()
    # X = X - np.median(XX, axis=0)
    # X = X / np.std(XX, axis=0)
    # X = X / np.max(np.abs(X))
    
    # Reorder
    ind = reorder_table(X)
    X = X[ind]
    
    # Reorder condition labels safely
    condition_labels = condition_labels.reset_index(drop=True)
    condition_labels = condition_labels.iloc[ind].reset_index(drop=True)
    
    if cond_group_labels is not None:
        # Convert cond_group_labels to Series if it's not already
        if not isinstance(cond_group_labels, pd.Series):
            cond_group_labels = pd.Series(cond_group_labels)
        
        # Reset index for both to ensure alignment
        condition_labels = condition_labels.reset_index(drop=True)
        cond_group_labels = cond_group_labels.reset_index(drop=True)
        
        # Align cond_group_labels with the reordered condition_labels
        # Create a mapping from condition label to group label
        label_to_group = dict(zip(condition_labels, cond_group_labels))
        cond_group_labels = pd.Series([label_to_group.get(label, "Unknown") 
                                     for label in condition_labels])
            
        if group_cond:
            # Get sorted indices based on group labels
            sort_indices = cond_group_labels.argsort().values
            cond_group_labels = cond_group_labels.iloc[sort_indices]
            X = X[sort_indices]
            condition_labels = condition_labels.iloc[sort_indices].reset_index(drop=True)
            cond_group_labels = cond_group_labels.reset_index(drop=True)
        
        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            colors = sns.color_palette(
                "Set2", n_colors=cond_group_labels.nunique()
            ).as_hex()
        else:
            colors = color_key
        
        lut = {}
        legend_elements = []
        for index, group in enumerate(cond_group_labels.unique()):
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
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=10,
            title="Condition",
            title_fontsize=12,
        )
    
    xticks = np.arange(1, X.shape[1] + 1)
    
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=condition_labels.tolist(),
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component")
    
    

def plot_eigenstate_factors(
    data: Union[anndata.AnnData, Dict[str, np.ndarray], np.ndarray], 
    ax: Axes, 
    factor_type: str = "B" or None,
    labels: Optional[Union[List[str], np.ndarray]] = None,
    vmin: float = 0,
):
    """Plots eigenstate factors from Pf2 or similar decomposition"""
    
    # Extract data based on input type
    if isinstance(data, anndata.AnnData):
        X = data.uns[f"Pf2_{factor_type}"]
    else:
        X = data
    
    # Handle eigenstate labels
    if labels is None:
        labels = np.arange(1, X.shape[0] + 1)

    X = X / np.max(np.abs(np.array(X)))
    xticks = np.arange(1, X.shape[1] + 1)
    
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=labels,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
    )
    ax.set(xlabel="Component")


def plot_lr_factors(
    data: Union[anndata.AnnData, np.ndarray], 
    ax: Axes, 
    lr_pairs: Optional[pd.DataFrame] = None,
    trim: bool = True, 
    weight: float = 0.08,
    vmin: float = 0
):
    """Plots ligand-receptor factors from Pf2 or similar decomposition"""
    
    # Extract data based on input type
    if isinstance(data, anndata.AnnData):
        X = np.array(data.uns["Pf2_D"])
        if lr_pairs is None:
            lr_pairs = data.uns["Pf2_lr_pairs"]
    else: 
        X = np.array(data)

    # Create labels from the ligand and receptor columns
    yt = [f"{row['ligand']}-{row['receptor']}" for _, row in lr_pairs.iterrows()]
    
    if trim:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > weight
        X = X[kept_idxs]
        yt = [y for i, y in enumerate(yt) if kept_idxs[i]]
    
    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = [yt[ii] for ii in ind]
    xticks = np.arange(1, X.shape[1] + 1)
    
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=vmin,
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


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)

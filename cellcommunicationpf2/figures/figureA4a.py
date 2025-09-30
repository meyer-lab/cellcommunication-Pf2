"""
Figure S1c_d: FMS across CPD ranks (only) for COVID-19
"""

import seaborn as sns
import numpy as np
from ..import_data import (
    add_cond_idxs,
    import_alad,
    import_ligand_receptor_pairs
)
from .common import getSetup, subplotLabel
from ..tensor import run_fms_r2x_analysis, calculate_interaction_tensor
from ..utils import correct_conditions
import anndata
import numpy as np
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from ..ccc_rise import ccc_rise, standardize_cc_pf2
from ..import_data import add_cond_idxs
import pandas as pd
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2, anndata_to_list
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from ..ccc_rise import calc_communication_score
from ..tensor import resample_tensor, rise_store_r2x, calculate_fms_cpd, calculate_fms_rise, calculate_r2x
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)


    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    print(X)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)
    group_col = "ALADstatus"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    sample_to_group = sample_to_group.loc[np.unique(X.obs[condition_column], return_index=True)[0]]
    print(sample_to_group)
    # Make non-control samples be called alad for simplicity
    sample_to_group = sample_to_group.apply(lambda x: "alad" if x != "control" else "control")
    print(sample_to_group)
    # Convert categories to 0 and 1
    sample_to_group = sample_to_group.astype("category").cat.codes
    print(sample_to_group)
    scoring = ["roc_auc", "accuracy"]

    # Initialize score lists outside the rank loop
    scores_aucroc = []
    scores_accuracy = []

    # Run FMS and R2X analysis
    rank_list = list(np.append([1], range(5, 66, 5)))
    # rank_list = list(range(1, 4, 2))
    for i in rank_list:
        print(f"Rank {i}")
        X, _ = rise_store_r2x(X, rank=i, n_iter_max=10000, tolerance=1e-9)
        X.uns["A"] = X.uns["Pf2_A"]
        X.uns["A"] = correct_conditions(X)
        
        # Calculate scores for this rank
        for i in scoring:
            lr_fit = logistic_regression(i).fit(X.uns["A"], sample_to_group)
            score = float(np.max(np.mean(lr_fit.scores_[1], axis=0)))  # Convert to plain float
            if i == "roc_auc":
                scores_aucroc.append(score) 
            elif i == "accuracy":
                scores_accuracy.append(score)

    print("AUC scores:", scores_aucroc)
    print("Accuracy scores:", scores_accuracy)
    
    ax[0].plot(rank_list, scores_aucroc)
    ax[0].set_xlabel("RISE Rank")    
    ax[0].set_ylabel("10-Fold CV: roc_auc")
    
    ax[1].plot(rank_list, scores_accuracy)
    ax[1].set_xlabel("RISE Rank")    
    ax[1].set_ylabel("10-Fold CV: accuracy")

            
            

    return f




def logistic_regression(scoring):
    """Standardizing LogReg for all functions"""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    return LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        cv=cv,
        scoring=scoring,
    )


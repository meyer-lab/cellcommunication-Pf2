
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from .tensor import rise_store_r2x
from .utils import correct_conditions
from tensorly.decomposition import parafac

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


def rise_ranks_logreg(
    X, 
    rank_list, 
    sample_to_group, 
    scoring=["roc_auc", "accuracy"], 
    n_iter_max=10000, 
    tolerance=1e-9,
    random_state=0
):
    """
    Evaluate logistic regression performance across different RISE ranks.
    Returns:
    --------
    scores_aucroc : list
        AUC scores for each rank
    scores_accuracy : list
        Accuracy scores for each rank
    """
    # Initialize score lists
    scores_aucroc = []
    scores_accuracy = []
    
    # Make a copy to avoid modifying original data
    X_work = X.copy()
    
    for rank in rank_list:
        print(f"Rank {rank}")
        
        # Perform RISE decomposition for this rank
        X_work, _ = rise_store_r2x(X_work, rank=rank, n_iter_max=n_iter_max, 
                                   tolerance=tolerance, random_state=random_state)
        
        # Extract A factor and correct conditions
        X_work.uns["A"] = X_work.uns["Pf2_A"]
        X_work.uns["A"] = correct_conditions(X_work)
        
        # Calculate scores for this rank
        for metric in scoring:
            lr_fit = logistic_regression(metric).fit(X_work.uns["A"], sample_to_group)
            score = float(np.max(np.mean(lr_fit.scores_[1], axis=0)))  # Convert to plain float
            
            if metric == "roc_auc":
                scores_aucroc.append(score) 
            elif metric == "accuracy":
                scores_accuracy.append(score)
    
    return scores_aucroc, scores_accuracy


def cpd_ranks_logreg(
    X,
    interaction_tensor,
    rank_list,
    sample_to_group,
    scoring=["roc_auc", "accuracy"],
    n_iter_max=10000,
    random_state=0
):
    """
    Evaluate logistic regression performance across different CPD ranks with adaptive initialization.
    
    Returns:
    --------
    scores_aucroc : list
        AUC scores for each rank
    scores_accuracy : list
        Accuracy scores for each rank
    """

    
    # Initialize score lists
    scores_aucroc = []
    scores_accuracy = []

    for j in rank_list:
        print(f"Rank {j}")
        
        # Adaptive initialization based on rank
        if j <= 15:
            svd_init = "svd"
        else:
            svd_init = "random"
            
        _, cp_factors = parafac(
            tensor=interaction_tensor,
            rank=j,
            n_iter_max=n_iter_max,
            init=svd_init,
            normalize_factors=True,
            random_state=random_state
        )
        
        # Store A factor and correct conditions
        X.uns["A"] = cp_factors[0]
        cp_factors[0] = correct_conditions(X)
        
        # Calculate scores for this rank
        for i in scoring:
            lr_fit = logistic_regression(i).fit(cp_factors[0], sample_to_group)
            score = float(np.max(np.mean(lr_fit.scores_[1], axis=0)))  # Convert to plain float
            if i == "roc_auc":
                scores_aucroc.append(score) 
            elif i == "accuracy":
                scores_accuracy.append(score)

    return scores_aucroc, scores_accuracy

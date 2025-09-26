"""
Figure A2b: FMS across CPD ranks (only) for bal alad
"""

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from ..import_data import (
    add_cond_idxs,
    import_alad,
    import_ligand_receptor_pairs
)
from .common import getSetup, subplotLabel
from ..utils import run_fms_r2x_analysis, calculate_interaction_tensor

def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    print(X)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X_filtered = add_cond_idxs(X, condition_column)

    # Calculate interaction tensor
    interaction_tensor = calculate_interaction_tensor(X_filtered, lr_pairs, rise_rank=25)
    print("Interaction tensor shape:", interaction_tensor.shape)
    
    rank_list = list(range(1, 3, 1))
    for i, rank in enumerate(rank_list):
        
        cp_weights, cp_factors = parafac(
            tensor=interaction_tensor,
            rank=rank,
            n_iter_max=1000,
            init="svd",
                normalize_factors=True,
            )
        cp_factors[0] = correct_conditions(cp_factors[0])
        
        
        lr = logistic_regression(scoring)
        
    
    
    group_col = "ALADstatus"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    


def logistic_regression(scoring):
    """Standardizing LogReg for all functions"""
    return LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        scoring=scoring,
    )


def run_lr_alad(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    patient_id_col: str = "dsco_id",
    alad_status_col: str = "ALADstatus",
    proba: bool = False,
    scoring: str = 'accuracy'
) -> tuple[pd.Series, LogisticRegressionCV]:
    """
    Predicts ALAD status (yes/no) via logistic regression cross-validation.

    Args:
        data (pd.DataFrame): feature data to predict (e.g., PARAFAC factors)
        labels (pd.DataFrame): dataframe containing patient IDs and ALAD status
        patient_id_col (str): column name for patient IDs
        alad_status_col (str): column name for ALAD status (yes/no)
        proba (bool, default:False): return probability of prediction
        scoring (str, default:'accuracy'): scoring metric for LogisticRegressionCV

    Returns:
        predicted (pd.Series): predicted ALAD status for patients; if proba is
            True, returns probabilities of positive ALAD status
        lr (LogisticRegressionCV): fitted logistic regression model
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Convert ALAD status to binary (assuming 'yes'=1, 'no'=0)
    binary_alad = labels[alad_status_col].map({'yes': 1, 'no': 0})
    if binary_alad.isna().any():
        # If mapping doesn't work, try other common encodings
        binary_alad = labels[alad_status_col].map({True: 1, False: 0})
        if binary_alad.isna().any():
            # Try direct boolean conversion
            binary_alad = labels[alad_status_col].astype(bool).astype(int)

    # Create StratifiedGroupKFold for patient-level cross-validation
    SGKF = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr = logistic_regression(scoring)

    probabilities = pd.Series(0, dtype=float, index=data.index)
    
    # Perform cross-validation splits based on patients
    for train_index, test_index in SGKF.split(
        data,
        binary_alad,
        labels[patient_id_col]
    ):
        train_group_data = data.iloc[train_index, :]
        train_labels = binary_alad.iloc[train_index]
        test_group_data = data.iloc[test_index]
        
        lr.fit(train_group_data, train_labels)
        
        if proba:
            probabilities.iloc[test_index] = lr.predict_proba(test_group_data)[:, 1]
        else:
            probabilities.iloc[test_index] = lr.predict(test_group_data)

    # Fit final model on all data
    lr.fit(data, binary_alad)
    
    # Store coefficients with feature names for interpretability
    if hasattr(lr, 'coef_') and lr.coef_ is not None:
        lr.coef_ = pd.Series(lr.coef_.squeeze(), index=data.columns)
    
    return probabilities, lr


def evaluate_alad_prediction(X_filtered, parafac_factors, ax):
    """
    Evaluate ALAD status prediction using PARAFAC factors.
    
    Args:
        X_filtered: AnnData object with sample information
        parafac_factors: PARAFAC decomposition factors
        ax: matplotlib axis for plotting
    """
    # Prepare sample-level data
    condition_column = "dsco_id"
    alad_status_col = "ALADstatus"
    
    # Get unique samples and their ALAD status
    sample_df = X_filtered.obs[[condition_column, alad_status_col]].drop_duplicates()
    
    # Use A factor (sample factor) as features
    A_factor = parafac_factors[0]  # Assuming first factor is sample factor
    feature_data = pd.DataFrame(A_factor, index=sample_df[condition_column])
    feature_data.columns = [f'Component_{i+1}' for i in range(A_factor.shape[1])]
    
    # Run logistic regression
    predictions, lr_model = run_lr_alad(
        data=feature_data,
        labels=sample_df,
        patient_id_col=condition_column,
        alad_status_col=alad_status_col,
        proba=True,
        scoring='roc_auc'
    )
    
    # Calculate performance metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    binary_alad = sample_df[alad_status_col].map({'yes': 1, 'no': 0})
    if binary_alad.isna().any():
        binary_alad = sample_df[alad_status_col].astype(bool).astype(int)
    
    # Get binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(binary_alad, binary_predictions)
    auc = roc_auc_score(binary_alad, predictions)
    
    print(f"ALAD Status Prediction Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Number of samples: {len(sample_df)}")
    print(f"Number of features: {A_factor.shape[1]}")
    
    # Plot feature importance (coefficients)
    coef_df = pd.DataFrame({
        'Component': lr_model.coef_.index,
        'Coefficient': lr_model.coef_.values,
        'Abs_Coefficient': np.abs(lr_model.coef_.values)
    }).sort_values('Abs_Coefficient', ascending=True)
    
    # Plot top 10 most important features
    top_features = coef_df.tail(10)
    
    sns.barplot(data=top_features, y='Component', x='Coefficient', ax=ax)
    ax.set_title(f'ALAD Status Prediction\nTop Features (AUC: {auc:.3f})')
    ax.set_xlabel('Logistic Regression Coefficient')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    return predictions, lr_model, accuracy, auc
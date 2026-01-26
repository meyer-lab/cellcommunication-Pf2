"""
PARAFAC2 Rank Selection via Cell Holdout Cross-Validation.

Main API:
    run_rank_selection(adata, ranks, condition_key) -> pd.DataFrame

Usage:
    from cellcommunicationpf2.rank_selection import run_rank_selection
    
    df_results = run_rank_selection(
        adata=your_anndata,
        ranks=list(range(2, 15)),
        condition_key="sample",
        n_folds=5
    )
    
    # Results are in a DataFrame with columns: rank, ot_score_mean, ot_score_std, r2x_mean
    print(df_results)
"""

from .rank_selection import run_rank_selection

__all__ = ["run_rank_selection"]

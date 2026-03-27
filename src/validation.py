"""
Leave-One-Family-Out cross-validation for the Retroviral Wall Challenge.

The 57 RT predictions are pooled across all 7 folds before computing CLS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .metrics import compute_cls


def get_family_splits(family_splits_path):
    """
    Load family splits and return a dict mapping family name to list of RT names.
    """
    df = pd.read_csv(family_splits_path)
    splits = {}
    for _, row in df.iterrows():
        family = row["family"]
        rt_names = row["rt_names"].split("|")
        splits[family] = rt_names
    return splits


def lofo_cv(train_df, family_splits, model_fn, feature_cols, target_col="pe_efficiency_pct"):
    """
    Run Leave-One-Family-Out CV and return pooled OOF predictions + CLS.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training data with features, target, rt_name, rt_family.
    family_splits : dict
        Output of get_family_splits().
    model_fn : callable(X_train, y_train, X_test) -> np.ndarray
        Function that fits a model and returns predictions on X_test.
    feature_cols : list of str
        Feature column names.
    target_col : str
        Target column name.

    Returns
    -------
    dict with keys:
        - oof_predictions: pd.DataFrame with rt_name, predicted_score, active, pe_efficiency_pct, rt_family
        - metrics: dict from compute_cls (cls, pr_auc, w_spearman)
        - per_fold: list of dicts with fold-level info
    """
    oof_rows = []
    per_fold = []

    for family, rt_names in family_splits.items():
        # Split
        test_mask = train_df["rt_name"].isin(rt_names)
        train_mask = ~test_mask

        X_train = train_df.loc[train_mask, feature_cols].values
        y_train = train_df.loc[train_mask, target_col].values
        X_test = train_df.loc[test_mask, feature_cols].values

        # Predict
        preds = model_fn(X_train, y_train, X_test)

        # Collect OOF predictions
        fold_df = train_df.loc[test_mask, ["rt_name", "active", "pe_efficiency_pct", "rt_family"]].copy()
        fold_df["predicted_score"] = preds
        oof_rows.append(fold_df)

        per_fold.append({
            "family": family,
            "n_test": test_mask.sum(),
            "n_active_test": train_df.loc[test_mask, "active"].sum(),
            "n_train": train_mask.sum(),
            "n_active_train": train_df.loc[train_mask, "active"].sum(),
        })

    # Pool all OOF predictions, reorder to match rt_sequences.csv
    # (the official evaluator merges on rt_name using rt_sequences.csv order,
    # and argsort-based ranking is sensitive to row order for ties)
    oof_df = pd.concat(oof_rows, ignore_index=True)
    gt_order = pd.read_csv(
        Path(__file__).parent.parent / "data" / "raw" / "rt_sequences.csv",
        usecols=["rt_name"],
    )["rt_name"].tolist()
    oof_df = oof_df.set_index("rt_name").loc[gt_order].reset_index()

    # Compute global CLS
    metrics = compute_cls(
        y_true=oof_df["active"].values,
        y_score=oof_df["predicted_score"].values,
        pe_efficiency=oof_df["pe_efficiency_pct"].values,
    )

    return {
        "oof_predictions": oof_df,
        "metrics": metrics,
        "per_fold": per_fold,
    }


def print_lofo_summary(result):
    """Pretty-print LOFO CV results."""
    metrics = result["metrics"]
    print("=" * 50)
    print("LOFO CV — Pooled OOF Results")
    print("=" * 50)
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  Weighted Spearman: {metrics['w_spearman']:.4f}")
    print(f"  CLS:               {metrics['cls']:.4f}")
    print()
    print("Per-fold breakdown:")
    print(f"  {'Family':<25s} {'n':>3s} {'Active':>6s} {'Train':>5s} {'Tr.Act':>6s}")
    print("  " + "-" * 48)
    for fold in result["per_fold"]:
        print(
            f"  {fold['family']:<25s} "
            f"{fold['n_test']:3d} "
            f"{fold['n_active_test']:6.0f} "
            f"{fold['n_train']:5d} "
            f"{fold['n_active_train']:6.0f}"
        )

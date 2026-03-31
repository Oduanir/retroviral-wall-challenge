"""
Bootstrap confidence intervals for CLS, PR-AUC, and Weighted Spearman.

Uses stratified resampling to maintain the active/inactive ratio.
"""

import numpy as np
from .metrics import compute_cls


def bootstrap_cls(oof_df, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for CLS and its components.

    Parameters
    ----------
    oof_df : pd.DataFrame
        Must contain columns: active, predicted_score, pe_efficiency_pct.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: cls, pr_auc, w_spearman — each a dict with
        point, mean, std, ci_low, ci_high.
    """
    rng = np.random.RandomState(seed)

    y_true = oof_df["active"].values
    y_score = oof_df["predicted_score"].values
    pe_eff = oof_df["pe_efficiency_pct"].values
    n = len(y_true)

    # Point estimates
    point = compute_cls(y_true, y_score, pe_eff)

    # Stratified indices
    active_idx = np.where(y_true == 1)[0]
    inactive_idx = np.where(y_true == 0)[0]
    n_active = len(active_idx)
    n_inactive = len(inactive_idx)

    boot_cls = np.empty(n_bootstrap)
    boot_prauc = np.empty(n_bootstrap)
    boot_wsp = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Stratified resample: same number of actives and inactives
        idx_a = rng.choice(active_idx, size=n_active, replace=True)
        idx_i = rng.choice(inactive_idx, size=n_inactive, replace=True)
        idx = np.concatenate([idx_a, idx_i])

        result = compute_cls(y_true[idx], y_score[idx], pe_eff[idx])
        boot_cls[b] = result["cls"]
        boot_prauc[b] = result["pr_auc"]
        boot_wsp[b] = result["w_spearman"]

    alpha = 1.0 - ci
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    def summarize(point_val, boot_arr):
        return {
            "point": point_val,
            "mean": float(np.mean(boot_arr)),
            "std": float(np.std(boot_arr)),
            "ci_low": float(np.percentile(boot_arr, lo)),
            "ci_high": float(np.percentile(boot_arr, hi)),
        }

    return {
        "cls": summarize(point["cls"], boot_cls),
        "pr_auc": summarize(point["pr_auc"], boot_prauc),
        "w_spearman": summarize(point["w_spearman"], boot_wsp),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


def print_bootstrap_results(results):
    """Pretty-print bootstrap CI results."""
    ci_pct = int(results["ci_level"] * 100)
    print(f"Bootstrap Confidence Intervals ({results['n_bootstrap']} iterations, {ci_pct}% CI)")
    print("=" * 65)
    for metric in ["pr_auc", "w_spearman", "cls"]:
        r = results[metric]
        name = {"pr_auc": "PR-AUC", "w_spearman": "W-Spearman", "cls": "CLS"}[metric]
        print(f"  {name:<15s}  {r['point']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
              f"(std={r['std']:.4f})")
    print()

"""
Nested Leave-One-Family-Out blending engine.

Orchestrates:
1. Inner LOFO to generate OOF predictions for each model
2. Weight optimization (grid + scipy refinement)
3. Optional inner-loop hyperparameter tuning
4. Outer fold prediction with best weights/hyperparams
"""

import numpy as np
import pandas as pd
from itertools import product as iprod
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Normalization strategies
# ---------------------------------------------------------------------------

def normalize_minmax(arr):
    """Min-max normalization to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def normalize_rank(arr):
    """Rank-based normalization to [0, 1]."""
    ranks = np.argsort(np.argsort(arr)).astype(float)
    n = len(arr)
    return ranks / (n - 1) if n > 1 else np.zeros_like(arr, dtype=float)


def normalize_zscore(arr):
    """Z-score normalization."""
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sigma


NORMALIZERS = {
    "minmax": normalize_minmax,
    "rank": normalize_rank,
    "zscore": normalize_zscore,
}


# ---------------------------------------------------------------------------
# CLS computation (inline, avoids import cycle)
# ---------------------------------------------------------------------------

def _compute_cls(y_true, y_score, pe_efficiency, epsilon=0.01):
    """Compute CLS = harmonic_mean(PR-AUC, Weighted Spearman)."""
    pr_auc = average_precision_score(y_true, y_score)

    pred_ranks = np.argsort(np.argsort(y_score)).astype(float)
    true_ranks = np.argsort(np.argsort(pe_efficiency)).astype(float)
    w = pe_efficiency + epsilon
    w = w / w.sum()
    mu_p, mu_t = w @ pred_ranks, w @ true_ranks
    dp, dt = pred_ranks - mu_p, true_ranks - mu_t
    cov = (w * dp * dt).sum()
    std_p = np.sqrt((w * dp ** 2).sum())
    std_t = np.sqrt((w * dt ** 2).sum())
    w_sp = cov / (std_p * std_t) if std_p > 1e-12 and std_t > 1e-12 else 0.0
    w_sp = max(w_sp, 0.0)

    if pr_auc <= 0 or w_sp <= 0:
        return 0.0
    return 2.0 * pr_auc * w_sp / (pr_auc + w_sp)


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def _blend_with_weights(weights, oof_arrays, normalize_fn):
    """Blend normalized OOF arrays with given weights."""
    return sum(w * normalize_fn(arr) for w, arr in zip(weights, oof_arrays))


def optimize_weights_grid(inner_oof_arrays, y_true, pe_efficiency, normalize_fn,
                          n_models, step=0.1):
    """
    Grid search over weight simplex.

    Returns (best_weights, best_cls).
    """
    best_cls, best_w = -1.0, np.ones(n_models) / n_models
    for weights in iprod(np.arange(0, 1.0 + step / 2, step), repeat=n_models):
        if abs(sum(weights) - 1.0) > 0.01:
            continue
        w = np.array(weights)
        blend = _blend_with_weights(w, inner_oof_arrays, normalize_fn)
        cls = _compute_cls(y_true, blend, pe_efficiency)
        if cls > best_cls:
            best_cls = cls
            best_w = w
    return best_w, best_cls


def optimize_weights_scipy(inner_oof_arrays, y_true, pe_efficiency, normalize_fn,
                           n_models, init_weights=None):
    """
    Continuous weight optimization via scipy (Nelder-Mead on simplex).

    Uses softmax reparameterization to enforce sum=1 and non-negativity.
    """
    if init_weights is None:
        init_weights = np.ones(n_models) / n_models

    # Softmax reparameterization: raw params -> weights via softmax
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    # Convert init_weights to raw params (inverse softmax = log)
    init_raw = np.log(np.clip(init_weights, 1e-6, None))

    def neg_cls(raw):
        w = softmax(raw)
        blend = _blend_with_weights(w, inner_oof_arrays, normalize_fn)
        return -_compute_cls(y_true, blend, pe_efficiency)

    result = minimize(neg_cls, init_raw, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6})
    best_w = softmax(result.x)
    best_cls = -result.fun
    return best_w, best_cls


# ---------------------------------------------------------------------------
# Model definitions with hyperparameter grids
# ---------------------------------------------------------------------------

class ModelSpec:
    """
    Specification for a model in the blend.

    Parameters
    ----------
    name : str
        Model identifier.
    make_fn : callable(hyperparams) -> callable(X_train, y_train, X_test) -> predictions
        Factory function: given hyperparams dict, returns a model_fn.
    feature_cols : list of str
        Feature column names.
    param_grid : list of dict, optional
        Hyperparameter grid for inner-loop tuning.
        If None, the model uses fixed hyperparameters.
    """

    def __init__(self, name, make_fn, feature_cols, param_grid=None):
        self.name = name
        self.make_fn = make_fn
        self.feature_cols = feature_cols
        self.param_grid = param_grid or [{}]


# ---------------------------------------------------------------------------
# Nested LOFO blend
# ---------------------------------------------------------------------------

def nested_lofo_blend(train_df, splits, model_specs, gt_order,
                      normalize_name="minmax", use_scipy=True,
                      tune_hyperparams=True, verbose=True):
    """
    Run nested LOFO CV with multi-model blending.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with features, pe_efficiency_pct, active, rt_name.
    splits : dict
        family -> list of rt_names.
    model_specs : list of ModelSpec
        Model specifications.
    gt_order : list of str
        RT name ordering for deterministic ranking.
    normalize_name : str
        Normalization strategy: "minmax", "rank", or "zscore".
    use_scipy : bool
        If True, refine grid weights with scipy optimization.
    tune_hyperparams : bool
        If True, search over model param_grids in inner loop.
    verbose : bool
        Print per-fold info.

    Returns
    -------
    dict with keys:
        - oof: pd.DataFrame (57 rows) with rt_name, active, pe_efficiency_pct, predicted_score
        - cls: float
        - per_fold: list of dicts with family, weights, inner_cls, best_params
    """
    normalize_fn = NORMALIZERS[normalize_name]
    families = list(splits.keys())
    n_models = len(model_specs)
    all_oof = []
    per_fold_info = []

    for outer_family in families:
        outer_rts = splits[outer_family]
        outer_mask = train_df["rt_name"].isin(outer_rts)
        inner_mask = ~outer_mask
        inner_splits = {f: rts for f, rts in splits.items() if f != outer_family}
        inner_df = train_df[inner_mask].reset_index(drop=True)

        # Generate all model/hyperparams combos for inner LOFO
        if tune_hyperparams:
            combo_oof, combo_info = _inner_lofo_with_tuning(
                inner_df, inner_splits, model_specs
            )
        else:
            combo_oof, combo_info = _inner_lofo_fixed(
                inner_df, inner_splits, model_specs
            )

        # Find best (hyperparams, weights) on inner OOF
        y_true_inner = inner_df["active"].values
        pe_inner = inner_df["pe_efficiency_pct"].values

        best_cls = -1.0
        best_combo_idx = 0
        best_weights = np.ones(n_models) / n_models

        for cidx, oof_arrays in enumerate(combo_oof):
            # Grid search
            w_grid, cls_grid = optimize_weights_grid(
                oof_arrays, y_true_inner, pe_inner, normalize_fn, n_models
            )
            if use_scipy:
                w_scipy, cls_scipy = optimize_weights_scipy(
                    oof_arrays, y_true_inner, pe_inner, normalize_fn,
                    n_models, init_weights=w_grid
                )
                w, cls = (w_scipy, cls_scipy) if cls_scipy >= cls_grid else (w_grid, cls_grid)
            else:
                w, cls = w_grid, cls_grid

            if cls > best_cls:
                best_cls = cls
                best_combo_idx = cidx
                best_weights = w

        best_params = combo_info[best_combo_idx]

        # Predict outer fold with best hyperparams
        blended = np.zeros(outer_mask.sum())
        best_oof_arrays = combo_oof[best_combo_idx]

        for midx, spec in enumerate(model_specs):
            model_fn = spec.make_fn(best_params[spec.name])
            X_train = train_df.loc[inner_mask, spec.feature_cols].values
            y_train = train_df.loc[inner_mask, "pe_efficiency_pct"].values
            X_test = train_df.loc[outer_mask, spec.feature_cols].values
            pred = model_fn(X_train, y_train, X_test)

            # Normalize using inner OOF statistics
            inner_arr = best_oof_arrays[midx]
            mn, mx = inner_arr.min(), inner_arr.max()
            if normalize_name == "minmax":
                normed = (pred - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(pred)
            elif normalize_name == "rank":
                # For outer fold, use rank within inner OOF + outer
                combined = np.concatenate([inner_arr, pred])
                ranks = np.argsort(np.argsort(combined)).astype(float) / (len(combined) - 1)
                normed = ranks[len(inner_arr):]
            elif normalize_name == "zscore":
                mu, sigma = inner_arr.mean(), inner_arr.std()
                normed = (pred - mu) / sigma if sigma > 1e-12 else np.zeros_like(pred)
            else:
                normed = normalize_fn(pred)

            blended += best_weights[midx] * normed

        fold_df = train_df.loc[outer_mask, ["rt_name", "active", "pe_efficiency_pct", "rt_family"]].copy()
        fold_df["predicted_score"] = blended
        all_oof.append(fold_df)

        if verbose:
            w_str = "  ".join(f"{s.name}={best_weights[i]:.2f}" for i, s in enumerate(model_specs))
            param_str = "  ".join(
                f"{s.name}={best_params[s.name]}" for s in model_specs if best_params[s.name]
            )
            print(f"  {outer_family:<25s} w=[{w_str}]  inner_cls={best_cls:.4f}"
                  + (f"  params=[{param_str}]" if param_str else ""))

        per_fold_info.append({
            "family": outer_family,
            "weights": best_weights.copy(),
            "inner_cls": best_cls,
            "best_params": best_params,
        })

    # Pool and reorder
    oof = pd.concat(all_oof).set_index("rt_name").loc[gt_order].reset_index()

    # Compute global CLS
    cls = _compute_cls(
        oof["active"].values,
        oof["predicted_score"].values,
        oof["pe_efficiency_pct"].values,
    )

    return {"oof": oof, "cls": cls, "per_fold": per_fold_info}


def _inner_lofo_fixed(inner_df, inner_splits, model_specs):
    """Run inner LOFO with fixed hyperparams (single combo)."""
    n = len(inner_df)
    oof_arrays = [np.full(n, np.nan) for _ in model_specs]

    for ifam, irts in inner_splits.items():
        tm = inner_df["rt_name"].isin(irts)
        trm = ~tm
        for midx, spec in enumerate(model_specs):
            model_fn = spec.make_fn(spec.param_grid[0])
            preds = model_fn(
                inner_df.loc[trm, spec.feature_cols].values,
                inner_df.loc[trm, "pe_efficiency_pct"].values,
                inner_df.loc[tm, spec.feature_cols].values,
            )
            oof_arrays[midx][tm.values] = preds

    # combo_info: one combo, one set of (default) params per model
    combo_info = [{spec.name: spec.param_grid[0] for spec in model_specs}]
    return [oof_arrays], combo_info


def _inner_lofo_with_tuning(inner_df, inner_splits, model_specs):
    """Run inner LOFO for all hyperparameter combinations."""
    # Build cartesian product of all param grids
    grids = [spec.param_grid for spec in model_specs]
    all_combos = list(iprod(*grids))

    n = len(inner_df)
    combo_oof = []
    combo_info = []

    for combo in all_combos:
        oof_arrays = [np.full(n, np.nan) for _ in model_specs]

        for ifam, irts in inner_splits.items():
            tm = inner_df["rt_name"].isin(irts)
            trm = ~tm
            for midx, spec in enumerate(model_specs):
                model_fn = spec.make_fn(combo[midx])
                preds = model_fn(
                    inner_df.loc[trm, spec.feature_cols].values,
                    inner_df.loc[trm, "pe_efficiency_pct"].values,
                    inner_df.loc[tm, spec.feature_cols].values,
                )
                oof_arrays[midx][tm.values] = preds

        combo_oof.append(oof_arrays)
        combo_info.append({spec.name: combo[midx] for midx, spec in enumerate(model_specs)})

    return combo_oof, combo_info

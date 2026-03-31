"""Tests for the blend module."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blend import (
    normalize_minmax,
    normalize_rank,
    normalize_zscore,
    optimize_weights_grid,
    optimize_weights_scipy,
    _compute_cls,
)


class TestNormalization:
    def test_minmax_range(self):
        arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = normalize_minmax(arr)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_minmax_constant(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = normalize_minmax(arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_rank_range(self):
        arr = np.array([10.0, 30.0, 20.0, 40.0])
        result = normalize_rank(arr)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_rank_ordering(self):
        arr = np.array([10.0, 30.0, 20.0])
        result = normalize_rank(arr)
        assert result[0] < result[2] < result[1]

    def test_zscore_mean_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_zscore(arr)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_zscore_constant(self):
        arr = np.array([3.0, 3.0, 3.0])
        result = normalize_zscore(arr)
        np.testing.assert_array_equal(result, np.zeros(3))


class TestWeightOptimization:
    @pytest.fixture
    def simple_problem(self):
        """Simple blending problem where model 0 is clearly best."""
        np.random.seed(42)
        n = 50
        y_true = np.array([0] * 30 + [1] * 20)
        pe_eff = np.where(y_true == 1, np.random.rand(n) * 40, 0.0)
        # Model 0: good, Model 1: noise
        oof_0 = pe_eff + np.random.randn(n) * 2  # correlated with target
        oof_1 = np.random.randn(n) * 10  # noise
        return [oof_0, oof_1], y_true, pe_eff

    def test_grid_weights_sum_to_one(self, simple_problem):
        oof_arrays, y_true, pe_eff = simple_problem
        w, cls = optimize_weights_grid(oof_arrays, y_true, pe_eff, normalize_minmax, 2)
        assert abs(w.sum() - 1.0) < 0.02
        assert cls > 0.0

    def test_grid_prefers_good_model(self, simple_problem):
        oof_arrays, y_true, pe_eff = simple_problem
        w, cls = optimize_weights_grid(oof_arrays, y_true, pe_eff, normalize_minmax, 2)
        assert w[0] > w[1], "Weight for good model should be higher"

    def test_scipy_weights_sum_to_one(self, simple_problem):
        oof_arrays, y_true, pe_eff = simple_problem
        w, cls = optimize_weights_scipy(oof_arrays, y_true, pe_eff, normalize_minmax, 2)
        assert abs(w.sum() - 1.0) < 1e-6
        assert all(w >= 0)

    def test_scipy_at_least_as_good_as_grid(self, simple_problem):
        oof_arrays, y_true, pe_eff = simple_problem
        _, cls_grid = optimize_weights_grid(oof_arrays, y_true, pe_eff, normalize_minmax, 2)
        w_grid, _ = optimize_weights_grid(oof_arrays, y_true, pe_eff, normalize_minmax, 2)
        _, cls_scipy = optimize_weights_scipy(
            oof_arrays, y_true, pe_eff, normalize_minmax, 2, init_weights=w_grid
        )
        assert cls_scipy >= cls_grid - 1e-6


class TestInlineCLS:
    def test_matches_metrics_module(self):
        """Inline CLS in blend.py should match metrics.py."""
        from src.metrics import compute_cls as official_cls
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 5 + [1] * 5)
        y_score = rng.rand(10)
        pe_eff = np.where(y_true == 1, rng.rand(10) * 30, 0.0)

        inline_result = _compute_cls(y_true, y_score, pe_eff)
        official_result = official_cls(y_true, y_score, pe_eff)
        assert abs(inline_result - official_result["cls"]) < 1e-10

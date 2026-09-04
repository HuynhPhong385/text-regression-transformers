"""Test spec for src/evaluation/metrics.py (contract-first).

Defines the expected behavior of the four regression metrics required by
knowledge_base/eval.md (MAE primary; MSE, RMSE, R2 secondary).

The module under test (src.evaluation.metrics) does not exist yet — when it is
implemented it must provide exactly these callables, operating on 1-D
array-likes (list/np.ndarray/torch.Tensor accepted):

    mean_absolute_error(y_true, y_pred) -> float
    mean_squared_error(y_true, y_pred) -> float
    root_mean_squared_error(y_true, y_pred) -> float
    r2_score(y_true, y_pred) -> float
    compute_all_metrics(y_true, y_pred) -> dict with keys:
        {"mae", "mse", "rmse", "r2"}

Tests SKIP automatically until that module exists, so this file is safe to
commit now (per docs/TEST_STRATEGY.md "Unit: metric functions").

Ground truth is scikit-learn — the project already depends on it
(requirements.txt) and it is the reference implementation for regression
metrics.
"""

import importlib.util

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

METRICS_MODULE_AVAILABLE = importlib.util.find_spec("src.evaluation.metrics") is not None

requires_metrics_module = pytest.mark.skipif(
    not METRICS_MODULE_AVAILABLE,
    reason="src/evaluation/metrics.py not implemented yet (spec-first test)",
)

# Deterministic synthetic data in target space [0, 1] (ADR-001).
RNG = np.random.default_rng(seed=42)
Y_TRUE = RNG.uniform(0.0, 1.0, size=200)
Y_PRED = np.clip(Y_TRUE + RNG.normal(0.0, 0.1, size=200), 0.0, 1.0)

TOL = 1e-6


@requires_metrics_module
def test_mae_matches_sklearn():
    from src.evaluation.metrics import mean_absolute_error

    assert mean_absolute_error(Y_TRUE, Y_PRED) == pytest.approx(
        sk_metrics.mean_absolute_error(Y_TRUE, Y_PRED), abs=TOL
    )


@requires_metrics_module
def test_mse_matches_sklearn():
    from src.evaluation.metrics import mean_squared_error

    assert mean_squared_error(Y_TRUE, Y_PRED) == pytest.approx(
        sk_metrics.mean_squared_error(Y_TRUE, Y_PRED), abs=TOL
    )


@requires_metrics_module
def test_rmse_is_sqrt_of_mse():
    from src.evaluation.metrics import mean_squared_error, root_mean_squared_error

    mse = mean_squared_error(Y_TRUE, Y_PRED)
    rmse = root_mean_squared_error(Y_TRUE, Y_PRED)
    assert rmse == pytest.approx(np.sqrt(mse), abs=TOL)


@requires_metrics_module
def test_r2_matches_sklearn():
    from src.evaluation.metrics import r2_score

    assert r2_score(Y_TRUE, Y_PRED) == pytest.approx(
        sk_metrics.r2_score(Y_TRUE, Y_PRED), abs=TOL
    )


@requires_metrics_module
def test_perfect_prediction():
    from src.evaluation.metrics import compute_all_metrics

    result = compute_all_metrics(Y_TRUE, Y_TRUE)
    assert result["mae"] == pytest.approx(0.0, abs=TOL)
    assert result["mse"] == pytest.approx(0.0, abs=TOL)
    assert result["rmse"] == pytest.approx(0.0, abs=TOL)
    assert result["r2"] == pytest.approx(1.0, abs=TOL)


@requires_metrics_module
def test_metrics_are_non_negative_where_required():
    from src.evaluation.metrics import compute_all_metrics

    result = compute_all_metrics(Y_TRUE, Y_PRED)
    assert result["mae"] >= 0.0
    assert result["mse"] >= 0.0
    assert result["rmse"] >= 0.0


@requires_metrics_module
def test_accepts_python_lists():
    """Metrics must accept plain lists, not only ndarrays (interface contract)."""
    from src.evaluation.metrics import compute_all_metrics

    y_true = [0.0, 0.5, 1.0]
    y_pred = [0.1, 0.4, 1.1]
    result = compute_all_metrics(y_true, y_pred)
    assert set(result.keys()) == {"mae", "mse", "rmse", "r2"}


@requires_metrics_module
def test_rejects_length_mismatch():
    """Silent broadcasting of mismatched arrays would corrupt metrics; must raise."""
    from src.evaluation.metrics import mean_absolute_error

    with pytest.raises((ValueError, AssertionError)):
        mean_absolute_error([0.1, 0.2, 0.3], [0.1, 0.2])

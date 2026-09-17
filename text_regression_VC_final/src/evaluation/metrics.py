"""metrics.py — Regression metrics for text regression.

Primary metric:   MAE (per knowledge_base/eval.md, docs/EVALUATION.md, ADR-003)
Secondary metrics: MSE, RMSE, R^2 (R2 is coefficient of determination)
Resource metrics: see docs/EVALUATION.md (handled by caller)

Teaching notes
--------------
* Every function validates equal length and rejects empty input — silent
  broadcasting would corrupt an experiment report.
* Comparison in report must use the same data split (R02 Fair comparison) and
  the same target definition ``target = (rating - 1) / 4``.
* Test set is never used for tuning (R03 Test isolation).

Interface contract (kept intentionally small and reuse-friendly)
-----------------------------------------------------------------
    mean_absolute_error(y_true, y_pred)     -> float
    mean_squared_error(y_true, y_pred)      -> float
    root_mean_squared_error(y_true, y_pred) -> float
    r2_score(y_true, y_pred)                -> float
    compute_all_metrics(y_true, y_pred)     -> dict with keys {"mae","mse","rmse","r2"}

Inputs accept any 1-D array-like convertible to ``np.ndarray`` (list,
``np.ndarray``, or ``torch.Tensor`` when torch is installed) so the caller
never has to reshape. ``y_true`` and ``y_pred`` must have the same length.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_array(x: Any) -> np.ndarray:
    """Coerce array-like to float64 1-D ndarray without silent reshaping."""
    # Support torch tensors if the caller holds them; avoid hard dependency.
    if hasattr(x, "detach"):
        try:
            x = x.detach().cpu().numpy()  # type: ignore[attr-defined]
        except Exception:
            pass
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        # A single scalar is not a regression sample; reject early.
        raise ValueError(f"Expected 1-D array-like with >=1 element, got scalar {x!r}")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array-like, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Empty y array")
    return arr


def _validate_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"Length mismatch: len(y_true)={yt.shape[0]} vs len(y_pred)={yp.shape[0]}")
    return yt, yp


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """MAE — primary metric per docs/EVALUATION.md."""
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """MSE — secondary metric."""
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """RMSE = sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination. 1.0 is perfect; 0.0 is mean-baseline."""
    yt, yp = _validate_pair(y_true, y_pred)
    # Center y_true around its own mean (R^2 definition); avoid divide-by-zero
    # when y_true is constant (dataset sanity-check case E00).
    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    if ss_tot == 0.0:
        # Constant target is degenerate for R^2; return 0.0 by convention and
        # let the caller surface the limitation (R06 dataset limitations).
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_all_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Convenience: all four regression metrics in one call."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

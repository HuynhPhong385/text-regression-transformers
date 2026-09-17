"""plots.py — Six required visualizations (docs/EVALUATION.md, knowledge_base/eval.md).

Required:
 1. loss curve                 (training_history.csv)
 2. prediction vs target       (predictions.csv or y_true/y_pred)
 3. residual (distribution)    (y_true vs y_pred)
 4. error bucket (bar)         (per docs/ERROR_ANALYSIS.md)
 5. model comparison           (summary.csv or metrics dict)
 6. resource/performance trade-off (accuracy vs params/latency/memory)

All functions accept an explicit ``out_path`` and create parent dirs. They never
write outside the supplied path (audit rule: no side-effects). When matplotlib
is unavailable, functions raise a clear ImportError so the caller can decide
whether to degrade gracefully (e.g. in a smoke test).

Helpers that callers use directly
----------------------------------
    plot_loss(history_df, out_path)
    plot_prediction_vs_target(y_true, y_pred, out_path)
    plot_residual(y_true, y_pred, out_path)
    plot_error_buckets(errors, out_path)        # absolute errors 1-D
    plot_model_comparison(metrics_df, out_path) # cols: model, metric value
    plot_resource_tradeoff(df, x, y, out_path)  # df with x/y cols, hue=model

``history_df`` must carry columns named by the trainer's history CSV
(``epoch, train_loss`` and optional ``val_loss``). ``errors`` is 1-D
absolute errors in [0, 1]. ``metrics_df`` should be long-form where each
row is one model and one metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:  # pragma: no cover - import error is environment-specific
        raise ImportError(
            "matplotlib is required for src/evaluation/plots.py. "
            "Install it with: pip install matplotlib"
        ) from e


def plot_loss(history_df: pd.DataFrame, out_path: str | Path) -> Path:
    """1. Loss curve: train_loss (and val_loss if present) over epoch."""
    if "epoch" not in history_df.columns or "train_loss" not in history_df.columns:
        raise ValueError("history_df must contain epoch and train_loss columns")
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss", marker="o")
    if "val_loss" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSELoss)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_prediction_vs_target(
    y_true: Any,
    y_pred: Any,
    out_path: str | Path,
) -> Path:
    """2. Scatter: prediction vs target with y=x reference."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true/y_pred length mismatch")
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(yt, yp, s=10, alpha=0.35)
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="y=x")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_residual(y_true: Any, y_pred: Any, out_path: str | Path) -> Path:
    """3. Residual distribution: histogram of (prediction - target)."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true/y_pred length mismatch")
    resid = yp - yt
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(resid, bins=40, edgecolor="white")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("residual (prediction - target)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_error_buckets(errors: Any, out_path: str | Path) -> Path:
    """4. Bar chart of error bucket counts (docs/ERROR_ANALYSIS.md)."""
    from src.evaluation.error_analysis import bucket_counts

    arr = np.asarray(errors, dtype=float).reshape(-1)
    counts = bucket_counts(arr)
    labels = list(counts.keys())
    values = list(counts.values())
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, values)
    ax.set_xlabel("absolute error bucket")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: str | Path) -> Path:
    """5. Bar chart comparing one metric across models.

    metrics_df must have columns ``model`` and the metric column(s); the
    caller decides which metric to compare. The implementation treats the
    second column as the value and the first as the label so that generic
    inputs such as ``summary.csv`` still render.
    """
    if metrics_df.shape[1] < 2:
        raise ValueError("metrics_df needs at least 2 columns (model, value)")
    label_col = metrics_df.columns[0]
    value_col = metrics_df.columns[1]
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(metrics_df[label_col].astype(str), metrics_df[value_col].astype(float))
    ax.set_xlabel(label_col)
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_resource_tradeoff(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: str | Path,
    label_col: str | None = None,
) -> Path:
    """6. Scatter: resource metric (x) vs accuracy (y), e.g. params vs MAE."""
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns not found: need {x!r} and {y!r} in {list(df.columns)}")
    plt = _require_mpl()
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df[x].astype(float), df[y].astype(float))
    if label_col and label_col in df.columns:
        for _, r in df.iterrows():
            ax.annotate(str(r[label_col]), (float(r[x]), float(r[y])), fontsize=7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

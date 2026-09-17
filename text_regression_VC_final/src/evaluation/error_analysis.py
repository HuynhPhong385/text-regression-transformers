"""error_analysis.py — Residuals, error buckets, and top-k error samples.

Spec source: docs/ERROR_ANALYSIS.md + .ai/skills/evaluation.md.
Teaching notes:
- Error = absolute_error = |target - prediction|; never reuse the signed residual.
- Bucket edges are fixed: 0.00–0.05 / 0.05–0.10 / 0.10–0.20 / 0.20–0.30 / >0.30.
- Always return both largest- and smallest-error samples (20 of each) so that
  later qualitative inspection is not cherry-picked.
- Patterns to investigate later: sarcasm, mixed sentiment, ambiguous language,
  long context, rating/text mismatch — but do not infer causal claims.

Public helpers:
    bucket_counts(errors, buckets=None)         -> dict[str, int]
    top_errors(df, n=20)                        -> pd.DataFrame
    top_correct(df, n=20)                       -> pd.DataFrame
    build_error_report(df, n=20, buckets=None)  -> dict  (counts + frames)
    residual_stats(y_true, y_pred)              -> dict  (y_true/y_pred length validated)

Expected DataFrame columns for df-based helpers: ``text, target, prediction,
absolute_error``. ``y_true/y_pred`` pair helpers for residual_stats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Fixed bucket edges per docs/ERROR_ANALYSIS.md — do not make these
# configurable in evaluation call-sites (keep reporting comparable).
BUCKET_EDGES = [0.0, 0.05, 0.10, 0.20, 0.30, float("inf")]
BUCKET_LABELS = ["0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", ">0.30"]


def bucket_counts(
    errors: Any,
    buckets: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict[str, int]:
    """Count samples per error bucket. Input is 1-D absolute errors."""
    buckets = buckets or BUCKET_EDGES
    labels = labels or BUCKET_LABELS
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    # Fixed-bucket fast path: single digitize pass so sum(counts) == len(errors)
    # is guaranteed and easy to audit. Buckets are [0,0.05), [0.05,0.10),
    # [0.10,0.20), [0.20,0.30), [0.30, inf) — right edge exclusive except the
    # last bucket is open-ended. 0.05 lands in 0.05-0.10 by design.
    if buckets is BUCKET_EDGES and labels is BUCKET_LABELS:
        cuts = np.array([0.05, 0.10, 0.20, 0.30])
        idx = np.digitize(arr, cuts, right=False)
        return {label: int(np.sum(idx == i)) for i, label in enumerate(labels)}
    # Generic path for custom bucket configs (e.g. ablation).
    counts: dict[str, int] = {}
    for i, (lo, hi, label) in enumerate(zip(buckets[:-1], buckets[1:], labels)):
        if np.isinf(hi):
            counts[label] = int(np.sum(arr >= lo))
        else:
            # Include exact upper edge only for the last finite bucket's
            # neighbour; simple half-open [lo, hi) otherwise.
            counts[label] = int(np.sum((arr >= lo) & (arr < hi)))
    return counts


def _require_columns(df: pd.DataFrame) -> None:
    missing = {"text", "target", "prediction", "absolute_error"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


def top_errors(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Largest absolute errors first."""
    _require_columns(df)
    return df.sort_values("absolute_error", ascending=False).head(n).reset_index(drop=True)


def top_correct(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Smallest absolute errors first."""
    _require_columns(df)
    return df.sort_values("absolute_error", ascending=True).head(n).reset_index(drop=True)


def build_error_report(
    df: pd.DataFrame,
    n: int = 20,
    buckets: list[float] | None = None,
) -> dict[str, Any]:
    """Bundle counts + both top-k frames for downstream plots/report.

    Returns dict with keys: ``bucket_counts, top_worst, top_best, n, total``.
    """
    _require_columns(df)
    counts = bucket_counts(df["absolute_error"].to_numpy(dtype=float), buckets=buckets)
    return {
        "bucket_counts": counts,
        "top_worst": top_errors(df, n=n),
        "top_best": top_correct(df, n=n),
        "n": n,
        "total": int(len(df)),
    }


def residual_stats(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Signed residual = prediction - target (useful for bias plots)."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"Length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    resid = yp - yt
    return {
        "residual": resid,
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid)),
        "mae": float(np.mean(np.abs(resid))),
        "n": int(yt.shape[0]),
    }

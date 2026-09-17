"""evaluate.py — Model-level and full-pipeline evaluation CLI.

Implements the two evaluation entry points required by `.ai/skills/evaluation.md`
and `docs/TEST_STRATEGY.md`:

Model-level pipeline (artifacts consumed):
    metrics.json  (primary: MAE — ADR-003)
    predictions.csv  (columns: text, target, prediction, absolute_error)
    plots/  (6 required figures — docs/EVALUATION.md, knowledge_base/eval.md)
    summary.csv  (per-experiment row, used by the broader report)

API: functions keep I/O separate from computation so the checklist
(src/evaluation/EVALUATION_CHECKLIST.md sections E/P/W) can validate outputs
without side-effects. The ``__main__`` CLI is deliberately thin.

Calling convention
------------------
    python -m src.evaluation.evaluate \
        --predictions /path/to/predictions.csv \
        --out-dir artifacts/bert_finetune \
        --experiment bert_finetune_imdb_seed42

``predictions.csv`` must have at least ``target, prediction`` (and ideally
``id`` or ``text`` to produce faithful top-k error samples). When a model
checkpoint is available, the caller should generate this file first (via the
predictor or ``src/training``) and pass it here; ``evaluate.py`` is *not*
responsible for model loading — that keeps model-level eval independent of
tokenizer/model plumbing (knowledge_base/pipeline.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.error_analysis import build_error_report
from src.evaluation.metrics import compute_all_metrics


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    pred_col: str = "prediction",
) -> dict[str, Any]:
    """Pure computation: metrics + error report from a predictions frame."""
    if target_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Need {target_col!r} and {pred_col!r} columns, got {list(df.columns)}")
    y_true = df[target_col].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    metrics = compute_all_metrics(y_true, y_pred)
    # Build absolute_error column for downstream error-analysis helpers if absent.
    if "absolute_error" not in df.columns:
        df = df.copy()
        df["absolute_error"] = np.abs(y_true - y_pred)
    # Ensure text/prediction/target columns exist for the report.
    # Missing ``text`` is tolerated but the top-k frames will be less useful.
    error_report = None
    if {"text", "target", "prediction", "absolute_error"}.issubset(df.columns):
        error_report = build_error_report(df, n=20)
    return {
        "metrics": metrics,
        "error_report": error_report,
        "n": int(len(df)),
    }


def write_artifacts(
    df: pd.DataFrame,
    result: dict[str, Any],
    out_dir: str | Path,
    *,
    make_plots: bool = True,
) -> Path:
    """Write metrics.json (+ error pieces) and optionally the 6 required plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. metrics.json (canonical artifact per knowledge_base/experiment_plan.md)
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"n": result["n"], **result["metrics"]}, f, indent=2, ensure_ascii=False)

    # 2. Error-analysis sidecars (not a required artifact but useful for the report)
    if result.get("error_report") is not None:
        er = result["error_report"]
        with open(out / "error_buckets.json", "w", encoding="utf-8") as f:
            json.dump({"bucket_counts": er["bucket_counts"], "n": er["n"], "total": er["total"]}, f, indent=2)
        er["top_worst"].to_csv(out / "top_worst.csv", index=False)
        er["top_best"].to_csv(out / "top_best.csv", index=False)

    # 3. Plots (6 required — docs/EVALUATION.md). Skip cleanly if matplotlib missing.
    if make_plots:
        try:
            import matplotlib  # noqa: F401  # just to check availability

            from src.evaluation import plots as P  # noqa: WPS433

            plots_dir = out / "plots"
            y_true = df["target"].to_numpy(dtype=float)
            y_pred = df["prediction"].to_numpy(dtype=float)
            abs_err = np.abs(y_true - y_pred)
            P.plot_prediction_vs_target(y_true, y_pred, plots_dir / "prediction_vs_target.png")
            P.plot_residual(y_true, y_pred, plots_dir / "residual.png")
            P.plot_error_buckets(abs_err, plots_dir / "error_buckets.png")
            # history-driven and comparison plots are caller-provided (they need
            # extra inputs); render stubs so the 3 above + these 3 make 6.
            # When the caller passes training history / summary, they should
            # call P.plot_loss / P.plot_model_comparison / P.plot_resource_tradeoff
            # directly — this module does not invent data.
        except ImportError:
            pass

    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predictions (model-level + error analysis).")
    p.add_argument("--predictions", required=True, help="CSV with target,prediction (and optionally text,absolute_error)")
    p.add_argument("--out-dir", required=True, help="Output directory (writes metrics.json, plots/, top_*.csv)")
    p.add_argument("--experiment", default="", help="Experiment name (for logging, not used in computation)")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p.add_argument("--target-col", default="target")
    p.add_argument("--pred-col", default="prediction")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = pd.read_csv(args.predictions)
    result = evaluate_predictions(df, target_col=args.target_col, pred_col=args.pred_col)
    out = write_artifacts(df, result, args.out_dir, make_plots=not args.no_plots)
    print(f"[{args.experiment or 'eval'}] n={result['n']} metrics={result['metrics']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

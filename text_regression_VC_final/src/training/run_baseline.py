"""Baseline — mean predictor + TF-IDF + Linear Regression."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.config import load_config, set_seed


def run_baseline(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    csv_path = Path("data/processed/imdb_processed.csv")
    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "validation"]
    df_test = df[df["split"] == "test"]

    exp_name = cfg.get("experiment_name", "baseline_imdb_seed42")
    artifact_dir = Path(f"artifacts/{exp_name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Mean baseline (MAE floor)
    train_mean = float(df_train["target"].mean())
    y_test = df_test["target"].to_numpy()
    mean_preds = [train_mean] * len(y_test)
    mean_metrics = {
        "mae": float(mean_absolute_error(y_test, mean_preds)),
        "mse": float(mean_squared_error(y_test, mean_preds)),
        "r2": float(r2_score(y_test, mean_preds)),
    }

    # 2. TF-IDF + Linear Regression
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(df_train["text"].astype(str))
    X_test = vectorizer.transform(df_test["text"].astype(str))
    y_train = df_train["target"].to_numpy()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    # Clamp to [0, 1]
    y_pred = y_pred.clip(0, 1)

    tfidf_metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2": float(r2_score(y_test, y_pred)),
        "mean_baseline_mae": mean_metrics["mae"],
    }
    tfidf_metrics["train_time_s"] = time.time() - t0

    import shutil
    shutil.copy(str(config_path), str(artifact_dir / "config.yaml"))

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(tfidf_metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "id": df_test["id"].values,
        "text": df_test["text"].values,
        "target": y_test,
        "prediction": y_pred,
    })
    pred_df["absolute_error"] = (pred_df["target"] - pred_df["prediction"]).abs()
    pred_df.to_csv(artifact_dir / "predictions.csv", index=False)

    with open(artifact_dir / "model_metadata.json", "w") as f:
        json.dump({"experiment": exp_name, "baseline": "tfidf_linear_regression", **tfidf_metrics}, f, indent=2)

    # Training history is trivial for baseline
    with open(artifact_dir / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        w.writeheader()
        w.writerow({"epoch": 1, "train_loss": tfidf_metrics["mse"]})

    print(f"Baseline {exp_name}: {tfidf_metrics}")
    return tfidf_metrics

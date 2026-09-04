"""CLI training script for Member 3."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import (
    load_dataset,
    split_dataset,
    RegressionTextDataset,
)
from src.data.preprocessing import clean_text
from src.evaluation.metrics import calculate_metrics, save_json
from src.models.model_factory import (
    EXPERIMENTS,
    ModelFactory,
    resolve_experiment,
)
from src.training.trainer import Trainer


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sklearn(
    experiment_id,
    train,
    val,
    test,
    output,
):
    """Train E01/E02 and save model.joblib + result.json."""
    experiment_dir = output / experiment_id
    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = ModelFactory.create(experiment_id)

    model.fit(
        train["text"],
        train["target"],
    )

    val_pred = model.predict(val["text"])
    test_pred = model.predict(test["text"])

    val_metrics = calculate_metrics(
        val["target"],
        val_pred,
    )
    test_metrics = calculate_metrics(
        test["target"],
        test_pred,
    )

    model_path = (
        experiment_dir / "model.joblib"
    )

    model.save(model_path)

    result = {
        "experiment": experiment_id,
        "name": EXPERIMENTS[experiment_id]["name"],
        "validation": val_metrics,
        "test": test_metrics,
        "model": str(model_path),
    }

    save_json(
        result,
        experiment_dir / "result.json",
    )

    pd.DataFrame(
        {
            "target": test["target"],
            "prediction": test_pred,
            "error": (
                np.asarray(test_pred)
                - np.asarray(test["target"])
            ),
        }
    ).to_csv(
        experiment_dir / "predictions.csv",
        index=False,
    )

    print(
        f"{experiment_id} | "
        f"MAE={test_metrics['MAE']:.4f} | "
        f"RMSE={test_metrics['RMSE']:.4f} | "
        f"R2={test_metrics['R2']:.4f}"
    )

    return result


def run_transformer(
    experiment_id,
    train,
    val,
    test,
    output,
    epochs,
    batch_size,
    max_length,
    learning_rate,
    patience,
):
    """Train E03-E06."""
    checkpoint = ModelFactory.tokenizer_name(
        experiment_id
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    train_ds = RegressionTextDataset(
        train["text"],
        train["target"],
        tokenizer,
        max_length,
    )

    val_ds = RegressionTextDataset(
        val["text"],
        val["target"],
        tokenizer,
        max_length,
    )

    test_ds = RegressionTextDataset(
        test["text"],
        test["target"],
        tokenizer,
        max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model = ModelFactory.create(
        experiment_id
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output,
        experiment_id=experiment_id,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    return trainer.fit()


def run_experiment(
    experiment_id,
    train,
    val,
    test,
    output,
    args,
):
    config = EXPERIMENTS[experiment_id]

    print(
        f"\n===== {experiment_id}: "
        f"{config['name']} ====="
    )

    if config["type"] == "sklearn":
        return run_sklearn(
            experiment_id,
            train,
            val,
            test,
            output,
        )

    return run_transformer(
        experiment_id,
        train,
        val,
        test,
        output,
        args.epochs,
        args.batch_size,
        args.max_length,
        args.lr,
        args.patience,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Member 3 — Model & Training"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Đường dẫn CSV, ví dụ data/dataset.csv",
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "mean/baseline, tfidf/linear, "
            "bert_frozen, bert, roberta, "
            "distilbert, E01-E06 hoặc all"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epoch cho Transformer.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size cho Transformer.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate cho Transformer.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Thư mục output.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs phải >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")

    if args.max_length < 1:
        parser.error("--max-length phải >= 1")

    if args.patience < 1:
        parser.error("--patience phải >= 1")

    set_seed()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(args.data)
    df["text"] = df["text"].apply(clean_text)

    train, val, test = split_dataset(df)

    print(
        f"Dataset: {len(df)} rows | "
        f"train={len(train)} | "
        f"val={len(val)} | "
        f"test={len(test)}"
    )

    resolved = resolve_experiment(args.model)

    experiment_ids = (
        list(EXPERIMENTS.keys())
        if resolved == "all"
        else [resolved]
    )

    results = []

    for experiment_id in experiment_ids:
        results.append(
            run_experiment(
                experiment_id,
                train,
                val,
                test,
                output,
                args,
            )
        )

    # Summary chung khi chạy all.
    if len(results) > 1:
        summary = []

        for result in results:
            metrics = result.get("test", {})

            summary.append(
                {
                    "experiment": result["experiment"],
                    "name": result.get(
                        "name",
                        EXPERIMENTS[
                            result["experiment"]
                        ]["name"],
                    ),
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "R2": metrics.get("R2"),
                }
            )

        summary_df = pd.DataFrame(summary)

        summary_df.to_csv(
            output / "experiment_summary.csv",
            index=False,
        )

        save_json(
            summary,
            output / "experiment_summary.json",
        )

        print("\n===== EXPERIMENT SUMMARY =====")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

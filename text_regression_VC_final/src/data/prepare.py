"""Data preparation — build canonical IMDb dataset into data/processed/.

Pipeline: load raw (HF stanfordnlp/imdb) -> validate -> clean -> target conversion
          -> split (official train/test, then 10% val from train) -> save CSV + metadata

Usage:
    conda run -n base python -m src.data.prepare
    conda run -n base python -m src.data.prepare --max-samples 1000  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validate import check_leakage, dedup_report, validate_schema

DEFAULT_SEED = 42
OUTPUT_DIR = Path("data/processed")


def rating_to_target(rating: float) -> float:
    return (rating - 1.0) / 4.0


def prepare_imdb(
    seed: int = DEFAULT_SEED,
    max_samples: int | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    from datasets import load_dataset

    random.seed(seed)
    np.random.seed(seed)

    ds = load_dataset("stanfordnlp/imdb")
    # IMDb is binary: label 0=neg (1 star), 1=pos (5 stars). Map to rating 1/5.
    def to_rows(split_name: str, split_ds):
        rows = []
        for i, ex in enumerate(split_ds):
            text = ex["text"].replace("<br />", " ").strip()
            label = int(ex["label"])
            rating = 5 if label == 1 else 1
            target = rating_to_target(rating)
            rows.append({
                "id": f"{split_name}_{i}",
                "text": text,
                "rating": rating,
                "target": target,
                "split": split_name,  # temporary, will re-split train -> train/val
            })
        return rows

    train_rows = to_rows("train", ds["train"])
    test_rows = to_rows("test", ds["test"])

    if max_samples is not None:
        # Subsample for smoke tests — keep proportions
        half = max_samples // 2
        train_rows = train_rows[:half]
        test_rows = test_rows[:half]

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    # Split train into train/validation 90/10 (stratified by rating if possible)
    # 80/10/10 overall: train holds 90% of original train (which is 50% of total)
    # For exact 80/10/10 on full data, take 90% train / 10% val from the 25k train split
    # so validation ~2500.
    df_train_shuffled = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(df_train_shuffled) * 0.1))
    df_val = df_train_shuffled.iloc[:n_val].copy()
    df_val["split"] = "validation"
    df_train_final = df_train_shuffled.iloc[n_val:].copy()
    df_train_final["split"] = "train"

    df_test = df_test.copy()
    df_test["split"] = "test"

    df_all = pd.concat([df_train_final, df_val, df_test], ignore_index=True)

    # Cleaning
    df_all["text"] = df_all["text"].astype(str).str.strip()
    df_all = df_all[df_all["text"] != ""]
    df_all = df_all.drop_duplicates(subset=["text"], keep="first")

    # Validate
    errors = validate_schema(df_all)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")

    train_ids = set(df_all[df_all["split"] == "train"]["id"])
    val_ids = set(df_all[df_all["split"] == "validation"]["id"])
    test_ids = set(df_all[df_all["split"] == "test"]["id"])
    leak = check_leakage(train_ids, val_ids, test_ids)
    if leak:
        raise ValueError(f"Data leakage: {leak}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "imdb_processed.csv"
    df_all.to_csv(csv_path, index=False)

    meta = {
        "dataset": "imdb",
        "seed": seed,
        "total": int(len(df_all)),
        "counts": df_all["split"].value_counts().to_dict(),
        "dedup": dedup_report(df_all),
        "sha256": hashlib.sha256(pd.util.hash_pandas_object(df_all, index=True).values.tobytes()).hexdigest()[:16],
        "leakage_check": "PASS" if not leak else "FAIL",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(df_all)} rows to {csv_path}")
    print(json.dumps(meta, indent=2))
    return df_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    prepare_imdb(seed=args.seed, max_samples=args.max_samples, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()

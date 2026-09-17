"""Data validation — canonical schema and quality checks (knowledge_base/data_prep.md)."""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS = ["id", "text", "rating", "target", "split"]
VALID_SPLITS = {"train", "validation", "test"}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return list of error messages; empty means PASS."""
    errors: list[str] = []
    missing_cols = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")

    if "text" in df.columns:
        null_text = int(df["text"].isna().sum())
        if null_text:
            errors.append(f"text has {null_text} null values")
        empty_text = int((df["text"].astype(str).str.strip() == "").sum())
        if empty_text:
            errors.append(f"text has {empty_text} empty strings")

    if "target" in df.columns:
        null_t = int(df["target"].isna().sum())
        if null_t:
            errors.append(f"target has {null_t} null values")
        out_of_range = int(((df["target"] < 0) | (df["target"] > 1)).sum())
        if out_of_range:
            errors.append(f"target has {out_of_range} values outside [0,1]")

    if "split" in df.columns:
        invalid = set(df["split"].unique()) - VALID_SPLITS
        if invalid:
            errors.append(f"Invalid split values: {invalid}")

    return errors


def check_leakage(train_ids: set, val_ids: set, test_ids: set) -> list[str]:
    errors: list[str] = []
    if train_ids & val_ids:
        errors.append(f"Leakage train∩val: {len(train_ids & val_ids)} ids")
    if train_ids & test_ids:
        errors.append(f"Leakage train∩test: {len(train_ids & test_ids)} ids")
    if val_ids & test_ids:
        errors.append(f"Leakage val∩test: {len(val_ids & test_ids)} ids")
    return errors


def dedup_report(df: pd.DataFrame) -> dict:
    dup_ids = int(df.duplicated(subset=["id"]).sum())
    dup_text = int(df.duplicated(subset=["text"]).sum())
    return {"duplicate_ids": dup_ids, "duplicate_text": dup_text, "total": len(df)}

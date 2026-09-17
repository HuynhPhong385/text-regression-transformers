"""Dataset helpers — IMDbDataset and collation utilities.

Canonical schema: id, text, rating, target, split
Target:          (rating - 1) / 4   -> [0, 1]
Tokenizer:       AutoTokenizer, max_length=256, truncation, dynamic padding
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class IMDbDataset(Dataset):
    """Torch dataset wrapping a DataFrame in canonical schema."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 256,
        text_col: str = "text",
        target_col: str = "target",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text: str = str(row[self.text_col])
        target: float = float(row[self.target_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding via collate_fn
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target": target,
            "text": text,
            "id": str(row["id"]) if "id" in row else str(idx),
        }


def collate_fn(batch: list[dict], tokenizer: Any, max_length: int = 256) -> dict[str, torch.Tensor]:
    """Dynamic padding collate — pads to longest in batch, up to max_length."""
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    # Use tokenizer's pad utility if available
    if hasattr(tokenizer, "pad"):
        enc = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_t = enc["input_ids"]
        attention_mask_t = enc["attention_mask"]
    else:
        # Fallback manual padding
        max_len = min(max(len(x) for x in input_ids), max_length)
        input_ids_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            l = min(len(ids), max_len)
            input_ids_t[i, :l] = torch.tensor(ids[:l], dtype=torch.long)
            attention_mask_t[i, :l] = torch.tensor(mask[:l], dtype=torch.long)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids_t, "attention_mask": attention_mask_t, "target": targets}

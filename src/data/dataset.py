"""Dataset loading, splitting and Transformer Dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_dataset(path, text_col="text", target_col="target"):
    """Đọc CSV, validate text/target và loại dòng lỗi."""
    df = pd.read_csv(path)

    missing = [
        column
        for column in [text_col, target_col]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV thiếu cột bắt buộc: {missing}"
        )

    df = df[[text_col, target_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    # Loại dòng target không hợp lệ và text rỗng.
    df = df.dropna(subset=[target_col])
    df = df[df[text_col].str.strip() != ""]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "Dataset không còn dòng hợp lệ sau khi validate."
        )

    return df


# Alias để tương thích code cũ.
load_data = load_dataset


def split_dataset(
    df,
    test_size=0.1,
    val_size=0.1,
    seed=42,
):
    """
    Chia train/val/test theo 80/10/10.

    Dataset rất nhỏ sẽ fallback: dùng toàn bộ dữ liệu cho
    train/val/test để pipeline có thể chạy kiểm tra.
    Không nên dùng kết quả fallback để đánh giá thống kê.
    """
    n = len(df)

    # Dataset quá nhỏ: tránh train_test_split crash.
    if n < 10:
        print(
            "[WARNING] Dataset quá nhỏ. "
            "Dùng toàn bộ dữ liệu cho train/val/test "
            "chỉ để kiểm tra pipeline."
        )
        copy = df.reset_index(drop=True).copy()
        return copy.copy(), copy.copy(), copy.copy()

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
    )

    relative_val = val_size / (1 - test_size)

    # Nếu phần train_val vẫn quá nhỏ thì fallback.
    if len(train_val) < 3:
        copy = df.reset_index(drop=True).copy()
        return copy.copy(), copy.copy(), copy.copy()

    try:
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            random_state=seed,
        )
    except ValueError:
        copy = df.reset_index(drop=True).copy()
        return copy.copy(), copy.copy(), copy.copy()

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# Alias để tương thích code cũ.
split_data = split_dataset


class RegressionTextDataset(Dataset):
    """PyTorch Dataset cho text regression + Transformer tokenizer."""

    def __init__(
        self,
        texts,
        targets,
        tokenizer,
        max_length=256,
    ):
        self.texts = list(texts)
        self.targets = np.asarray(
            targets,
            dtype=np.float32,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.targets[index],
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = (
                encoded["token_type_ids"].squeeze(0)
            )

        return item


# Alias để tương thích code cũ.
TextRegressionDataset = RegressionTextDataset

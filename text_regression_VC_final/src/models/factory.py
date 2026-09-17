"""Model factory — resolves model_name + strategy + dropout into a model instance."""

from __future__ import annotations

from src.models.regression_model import TextRegressionModel

SUPPORTED_MODELS = {
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
}

SUPPORTED_STRATEGIES = {"frozen", "finetune"}


def create_model(model_name: str, strategy: str = "finetune", dropout: float = 0.1) -> TextRegressionModel:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}, choose from {SUPPORTED_MODELS}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy!r}")
    return TextRegressionModel(model_name=model_name, dropout=dropout, strategy=strategy)

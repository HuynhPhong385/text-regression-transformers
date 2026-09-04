"""Model Factory and experiment registry for Member 3 (E01-E06)."""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from .base import BaseSklearnModel
from .transformer import TransformerRegressionModel

EXPERIMENTS = {
    "E01": {"name": "Mean Baseline", "type": "sklearn", "aliases": ["mean", "baseline"]},
    "E02": {"name": "TF-IDF + Linear Regression", "type": "sklearn", "aliases": ["tfidf", "linear"]},
    "E03": {"name": "BERT Frozen", "type": "torch", "checkpoint": "bert-base-uncased", "freeze_encoder": True, "aliases": ["bert_frozen"]},
    "E04": {"name": "BERT Fine-tuning", "type": "torch", "checkpoint": "bert-base-uncased", "freeze_encoder": False, "aliases": ["bert"]},
    "E05": {"name": "RoBERTa", "type": "torch", "checkpoint": "roberta-base", "freeze_encoder": False, "aliases": ["roberta"]},
    "E06": {"name": "DistilBERT", "type": "torch", "checkpoint": "distilbert-base-uncased", "freeze_encoder": False, "aliases": ["distilbert"]},
}

CLI_TO_EXPERIMENT = {
    key: experiment_id
    for experiment_id, config in EXPERIMENTS.items()
    for key in [experiment_id.lower(), *config["aliases"]]
}

PRETRAINED_CHECKPOINTS = {
    experiment_id: config["checkpoint"]
    for experiment_id, config in EXPERIMENTS.items()
    if "checkpoint" in config
}


class MeanBaseline(BaseSklearnModel):
    """E01: predict the mean target from the training set."""
    def __init__(self):
        self.mean_value = None

    def fit(self, texts, targets):
        self.mean_value = float(np.mean(targets))
        return self

    def predict(self, texts):
        if self.mean_value is None:
            raise RuntimeError("Model chưa được fit.")
        return np.full(len(texts), self.mean_value, dtype=float)

    def save(self, path):
        joblib.dump(self, path)


class TfidfLinearRegression(BaseSklearnModel):
    """E02: TF-IDF features + Linear Regression."""
    def __init__(self, max_features=20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        self.regressor = LinearRegression()

    def fit(self, texts, targets):
        x = self.vectorizer.fit_transform(texts)
        self.regressor.fit(x, targets)
        return self

    def predict(self, texts):
        return self.regressor.predict(self.vectorizer.transform(texts))

    def save(self, path):
        joblib.dump(self, path)


class ModelFactory:
    """Create the requested model without extra model/training files."""
    @staticmethod
    def create(experiment_id):
        if experiment_id == "E01":
            return MeanBaseline()
        if experiment_id == "E02":
            return TfidfLinearRegression()
        if experiment_id in PRETRAINED_CHECKPOINTS:
            config = EXPERIMENTS[experiment_id]
            return TransformerRegressionModel(
                checkpoint=config["checkpoint"],
                freeze_encoder=config["freeze_encoder"],
                hidden_size=128,
                dropout=0.1,
            )
        raise ValueError(f"Không tìm thấy model cho {experiment_id}.")

    @staticmethod
    def tokenizer_name(experiment_id):
        if experiment_id not in PRETRAINED_CHECKPOINTS:
            raise ValueError(f"{experiment_id} không phải Transformer.")
        return PRETRAINED_CHECKPOINTS[experiment_id]


def resolve_experiment(value):
    key = value.lower()
    if key == "all":
        return "all"
    if key not in CLI_TO_EXPERIMENT:
        valid = ", ".join(list(CLI_TO_EXPERIMENT.keys()) + ["all"])
        raise ValueError(f"Model '{value}' không hợp lệ. Chọn: {valid}")
    return CLI_TO_EXPERIMENT[key]

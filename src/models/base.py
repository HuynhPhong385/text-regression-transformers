"""Abstract interfaces for sklearn and PyTorch models.

E01/E02 dùng API:
    fit(texts, targets)
    predict(texts)

E03-E06 dùng API:
    forward(**inputs)
"""

from abc import ABC, abstractmethod
from torch import nn


class BaseSklearnModel(ABC):
    """Base interface cho model sklearn."""

    @abstractmethod
    def fit(self, texts, targets):
        raise NotImplementedError

    @abstractmethod
    def predict(self, texts):
        raise NotImplementedError


class BaseTorchModel(nn.Module, ABC):
    """Base interface cho model PyTorch."""

    @abstractmethod
    def forward(self, **inputs):
        raise NotImplementedError

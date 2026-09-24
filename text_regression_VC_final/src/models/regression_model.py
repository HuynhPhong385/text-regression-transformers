"""Text regression model — Transformer encoder + regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextRegressionModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, strategy: str = "finetune"):
        super().__init__()
        self.model_name = model_name
        self.strategy = strategy
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

        if strategy == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # First-token representation (CLS / <s>)
        hidden = outputs.last_hidden_state[:, 0, :]
        hidden = self.dropout(hidden)
        logits = self.regressor(hidden).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)  # [0, 1]

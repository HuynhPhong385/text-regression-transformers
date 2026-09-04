"""Generic Transformer text-regression model for E03-E06."""

from torch import nn
from transformers import AutoModel

from .base import BaseTorchModel
from .regression_head import RegressionHead


class TransformerRegressionModel(BaseTorchModel):
    """
    input text
        -> tokenizer
        -> transformer encoder
        -> masked mean pooling
        -> RegressionHead
        -> float prediction
    """

    def __init__(
        self,
        checkpoint,
        freeze_encoder=False,
        hidden_size=128,
        dropout=0.1,
    ):
        super().__init__()

        self.checkpoint = checkpoint
        self.encoder = AutoModel.from_pretrained(checkpoint)

        encoder_size = self.encoder.config.hidden_size

        self.regression_head = RegressionHead(
            input_size=encoder_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """Đóng băng toàn bộ Transformer encoder."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        """Cho phép fine-tuning toàn bộ encoder."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    @staticmethod
    def mean_pooling(hidden_state, attention_mask):
        """Masked mean pooling, bỏ qua padding token."""
        mask = attention_mask.unsqueeze(-1).expand(
            hidden_state.size()
        ).float()

        masked_hidden = hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **kwargs,
    ):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # BERT cần token_type_ids; RoBERTa/DistilBERT thường không có.
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        return self.regression_head(pooled)

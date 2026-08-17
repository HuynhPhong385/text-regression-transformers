# Training Knowledge

## Default starting configuration

```text
epochs: 3
learning_rate: 2e-5
batch_size: 16
max_length: 256
weight_decay: 0.01
warmup_ratio: 0.1
seed: 42
```

Đây là starting point, không phải optimal configuration.

## Optimizer

AdamW.

## Loss

MVP:

```text
MSELoss
```

Extension:

```text
HuberLoss
```

## Fine-tuning

Transformer + regression head trainable.

## Frozen

Transformer frozen, chỉ regression head trainable.

## Checkpoint

Phải lưu:
- weights;
- tokenizer;
- config;
- training state;
- best validation metric.

## Reproducibility

Set seed:
- Python;
- NumPy;
- PyTorch.

## Resource-aware strategy

Nếu GPU yếu:

```text
DistilBERT
max_length=128
small batch
gradient accumulation
mixed precision nếu hỗ trợ
```

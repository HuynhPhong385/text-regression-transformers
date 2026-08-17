# Training

## Starting config

```text
epochs=3
learning_rate=2e-5
batch_size=16
max_length=256
weight_decay=0.01
warmup_ratio=0.1
seed=42
```

## Optimizer

AdamW.

## Loss

MSELoss MVP.

HuberLoss extension.

## Checkpoint

Lưu:
- weights;
- tokenizer;
- config;
- optimizer/scheduler state;
- epoch;
- best validation metric.

## Hardware

Nếu CUDA có sẵn, sử dụng mixed precision khi phù hợp.

CPU fallback bắt buộc cho smoke tests.

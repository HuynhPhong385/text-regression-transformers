# Model

## Architecture

```text
Transformer Encoder
 ↓
First-token representation
 ↓
Dropout
 ↓
Linear
 ↓
Sigmoid
 ↓
Score [0,1]
```

## Models

```text
bert-base-uncased
roberta-base
distilbert-base-uncased
```

## Strategies

### Frozen

Encoder frozen.

### Fine-tuning

Encoder trainable.

## Abstraction

Training loop dùng common interface.

Model-specific differences nằm trong model factory/adapter.

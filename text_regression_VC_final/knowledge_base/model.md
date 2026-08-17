# Model Knowledge

## Common architecture

```text
Transformer Encoder
       ↓
First-token representation
       ↓
Dropout
       ↓
Linear(hidden_size, 1)
       ↓
Sigmoid
       ↓
Score [0,1]
```

## BERT

```text
bert-base-uncased
```

## RoBERTa

```text
roberta-base
```

## DistilBERT

```text
distilbert-base-uncased
```

## Strategy

### Frozen encoder

```text
Encoder: frozen
Head: trainable
```

### Fine-tuning

```text
Encoder: trainable
Head: trainable
```

## Interface

Model factory phải cho phép:

```text
model_name
strategy
dropout
```

Training loop không được phụ thuộc vào một model cụ thể.

## Extension

- mean pooling;
- different regression heads;
- Huber loss.

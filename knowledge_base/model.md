# Member 3 — Model

## Nhiệm vụ

```text
Baseline
   ↓
BERT
   ↓
Frozen Encoder
   ↓
Fine-tuning
   ↓
RoBERTa
   ↓
DistilBERT
```

## Implementation

| Thành phần | Implementation |
|---|---|
| Baseline | E01 Mean Baseline; E02 TF-IDF + Linear Regression |
| BERT | E03/E04 — `bert-base-uncased` |
| Frozen Encoder | E03 — freeze encoder, train regression head |
| Fine-tuning | E04 — train encoder + regression head |
| RoBERTa | E05 — `roberta-base` |
| DistilBERT | E06 — `distilbert-base-uncased` |

## Kiến trúc Transformer

```text
input text → tokenizer → transformer encoder
                     ↓
              masked mean pooling
                     ↓
 Linear → GELU → Dropout → Linear(1)
                     ↓
              float prediction
```

`src/models/base.py` định nghĩa `BaseSklearnModel` và `BaseTorchModel`.
`src/models/model_factory.py` đăng ký E01-E06 và checkpoint tương ứng.

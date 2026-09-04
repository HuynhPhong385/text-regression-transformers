# MODEL.md — Member 3

## Nhiệm vụ

Member 3 implement:

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

## Mapping

| Stage | Implementation |
|---|---|
| Baseline | E01 Mean Baseline; E02 TF-IDF + Linear Regression bổ sung |
| BERT | E03/E04, checkpoint `bert-base-uncased` |
| Frozen Encoder | E03, `freeze_encoder=True` |
| Fine-tuning | E04, `freeze_encoder=False` |
| RoBERTa | E05, checkpoint `roberta-base` |
| DistilBERT | E06, checkpoint `distilbert-base-uncased` |

## Kiến trúc E03-E06

```text
input text
    → tokenizer
    → transformer encoder
    → mean pooling (masked)
    → RegressionHead
       Linear → GELU → Dropout → Linear(1)
    → float prediction
```

- Dùng masked mean pooling thay vì chỉ lấy `[CLS]`.
- RegressionHead mặc định: 128 hidden units.
- E03: freeze encoder.
- E04-E06: fine-tune encoder.

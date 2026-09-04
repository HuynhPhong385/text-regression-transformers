# Member 3 — Model + Training

## Nhiệm vụ

Member 3 phụ trách **Model + Training**, triển khai chuỗi:

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

## Files bắt buộc theo yêu cầu

```text
knowledge_base/
├── model.md
└── training_info.md

docs/
├── MODEL.md
└── TRAINING.md
```

## Cấu trúc code chính

```text
src/models/
├── base.py
├── transformer.py
├── model_factory.py
└── regression_head.py

src/training/
├── trainer.py
├── train.py
├── losses.py
└── checkpoint.py
```

Các file hỗ trợ `baseline.py`, `experiment.py`, `src/data/` và `src/evaluation/` được giữ để triển khai đầy đủ pipeline E01-E06; chúng không thay thế các file bắt buộc ở trên.

## Mapping implementation

- **Baseline**: E01 Mean Baseline; E02 TF-IDF + Linear Regression là baseline học được bổ sung.
- **BERT**: E03/E04 dùng `bert-base-uncased`.
- **Frozen Encoder**: E03 đóng băng encoder, chỉ train regression head.
- **Fine-tuning**: E04 fine-tune BERT encoder + regression head.
- **RoBERTa**: E05, checkpoint `roberta-base`.
- **DistilBERT**: E06, checkpoint `distilbert-base-uncased`.

## Chạy

```bash
pip install -r requirements.txt
python src/training/train.py --data data/dataset.csv --model baseline
python src/training/train.py --data data/dataset.csv --model bert_frozen
python src/training/train.py --data data/dataset.csv --model bert
python src/training/train.py --data data/dataset.csv --model roberta
python src/training/train.py --data data/dataset.csv --model distilbert
```

Chạy toàn bộ:

```bash
python src/training/train.py --data data/dataset.csv --model all
```

# TRAINING.md — Member 3

## Entry point

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

Tuỳ chỉnh Transformer:

```bash
python src/training/train.py --data data/dataset.csv --model roberta \
  --epochs 10 --batch-size 8 --lr 3e-5 --max-length 256
```

## Quy trình

1. Load CSV và validate `text`/`target`.
2. Làm sạch text.
3. Chia train/val/test (80/10/10 khi dataset đủ lớn; fallback an toàn khi quá nhỏ).
4. Tokenize theo checkpoint.
5. Train bằng `src/training/trainer.py`.
6. Lưu `last.pt` và `best.pt` cho Transformer.
7. Early stopping theo validation loss.
8. Reload `best.pt` và đánh giá test.
9. Ghi `result.json`, predictions và training log.

## Outputs

```text
outputs/
├── E01/{model.joblib,result.json}
├── E02/{model.joblib,result.json}
├── E03/{best.pt,last.pt,result.json}
├── E04/{best.pt,last.pt,result.json}
├── E05/{best.pt,last.pt,result.json}
└── E06/{best.pt,last.pt,result.json}
```

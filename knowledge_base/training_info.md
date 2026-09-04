# Member 3 — Training

## Files theo yêu cầu

```text
knowledge_base/model.md
knowledge_base/training_info.md

docs/MODEL.md
docs/TRAINING.md
```

## Entry point

```bash
python src/training/train.py --data data/dataset.csv --model <model>
```

## Model values

- `baseline` / `mean` → E01
- `tfidf` / `linear` → E02
- `bert_frozen` → E03
- `bert` → E04
- `roberta` → E05
- `distilbert` → E06
- `all` → E01-E06

## Training Transformer

1. Load/validate CSV có `text`, `target`.
2. Clean text và chia train/validation/test.
3. Tokenize theo checkpoint.
4. Train theo epoch, tính loss/metrics và validation.
5. Lưu `last.pt` mỗi epoch và `best.pt` khi validation cải thiện.
6. Early stopping theo `patience`.
7. Reload `best.pt` để test.
8. Ghi kết quả vào `outputs/<experiment_id>/`.

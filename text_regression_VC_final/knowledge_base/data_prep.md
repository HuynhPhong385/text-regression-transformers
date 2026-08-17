# Data Preparation Knowledge

## Goal

Chuẩn hóa dataset thành format mà model training pipeline có thể dùng.

## Canonical schema

```text
id
text
rating
target
split
```

## Target

Rating 1–5:

```python
target = (rating - 1) / 4
```

## Validation

```text
text not null
target not null
0 <= target <= 1
```

## Cleaning

- remove missing text;
- remove invalid target;
- deduplicate;
- normalize whitespace.

Không áp dụng aggressive NLP preprocessing.

## Split

Nếu dataset có official split, ưu tiên official split.

Nếu tự split:

```text
80% train
10% validation
10% test
```

## Leakage rules

Không:
- fit preprocessing trên test;
- chọn hyperparameter bằng test;
- duplicate sample giữa splits.

## Tokenization

Default:

```text
max_length=256
truncation=true
dynamic padding
```

## Output

```text
data/processed/
```

# Data Pipeline

```text
Raw
 ↓
Load
 ↓
Validate
 ↓
Clean
 ↓
Target conversion
 ↓
Split
 ↓
Tokenize
 ↓
DataLoader
```

## Default tokenization

```text
max_length=256
truncation=true
dynamic padding
```

## Long text

MVP truncate.

Extension:
- chunking;
- sliding window;
- aggregation.

## Reproducibility

Seed phải được lưu cùng experiment.

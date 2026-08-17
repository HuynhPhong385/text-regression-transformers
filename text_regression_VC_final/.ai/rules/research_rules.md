# Research Rules

## R01 — No fabricated results

Mọi metric phải đến từ run thực tế.

## R02 — Fair comparison

Không so sánh hai model bằng các split khác nhau mà không ghi rõ.

## R03 — Test isolation

Test set chỉ dùng cho final evaluation.

## R04 — Reproducibility

Mỗi experiment có seed và config.

## R05 — Honest interpretation

Không suy diễn:

```text
high R² → model hiểu cảm xúc con người
```

Nên viết:

```text
model captures statistical relationships between text representations and target scores in this dataset.
```

## R06 — Dataset limitations

Rating có thể không phản ánh hoàn toàn sentiment intensity.

Phải ghi limitation này trong report.

# Dataset

## Recommendation

MVP: IMDb.

Extension: Yelp Reviews.

## Canonical schema

| Column | Type | Meaning |
|---|---|---|
| id | string | unique identifier |
| text | string | review |
| rating | float | original rating |
| target | float | normalized score |
| split | string | train/validation/test |

## Target

For rating 1–5:

```text
target = (rating - 1) / 4
```

## Caveat

Nếu dataset chỉ có binary sentiment, không nên mô tả output là intensity liên tục nếu methodology không hỗ trợ điều đó.

## Data leakage

Không:
- fit preprocessing trên test;
- tune trên test;
- duplicate giữa splits.

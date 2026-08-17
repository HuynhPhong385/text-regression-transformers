# Evaluation Knowledge

## Primary metric

MAE.

## Secondary metrics

- MSE
- RMSE
- R²

## Resource metrics

- train time;
- inference latency;
- peak memory;
- parameter count.

## Model-level test

```text
Dataset → Model → Prediction → Metrics
```

## Full-pipeline test

```text
Input → preprocessing → model → postprocessing → API/UI
```

## Required visualizations

- loss curve;
- actual vs predicted;
- residual distribution;
- error buckets;
- model comparison;
- accuracy/resource trade-off.

## Reporting

Không dùng số liệu placeholder trong final report.

Nếu experiment chưa chạy:

```text
TBD
```

hoặc không đưa vào bảng kết quả cuối.

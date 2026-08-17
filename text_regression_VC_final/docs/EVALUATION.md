# Evaluation

## Primary

MAE.

## Secondary

MSE, RMSE, R².

## Performance

- training time;
- inference time;
- peak memory;
- parameter count.

## Model-level

```text
Dataset → Model → Prediction → Metrics
```

## Full-pipeline

```text
User input
→ validation
→ tokenizer
→ model
→ postprocessing
→ API/UI
→ result
```

## Required plots

- loss;
- prediction vs target;
- residual;
- error bucket;
- model comparison;
- resource/performance trade-off.

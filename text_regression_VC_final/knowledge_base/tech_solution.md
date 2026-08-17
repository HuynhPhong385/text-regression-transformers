# Technical Solution

## Architecture

```text
Data Layer
    ↓
Training Layer
    ↓
Model Layer
    ↓
Evaluation Layer
    ↓
Inference Layer
    ↓
API/UI
```

## Data

Recommended MVP: IMDb.

Extension: Yelp Reviews.

## Model

Transformer encoder + regression head.

## Training

AdamW + learning-rate scheduler.

## Evaluation

MAE primary metric.

Secondary:
- MSE
- RMSE
- R²

Resource:
- training time
- inference time
- memory
- parameters

## Serving

FastAPI.

## Demo

Streamlit.

## Configuration

YAML configuration.

Không hard-code hyperparameters trong training source.

## Reproducibility

Mỗi experiment lưu:
- seed;
- model;
- dataset;
- config;
- metrics;
- runtime;
- checkpoint metadata.

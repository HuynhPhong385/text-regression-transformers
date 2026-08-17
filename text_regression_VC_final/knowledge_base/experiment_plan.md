# Experiment Plan

## E00 — Sanity check

Dataset validity and target distribution.

## E01 — Mean baseline

Establish naive lower baseline.

## E02 — TF-IDF + Linear Regression

Traditional ML baseline.

## E03 — BERT frozen

Train regression head only.

## E04 — BERT fine-tuning

Train complete model.

## E05 — RoBERTa fine-tuning

Compare architecture.

## E06 — DistilBERT fine-tuning

Measure efficiency.

## E07 — Resource comparison

Measure:
- runtime;
- memory;
- parameters;
- latency.

## E08 — Optional ablation

- max length;
- pooling;
- loss.

## Fairness

Same:
- dataset;
- target;
- split;
- primary metric.

Test set is not used for tuning.

## Required artifact

Mỗi experiment:

```text
config.yaml
metrics.json
training_history.csv
predictions.csv
model_metadata.json
```

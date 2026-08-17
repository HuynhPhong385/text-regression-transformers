# Experiments

## E00

Dataset sanity check.

## E01

Mean baseline.

## E02

TF-IDF + Linear Regression.

## E03

BERT frozen.

## E04

BERT fine-tuning.

## E05

RoBERTa fine-tuning.

## E06

DistilBERT fine-tuning.

## E07

Resource comparison.

## E08

Optional ablation.

## Experiment record

Every run stores:

```text
experiment_name
config
seed
dataset
model
strategy
metrics
runtime
hardware
checkpoint
```

## Fair comparison

Same data split and target definition.

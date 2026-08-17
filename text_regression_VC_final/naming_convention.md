# Naming Convention

## Python files

Dùng `snake_case`.

```text
data_loader.py
model_factory.py
train.py
evaluate.py
error_analysis.py
```

## Classes

PascalCase:

```python
TextRegressionModel
ExperimentRunner
IMDbDataset
```

## Functions

snake_case:

```python
load_dataset()
prepare_target()
train_one_epoch()
evaluate_model()
```

## Variables

snake_case:

```python
learning_rate
max_length
train_loader
validation_loss
```

## Constants

UPPER_SNAKE_CASE:

```python
DEFAULT_SEED
DEFAULT_MAX_LENGTH
```

## Experiment names

```text
{model}_{strategy}_{dataset}_seed{seed}
```

Ví dụ:

```text
bert_finetune_imdb_seed42
bert_frozen_imdb_seed42
roberta_finetune_imdb_seed42
distilbert_finetune_imdb_seed42
```

## Artifact names

```text
metrics.json
predictions.csv
training_history.csv
config.yaml
model_metadata.json
```

Không dùng:

```text
final.py
final2.py
new_final.py
best_final_really_final.py
```

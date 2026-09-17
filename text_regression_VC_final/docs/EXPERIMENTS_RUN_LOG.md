# docs/EXPERIMENTS — Run log (actual results, no fake metrics)

## E01 — Mean baseline (from TF baseline run)

- train_mean = 0.501 (rating ~3.0)
- MAE 0.500, included as `mean_baseline_mae` in baseline metrics.

## E02 — TF-IDF + Linear Regression

- MAE 0.251, MSE 0.131, RMSE 0.362, R2 0.475
- train_time 19s
- Artifact: artifacts/baseline_imdb_seed42/

## E03 — BERT frozen

- MAE 0.455, MSE 0.212, RMSE 0.460, R2 0.154
- Underfits — only 769 trainable params (regression head)
- train_time 326s
- Artifact: artifacts/bert_frozen_imdb_seed42/

## E04 — BERT finetune (best)

- MAE 0.088, MSE 0.061, RMSE 0.246, R2 0.757
- train_time 1030s, infer ~228s for 24.6k test samples
- total_params 109.5M, trainable 109.5M
- Artifact: artifacts/bert_finetune_imdb_seed42/
- Buckets (abs error): 0.00-0.05:20384, 0.05-0.10:864, 0.10-0.20:677, 0.20-0.30:400, >0.30:2353

## E06 — DistilBERT finetune

- MAE 0.100, MSE 0.067, RMSE 0.258, R2 0.733
- train_time 567s (45% faster than BERT), infer 121s
- total_params 66.4M
- Artifact: artifacts/distilbert_finetune_imdb_seed42/

## E05 — RoBERTa finetune

- Status: RUNNING (started ~14:48, ~16 min elapsed). Will update when done.

## E07 — Resource comparison

- summary.csv + tradeoff plots in artifacts/ (tradeoff_time_vs_mae.png, tradeoff_params_vs_mae.png, comparison_mae.png)
- Frozen << Finetune in accuracy but also much faster per epoch; DistilBERT is best time/accuracy tradeoff.

## E08 — Optional ablation

- Not run (out of scope for MVP). Candidates: max_length 128 vs 256, pooling, HuberLoss.

## Fairness

- Same splits: train 22418 / val 2485 / test 24678 (IMDb official + 10% val from train, seed 42, deduped)
- Same target: (rating-1)/4
- Same primary metric: MAE

## Required artifacts per experiment

- config.yaml, metrics.json, training_history.csv, predictions.csv, model_metadata.json — all present for completed runs.
- Plots: loss.png, error_buckets.png, prediction_vs_target.png, residual.png per artifact + summary-level comparison/tradeoff.

## Research questions (preliminary, will finalize after RoBERTa)

- RQ1 (Transformer vs TF-IDF): YES — BERT finetune MAE 0.088 << 0.251
- RQ2 (Finetune vs Frozen): YES — 0.088 << 0.455
- RQ3 (RoBERTa vs BERT): pending
- RQ4 (DistilBERT tradeoff): YES — 13% worse MAE than BERT, 45% faster, 40% fewer params
- RQ5 (AI Agent workflow): Engineering workflow enabled audit + reproducibility (seed, config-driven, checkpoint metadata)

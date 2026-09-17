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

## E04 — BERT finetune

- MAE 0.088, MSE 0.061, RMSE 0.246, R2 0.757
- train_time 1030s, infer 228s (24.6k test), 109.5M params
- Artifact: artifacts/bert_finetune_imdb_seed42/
- Buckets: 0.00-0.05:20384, 0.05-0.10:864, 0.10-0.20:677, 0.20-0.30:400, >0.30:2353

## E06 — DistilBERT finetune

- MAE 0.100, MSE 0.067, RMSE 0.258, R2 0.733
- train_time 567s (45% faster than BERT), infer 121s, 66.4M params
- Artifact: artifacts/distilbert_finetune_imdb_seed42/
- Buckets: 0.00-0.05:19689, 0.05-0.10:1006, 0.10-0.20:853, 0.20-0.30:444, >0.30:2686

## E05 — RoBERTa finetune (BEST)

- MAE 0.070, MSE 0.046, RMSE 0.214, R2 0.816
- train_time 1065s, infer 227s, 124.6M params
- Artifact: artifacts/roberta_finetune_imdb_seed42/
- Buckets: 0.00-0.05:21342, 0.05-0.10:626, 0.10-0.20:531, 0.20-0.30:308, >0.30:1871

## E07 — Resource comparison

- artifacts/summary.csv + tradeoff plots:
  - artifacts/comparison_mae.png, tradeoff_time_vs_mae.png, tradeoff_params_vs_mae.png
  - loss.png per artifact: bert_frozen, bert_finetune, distilbert, roberta
  - error_buckets.png, prediction_vs_target.png, residual.png per artifact

## E08 — Optional ablation

- Not run (out of scope for MVP). Candidates: max_length 128 vs 256, pooling, HuberLoss.

## Fairness

- Same splits: train 22418 / val 2485 / test 24678 (IMDb official + 10% val from train, seed 42, deduped, leakage PASS)
- Same target: (rating-1)/4 -> [0, 1]
- Same primary metric: MAE (ADR-003)

## Required artifacts per experiment

- config.yaml, metrics.json, training_history.csv, predictions.csv, model_metadata.json — all present.
- Plots: loss.png, error_buckets.png, prediction_vs_target.png, residual.png per artifact + summary-level comparison/tradeoff.
- Top-20: bert_finetune_top_*.csv, roberta_top_*.csv in artifacts/ (best models).

## Comparison table

| Experiment | Model | Strategy | MAE | MSE | RMSE | R2 | Train (s) | Params |
|---|---|---|---|---|---|---|---|---|
| baseline | tfidf-lr | — | 0.251 | 0.131 | 0.362 | 0.475 | 19 | — |
| bert_frozen | bert-base-uncased | frozen | 0.455 | 0.212 | 0.460 | 0.154 | 326 | 109.5M (769 trainable) |
| bert_finetune | bert-base-uncased | finetune | 0.088 | 0.061 | 0.246 | 0.757 | 1030 | 109.5M |
| distilbert | distilbert-base-uncased | finetune | 0.100 | 0.067 | 0.258 | 0.733 | 567 | 66.4M |
| roberta | roberta-base | finetune | 0.070 | 0.046 | 0.214 | 0.816 | 1065 | 124.6M |

## Research questions

- RQ1 (Transformer vs TF-IDF): YES — RoBERTa/BERT finetune (0.07/0.09) << TF-IDF 0.25. Frozen is worse (0.455) — shows finetuning is essential.
- RQ2 (Finetune vs Frozen): YES — BERT finetune 0.088 << frozen 0.455 (5x better).
- RQ3 (RoBERTa vs BERT): YES — RoBERTa 0.070 < BERT 0.088 (20% better), best R2 0.816.
- RQ4 (DistilBERT tradeoff): YES — 13% worse than BERT, 45% faster, 40% fewer params, 47% faster inference.
- RQ5 (AI Agent workflow): Workflow enabled audit (32-item checklist), reproducibility (seed 42, config-driven, checkpoint metadata with 6 fields), no fake metrics.

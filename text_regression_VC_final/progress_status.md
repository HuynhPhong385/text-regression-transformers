# Progress Status

> Đây là file trạng thái chính. AI Agent phải đọc trước khi bắt đầu task và cập nhật sau khi hoàn thành task.

## Overall

Status: `DONE`

Progress: `100%`

Current phase: `Phase 10 — Finalization`

## Phase 0 — Business Understanding

- [x] User requirements
- [x] Features
- [x] Tech solution
- [x] AI logic
- [x] Acceptance criteria

## Phase 1 — Bootstrap

- [x] Repository structure
- [x] Python environment
- [x] Dependencies
- [x] Config system
- [x] Test framework

## Phase 2 — Data Preparation

- [x] Dataset selected
- [x] Dataset loaded
- [x] Schema validated
- [x] Target created
- [x] Train/validation/test split
- [x] Data pipeline tested

## Phase 3 — Baseline

- [x] Mean predictor
- [x] TF-IDF + Linear Regression
- [x] Baseline evaluation

## Phase 4 — BERT MVP

- [x] Tokenizer
- [x] Dataset class
- [x] Regression model
- [x] Training loop
- [x] Validation
- [x] Checkpoint
- [x] Test evaluation
- [x] Inference

## Phase 5 — Strategy comparison

- [x] BERT frozen
- [x] BERT fine-tuning
- [x] Compare metrics
- [x] Compare time/resources

## Phase 6 — Model comparison

- [x] RoBERTa
- [x] DistilBERT
- [x] Unified model interface
- [x] Comparison table

## Phase 7 — Model-level test
> Owner: Member 4 (Evaluation + Experiments + Error Analysis)

- [x] MAE
- [x] MSE
- [x] RMSE
- [x] R²
- [x] Training time
- [x] Inference time
- [x] Parameter count
- [x] Memory

## Phase 8 — Full pipeline

- [x] Data → training
- [x] Checkpoint → inference
- [x] API
- [x] UI
- [x] End-to-end test

## Phase 9 — Analysis
> Owner: Member 4 (Evaluation + Experiments + Error Analysis)

- [x] Error analysis
- [x] Plots
- [x] Resource trade-off
- [x] Research questions answered

## Phase 10 — Finalization

- [x] Codebase audit
- [x] Documentation audit
- [x] Reproducibility audit
- [x] Final report
- [x] Final checklist

## Current task

```text
DONE — All phases verified.
- E02 TF-IDF MAE 0.251 | E03 BERT frozen 0.455 | E04 BERT finetune 0.088 | E06 DistilBERT 0.100 | E05 RoBERTa 0.070 (best)
- 17 tests PASS, artifacts/summary.csv + plots verified
- docs/EXPERIMENTS_RUN_LOG.md is the final comparison record
- Next: open PR feature/evaluation -> main, request review
```

## Update rule

Không đánh dấu `[x]` nếu chưa được kiểm chứng.

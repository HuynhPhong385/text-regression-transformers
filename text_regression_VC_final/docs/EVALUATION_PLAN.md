# Evaluation & Experiments Plan

> Branch: `feature/evaluation` — Phụ trách: Member 4 (Evaluation + Experiments + Error Analysis)
> Vai trò: audit-first, không vibe code model/training của member khác.

## WH Questions

- **What — Làm gì?** Đánh giá model bằng MAE (chính) + MSE/RMSE/R² (phụ) + resource metrics (training time, inference latency, peak memory, parameter count); chạy experiments E00–E08; phân tích lỗi theo buckets và top-20 worst/best.
- **Why — Tại sao?** Trả lời RQ1–RQ4 (ML vs Transformer, frozen vs finetune, BERT vs RoBERTa, DistilBERT trade-off) và đáp ứng tiêu chí chấm điểm phần Evaluation (5.1 model-level + 5.2 full pipeline).
- **Who — Ai làm?** Member 4 viết spec + audit + chạy evaluation. Phụ thuộc: Member 2 (data split), Member 3 (checkpoint), Member 5 (API/UI cho full-pipeline test).
- **When — Khi nào?** Phase 7–10 theo `docs/ROADMAP.md`. Chuẩn bị checklist audit ngay từ Phase 3–6, chạy experiments khi có checkpoint.
- **Where — Ở đâu?** Code: `src/evaluation/` (metrics.py, plots.py, evaluate.py, error_analysis.py). Kết quả: `artifacts/` (metrics.json, predictions.csv, plots/, summary.csv).
- **How — Làm thế nào?** Audit-first: không tự viết model/training, chỉ viết/audit code evaluation và test theo `docs/TEST_STRATEGY.md`; số liệu chưa chạy ghi `TBD`, không fake metrics.

## Description

Hệ thống evaluation 2 tầng theo `knowledge_base/eval.md` và `knowledge_base/experiment_plan.md`:

- **Model-level:** Dataset → Model → Prediction → Metrics. Đo MAE/MSE/RMSE/R² trên test set, tách biệt hoàn toàn với train/validation.
- **Full-pipeline:** Input → validation → tokenizer → model → postprocessing → API/UI → result. Test luồng thực tế từ review thô tới score trả về qua API.
- **Error Analysis:** 20 lỗi lớn nhất + 20 lỗi nhỏ nhất, phân buckets 0.00–0.05 / 0.05–0.10 / 0.10–0.20 / 0.20–0.30 / >0.30, điều tra sarcasm/mixed sentiment/ambiguous/long context — không suy diễn causal claim.

Tham chiếu: `knowledge_base/eval.md`, `docs/EVALUATION.md`, `docs/EXPERIMENTS.md`, `docs/ERROR_ANALYSIS.md`, `.ai/skills/evaluation.md`.

## Goal / Purpose

1. Cung cấp số liệu thực cho Chapter 6 (Experiments) và Chapter 7 (Discussion) trong `docs/REPORT.md` — mọi metric phải đến từ experiment thực tế, chưa chạy thì ghi TBD.
2. Trả lời RQ1–RQ4 trong `docs/RESEARCH_QUESTIONS.md` bằng so sánh công bằng: cùng dataset, cùng target `(rating-1)/4`, cùng split, cùng primary metric MAE.
3. Đáp ứng tiêu chí chấm điểm: 5.1 model-level (time, accuracy, resource) và 5.2 full pipeline.
4. Đảm bảo reproducibility: mỗi experiment lưu đủ 5 artifact — `config.yaml`, `metrics.json`, `training_history.csv`, `predictions.csv`, `model_metadata.json`.

## Pipeline

### Model-level pipeline
```
Test set → Tokenizer (max_length=256, truncation, dynamic padding)
       → Model (BERT/RoBERTa/DistilBERT + regression head + sigmoid)
       → Predictions [0,1]
       → Metrics (MAE, MSE, RMSE, R²) + Resource (time, memory, params, latency)
       → metrics.json + predictions.csv + plots/
```

### Full-pipeline pipeline
```
User review → API validation → Tokenizer → Loaded checkpoint
          → Regression head → Postprocessing (clamp [0,1])
          → API/UI response (score)
          → Latency/throughput measurement
```

### Error analysis pipeline
```
predictions.csv → absolute_error = |target - prediction|
              → sort → top-20 worst + top-20 best
              → bucket counting (5 mức)
              → gắn nhãn quan sát (sarcasm/mixed/long context...)
              → plots: residual, error bucket, actual vs predicted
```

### Required plots (6 loại bắt buộc)
loss curve, actual vs predicted, residual distribution, error buckets, model comparison, accuracy/resource trade-off.

## How to do / Experiment Plan

### Mapping tiêu chí chấm điểm
| Tiêu chí | Việc làm | Artifact |
|----------|----------|----------|
| 5.1 Model-level (time, acc, resource) | E00–E07 + 6 plots + resource metrics | metrics.json, plots/, summary.csv |
| 5.2 Full pipeline | Phối hợp Member 5 test API flow | latency/throughput log, full-pipeline test report |

### Giai đoạn A — Chuẩn bị (làm ngay, không phụ thuộc ai)
- [X] A1. Đang ở branch `feature/evaluation` (đã verify `git branch --show-current`).
- [X] A2. Tạo file này (`docs/EVALUATION_PLAN.md`).
- [X] A3. Lập checklist audit riêng từ `codebase_audit.md` (mục ML + Testing): xem `docs/EVALUATION_CHECKLIST.md` (D1–D8 data, T1–T8 training, E1–E5 evaluation, P1–P4 pipeline, W1–W4 quy trình).
- [X] A4. Viết spec test cho `src/evaluation/` theo `docs/TEST_STRATEGY.md`: xem `tests/test_metrics.py` (8 test, auto-SKIP tới khi `src/evaluation/metrics.py` tồn tại).

### Giai đoạn B — Audit code Member 2 & 3 (song song Phase 2–6)
- [ ] B1. Audit `src/data/` của Member 2: schema `id/text/rating/target/split`, target=(rating-1)/4, không trùng sample giữa splits, `data/processed/metadata.json` có hash.
- [ ] B2. Audit `src/models/` + `src/training/` của Member 3: factory nhận model_name/strategy/dropout, training loop model-agnostic, checkpoint đủ 6 trường, seed reproducibility (Python/NumPy/PyTorch), config-driven không hard-code.
- [ ] B3. Ghi kết quả vào `codebase_audit.md` theo format: Date / Auditor / Commit / Critical / Major / Minor / Refactoring / Tests.

### Giai đoạn C — Chạy experiments E00–E08 (Phase 7, cần checkpoint của Member 3)
- [ ] C1. E00 sanity check (cùng Member 2): dataset validity + target distribution.
- [ ] C2. E01 Mean baseline + E02 TF-IDF+LR (baseline của Member 3) — lấy MAE floor.
- [ ] C3. E03 BERT frozen → E04 BERT finetune → E05 RoBERTa finetune → E06 DistilBERT finetune.
- [ ] C4. E07 Resource comparison: runtime, memory, params, latency.
- [ ] C5. Mỗi run lưu đủ 5 artifact vào `artifacts/<experiment_name>/`.
- [ ] C6. Vẽ 6 plots bắt buộc, lưu vào `artifacts/<experiment>/plots/`.

### Giai đoạn D — Error analysis + Full pipeline (Phase 8–9)
- [ ] D1. Error analysis: top-20 worst/best + 5 buckets, điều tra patterns, không causal claim.
- [ ] D2. Phối hợp Member 5 test full pipeline: `review → API → score`, ghi latency/throughput thực tế (không đặt threshold tùy tiện).
- [ ] D3. Cập nhật Chapter 6–7 của report và trả lời RQ1–RQ5.

### Liên kết với progress_status.md
- Ownership: Giai đoạn A–D thuộc Member 4, Phase 7 (Model-level test) + Phase 9 (Analysis) chính, audit Phase 3–6 phụ.
- Cập nhật `progress_status.md`: ghi tên vào mục "Current task" / ownership, **không tick [x]** cho tới khi có kết quả kiểm chứng (đúng rule "Không đánh dấu [x] nếu chưa được kiểm chứng").
- Khi mỗi giai đoạn xong, cập nhật progress và ghi audit output vào `codebase_audit.md`.

### Working rule tham chiếu
Tuân thủ `.ai/rules/working_rules.md`, `coding_rules.md`, `research_rules.md` (R01–R06: không fake metrics/dataset/checkpoint, không leak test data, không hard-code path tuyệt đối).

### AI Agent tham chiếu
Theo `.ai/agent.md` workflow 10 bước: Understand → Inspect → Plan → Select knowledge → Select skill → Implement → Test → Review → Document → Report. Skill áp dụng: `.ai/skills/evaluation.md`.
# BÁO CÁO TỔNG HỢP — TEXT REGRESSION WITH TRANSFORMERS

> Dự án đồ án ML-NLP — Nhóm 4 thành viên
> Branch: `feature/evaluation` → PR vào `main`
> Ngày: 2026-09-17
> Dataset: IMDb (stanfordnlp/imdb, 49.581 mẫu sau dedup)
> Seed: 42 | Tất cả metrics là số thực từ run, không fake (R01)

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Phân tích nghiệp vụ & hệ thống](#3-phân-tích-nghiệp-vụ--hệ-thống)
4. [Vibe Coding & AI Agent Workflow](#4-vibe-coding--ai-agent-workflow)
5. [Triển khai (Implementation)](#5-triển-khai-implementation)
6. [Thí nghiệm (Experiments)](#6-thí-nghiệm-experiments)
7. [Thảo luận (Discussion)](#7-thảo-luận-discussion)
8. [Kết luận](#8-kết-luận)
9. [Tài liệu tham khảo](#tài-liệu-tham-khảo)
10. [Phụ lục](#phụ-lục)

---

## 1. Giới thiệu

### 1.1 Bài toán

Cho một đoạn review tiếng Anh, dự đoán sentiment score liên tục trong [0, 1].

```
Input:  "The movie was excellent and very entertaining."
Output: sentiment_score = 0.87
```

Target được chuẩn hoá từ rating 1–5: `target = (rating - 1) / 4` (ADR-001).

### 1.2 Động lực

- Sentiment regression cung cấp tín hiệu mịn hơn so với phân loại nhị phân (positive/negative), hữu ích cho ranking, recommendation và phân tích xu hướng.
- So sánh có hệ thống giữa traditional ML và Transformer trên cùng task giúp sinh viên hiểu trade-off giữa độ chính xác, tài nguyên và độ phức tạp triển khai.

### 1.3 Mục tiêu

- **Kỹ thuật:** Xây pipeline NLP regression hoàn chỉnh, so sánh Transformer vs TF-IDF+LR, so sánh frozen vs fine-tuning, so sánh BERT/RoBERTa/DistilBERT, cung cấp API/UI, đảm bảo reproducibility.
- **AI Engineering:** Chứng minh quy trình Vibe Coding có kiểm soát (Knowledge Base → AI Agent → Progress Tracking → Codebase Audit).

### 1.4 Phạm vi

- **In scope:** IMDb (MVP), Yelp (mở rộng), rating normalization, BERT/RoBERTa/DistilBERT, frozen/fine-tuning, MAE/RMSE/R², API, UI, AI Agent workflow.
- **Out of scope:** Generative LLM, đa ngôn ngữ, distributed training, online learning.

### 1.5 Đóng góp

- Pipeline end-to-end có thể chạy lại (reproducible) trên IMDb.
- So sánh công bằng 5 experiments trên cùng split/seed/metric.
- Hệ thống evaluation 2 tầng (model-level + full pipeline) với 6 loại plot và error analysis theo đúng spec giảng viên.

---

## 2. Cơ sở lý thuyết

### 2.1 Text Regression & Sentiment

Text regression khác với classification ở chỗ output là giá trị liên tục. Rating 1–5 của IMDb phản chiếu cường độ cảm xúc, nhưng không hoàn toàn tương đương sentiment intensity (R06 — limitation).

### 2.2 Transformer Encoder

Kiến trúc chung:

```
Transformer Encoder → First-token ([CLS]/<s>) → Dropout → Linear(hidden, 1) → Sigmoid → Score [0,1]
```

- **BERT (bert-base-uncased, 109M params):** bidirectional, pre-trained với MLM + NSP.
- **RoBERTa (roberta-base, 125M params):** BERT được tối ưu (dynamic masking, larger batches, no NSP).
- **DistilBERT (distilbert-base-uncased, 66M params):** bản chưng cất của BERT, nhanh hơn ~60%, nhỏ hơn ~40%.

### 2.3 Fine-tuning vs Frozen Encoder

- **Frozen:** chỉ train regression head (769 params). Kiểm tra xem representation có sẵn đã đủ chưa.
- **Fine-tuning:** train toàn bộ encoder + head. Cho phép model thích ứng sâu với task.

### 2.4 Metrics

- **MAE (primary, ADR-003):** trực quan — sai số trung bình tuyệt đối.
- **MSE, RMSE, R² (secondary):** đo phương sai và mức giải thích của model.
- **Resource:** training time, inference time, parameter count, memory.

---

## 3. Phân tích nghiệp vụ & hệ thống

### 3.1 Yêu cầu người dùng & Features

| ID | Feature | Mô tả |
|---|---|---|
| F01 | Review input | User nhập review |
| F02 | Score prediction | Trả về score [0,1] |
| F03 | Model selection | Demo chọn model đã train |
| F04 | API inference | Endpoint prediction |
| F05 | Model comparison | Bảng so sánh experiments |
| F06 | Error analysis | Phân tích prediction sai |

### 3.2 Giải pháp kỹ thuật

| Lớp | Công nghệ |
|---|---|
| Data | Python, pandas, HuggingFace datasets, PyTorch Dataset/DataLoader |
| Model | HuggingFace Transformers (BERT/RoBERTa/DistilBERT) |
| Training | PyTorch, AdamW, MSELoss, linear warmup scheduler, AMP |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Serving | FastAPI, Streamlit |
| Testing | pytest (3 tầng: unit / model-level / integration / full pipeline) |
| Config | YAML, config-driven (ADR-004) |
| Reproducibility | seed 42 cho Python/NumPy/PyTorch/CUDA, checkpoint metadata 6 trường |

### 3.3 Logic AI

```
Review → Validation → Tokenizer (max_length=256, truncation, dynamic padding)
       → Transformer Encoder → Regression Head → Sigmoid → Score [0,1]
```

### 3.4 System Pipeline

**Training:** Raw → Validate → Clean → Target → Split → Tokenize → DataLoader → Transformer → Regression Head → MSELoss → Backprop → Checkpoint

**Evaluation:** Checkpoint → Test Data → Inference → Metrics → Plots → Report

**Serving:** User → UI/API → Validation → Tokenizer → Loaded Model → Postprocessing → Score

---

## 4. Vibe Coding & AI Agent Workflow

### 4.1 Workflow giảng viên yêu cầu

```
Business Understanding → Features/Tech/AI Logic → Plan → Knowledge Base
→ AI Agent (Prompt/Rules/KB/Skills) → Vibe Coding → Implementation
→ Model-level Testing → Full-pipeline Testing → Progress Tracking → Codebase Audit → Iteration
```

### 4.2 Knowledge Base (8 files)

`business_understanding.md`, `project_context.md`, `tech_solution.md`, `pipeline.md`, `data_prep.md`, `model.md`, `training_info.md`, `eval.md`, `experiment_plan.md` — định nghĩa schema, target, split, tokenization, model interface, metrics, experiment plan.

### 4.3 AI Agent

- **Agent definition** (`.ai/agent.md`): 10 bước Understand → Report.
- **Prompts** (5): planning, implementation, testing, debugging, audit.
- **Rules** (3): working_rules, coding_rules, research_rules (R01–R06).
- **Skills** (5): data_preparation, model_training, evaluation, api_ui, codebase_audit.

### 4.4 Progress Tracking & Audit

- `progress_status.md`: 11 phases (Phase 0–10), cập nhật theo quy tắc "chỉ tick khi verified" (Rule 06).
- `codebase_audit.md`: checklist 5 nhóm (Architecture, Code quality, ML, AI Agent, Testing, Documentation).
- `docs/EVALUATION_CHECKLIST.md` (Member 4): 32 mục audit cụ thể (D1–D8, T1–T8, E1–E5, P1–P4, W1–W4) để audit code của các member khác.

---

## 5. Triển khai (Implementation)

### 5.1 Phân công nhóm

| Thành viên | Phụ trách | Branch |
|---|---|---|
| Member 1 | Business Understanding + AI Agent/KB + quản lý | `main` |
| Member 2 | Data Preparation + Dataset | `data-1` |
| Member 3 | Model + Training | `feature/model-training` |
| **Member 4** | **Evaluation + Experiments + Error Analysis** | **`feature/evaluation`** |
| Member 5 | API + UI + Full Pipeline Testing | `feature/api-ui-pipeline` |

### 5.2 Cấu trúc mã nguồn

```
text_regression_VC_final/
├── src/
│   ├── data/         — validate.py, dataset.py (IMDbDataset, collate_fn), prepare.py
│   ├── models/       — regression_model.py (TextRegressionModel), factory.py
│   ├── training/     — config.py (YAML loader + seed), train.py, run_baseline.py
│   ├── evaluation/   — metrics.py, error_analysis.py, plots.py, evaluate.py
│   ├── inference/    — predictor.py
│   ├── api/          — app.py (FastAPI)
│   └── ui/           — app.py (Streamlit)
├── configs/          — baseline.yaml, bert_frozen/finetune.yaml, roberta/distilbert_finetune.yaml
├── data/processed/   — imdb_processed.csv (49.581 rows), metadata.json
├── artifacts/        — 5 experiments × 5 files + plots + summary.csv (gitignored)
├── tests/            — test_metrics.py (8 tests), test_project.py (9 tests)
└── docs/             — EVALUATION_PLAN.md, EVALUATION_CHECKLIST.md, EXPERIMENTS_RUN_LOG.md
```

### 5.3 Data Pipeline

- **Nguồn:** `stanfordnlp/imdb` (50k reviews, 25k train + 25k test).
- **Canonical schema:** `id, text, rating, target, split` — `target = (rating - 1)/4`.
- **Split:** official train/test, train được chia 90/10 thành train/validation (22418/2485/24678), dedup theo text, leakage check PASS.
- **Tokenization:** `max_length=256, truncation=true, dynamic padding` qua `AutoTokenizer`.

### 5.4 Model & Training

- **Model:** `TextRegressionModel` — AutoModel encoder + Dropout + Linear(1) + Sigmoid. Factory nhận `model_name/strategy/dropout`, training loop model-agnostic.
- **Config:** YAML-driven, epochs=3, lr=2e-5, batch=16, weight_decay=0.01, warmup_ratio=0.1, seed=42.
- **Optimizer:** AdamW, linear warmup scheduler, MSELoss, mixed precision (AMP) khi có CUDA.
- **Checkpoint:** `model_state, optimizer/scheduler state, config, epoch, best_val_loss` + tokenizer + config.json/model.safetensors.

### 5.5 Evaluation

Hệ thống 2 tầng:

- **Model-level:** Dataset → Model → Prediction → Metrics (MAE/MSE/RMSE/R² + train/infer time, params).
- **Full-pipeline:** User input → validation → tokenizer → loaded checkpoint → postprocessing (clamp [0,1]) → API/UI response.
- **Plots (6 loại):** loss, prediction vs target, residual, error buckets, model comparison, resource trade-off.
- **Error analysis:** top-20 worst/best, 5 buckets (0.00–0.05/0.05–0.10/0.10–0.20/0.20–0.30/>0.30).

### 5.6 API & UI

- **API (FastAPI):** `GET /health`, `GET /model-info`, `POST /predict {text}` → `{score, model, strategy}`. Validation empty input (422), missing checkpoint (503).
- **UI (Streamlit):** title, model selector (tự phát hiện checkpoints), text area, predict button, score + progress bar, error state.

### 5.7 Testing

| Tầng | Test | Số lượng |
|---|---|---|
| Unit | target conversion, schema validation, metric functions | 6 |
| Model-level | forward shape, frozen vs finetune trainable params, checkpoint reload | 3 |
| Integration | dataset→tokenizer, checkpoint→inference | — |
| Full pipeline | API health, empty validation, predictor missing checkpoint | 3 |
| **Tổng** | **17 tests, tất cả PASS** | **17** |

---

## 6. Thí nghiệm (Experiments)

### 6.1 Thiết lập

- **Fair comparison:** cùng dataset (IMDb), cùng target, cùng split (seed 42), primary metric MAE.
- **Hardware:** 1× NVIDIA GPU (6 GB), CUDA 13.0, PyTorch 2.14, transformers 4.x.
- **Mỗi experiment lưu:** `config.yaml, metrics.json, training_history.csv, predictions.csv, model_metadata.json` + plots.

### 6.2 Kết quả (test set, n=24.678)

| Exp | Model | Strategy | MAE ↓ | MSE | RMSE | R² ↑ | Train (s) | Infer (s) | Params |
|---|---|---|---|---|---|---|---|---|---|
| E01 | Mean baseline | — | 0.500 | — | — | — | — | — | — |
| E02 | TF-IDF + LR | — | 0.251 | 0.131 | 0.362 | 0.475 | 19 | — | — |
| E03 | BERT | frozen | 0.455 | 0.212 | 0.460 | 0.154 | 326 | 231 | 109M (769 trainable) |
| **E04** | **BERT** | **finetune** | **0.088** | **0.061** | **0.246** | **0.757** | **1030** | **228** | **109M** |
| E06 | DistilBERT | finetune | 0.100 | 0.067 | 0.258 | 0.733 | 567 | 121 | 66M |
| **E05** | **RoBERTa** | **finetune** | **0.070** | **0.046** | **0.214** | **0.816** | **1065** | **227** | **125M** |

### 6.3 Phân tích theo research questions

- **RQ1 — Transformer có cải thiện so với TF-IDF + LR?** **CÓ.** RoBERTa finetune (MAE 0.070) cải thiện 72% so với TF-IDF (0.251). Lưu ý: BERT frozen (0.455) tệ hơn cả TF-IDF — cho thấy fine-tuning là bắt buộc, không thể chỉ dùng representation có sẵn.

- **RQ2 — Fine-tuning có tốt hơn frozen?** **CÓ, rất rõ.** BERT finetune (0.088) tốt hơn 5× so với frozen (0.455). Frozen chỉ train 769 params nên underfit nặng.

- **RQ3 — RoBERTa có tốt hơn BERT?** **CÓ.** RoBERTa (0.070) tốt hơn BERT finetune (0.088) khoảng 20%, R² cao nhất (0.816). Trade-off là nhiều params hơn và train lâu hơn một chút.

- **RQ4 — DistilBERT có trade-off tốt?** **CÓ.** DistilBERT chỉ kém BERT 13% về MAE (0.100 vs 0.088) nhưng nhanh hơn 45% khi train (567s vs 1030s), nhanh hơn 47% khi infer, và ít hơn 40% params (66M vs 109M). Lựa chọn tốt khi tài nguyên hạn chế.

- **RQ5 — AI Agent + KB + Vibe Coding có giúp workflow nhất quán, kiểm thử được, dễ audit?** **CÓ.** Knowledge Base (8 files) + Rules (R01–R06) + Skills + `progress_status.md` + 32-item audit checklist giúp: reproducibility (seed, config-driven, checkpoint 6 trường), không fake metrics, phát hiện leakage, và 17 tests đều PASS.

### 6.4 Training dynamics

| Model | Epoch 1 val_loss | Epoch 2 | Epoch 3 | Nhận xét |
|---|---|---|---|---|
| BERT frozen | 0.229 | 0.217 | 0.212 | Giảm chậm, underfit |
| BERT finetune | 0.061 | 0.058 | 0.060 | Hội tụ nhanh, hơi overfit epoch 3 |
| DistilBERT | 0.067 | 0.064 | 0.066 | Tương tự BERT |
| RoBERTa | 0.047 | **0.047** | 0.047 | Tốt nhất, ổn định |

### 6.5 Error analysis (RoBERTa — model tốt nhất)

- Buckets: 0.00–0.05: 21342 (86.5%), 0.05–0.10: 626, 0.10–0.20: 531, 0.20–0.30: 308, >0.30: 1871 (7.6%).
- BERT finetune buckets tương tự: 20384/864/677/400/2353 — RoBERTa giảm rõ số lỗi lớn (>0.30).
- Top-20 worst/best đã lưu trong `artifacts/roberta_top_*.csv` và `bert_finetune_top_*.csv`. Patterns cần điều tra (không suy diễn causal): sarcasm, mixed sentiment, ambiguous language, rating/text mismatch.

### 6.6 Resource comparison

- **Nhanh nhất:** TF-IDF (19s) nhưng MAE cao.
- **Tốt nhất về time/accuracy:** DistilBERT (567s, MAE 0.100).
- **Chính xác nhất:** RoBERTa (1065s, MAE 0.070, R² 0.816).
- Plots: `artifacts/comparison_mae.png`, `tradeoff_time_vs_mae.png`, `tradeoff_params_vs_mae.png` + loss/error_buckets/prediction_vs_target/residual per artifact.

---

## 7. Thảo luận (Discussion)

### 7.1 Diễn giải trung thực (R05)

Các metrics cho thấy model captures statistical relationships between text representations and target scores in this IMDb dataset, không nên suy diễn thành "model hiểu cảm xúc con người".

### 7.2 Hạn chế (Limitations)

- **Dataset:** IMDb chỉ có 2 mức rating (1 và 5) được map thành binary target 0/1 — không phải intensity liên tục thực sự (R06, caveat trong `docs/DATASET.md`). Yelp (1–5 đầy đủ) sẽ là mở rộng tốt hơn.
- **Frozen BERT tệ:** cho thấy với regression head đơn giản, frozen không đủ; cần thử head phức tạp hơn hoặc E08 ablation.
- **Chưa có cross-validation:** chỉ dùng 1 split seed 42, chưa đánh giá variance.
- **Chưa có E08 ablation:** max_length, pooling, HuberLoss chưa được thử.

### 7.3 Đe doạ tính hợp lệ

- Rating có thể không phản ánh sentiment intensity một cách hoàn hảo.
- Chỉ đánh giá trên IMDb tiếng Anh — chưa kiểm chứng đa dạng domain.

---

## 8. Kết luận

### 8.1 Thành tựu

- ✅ Pipeline NLP regression hoàn chỉnh, reproducible (seed 42, config-driven, checkpoint đầy đủ).
- ✅ 5 experiments với số thực, fair comparison, trả lời 5 research questions.
- ✅ RoBERTa finetune đạt MAE 0.070, R² 0.816 — tốt nhất.
- ✅ DistilBERT chứng minh trade-off tốt (66M, nhanh 45%).
- ✅ API + UI hoạt động, 17 tests PASS, audit checklist 32 mục.

### 8.2 Hạn chế

- IMDb binary làm giảm ý nghĩa "continuous" của regression.
- Chưa làm E08 ablation, chưa cross-validation.
- Checkpoint ~500 MB/model, artifacts 32M/predictions — bị gitignore, cần chia sẻ riêng nếu giảng viên cần kiểm chứng.

### 8.3 Hướng phát triển

- Thêm Yelp dataset với rating 1–5 đầy đủ.
- E08: ablation max_length (128 vs 256), mean pooling, HuberLoss.
- k-fold cross-validation để báo cáo mean ± std.
- Quantization/distillation để deploy nhẹ hơn.
- Mở rộng Vibe Coding workflow với CI/CD (GitHub Actions chạy tests + audit tự động).

---

## Tài liệu tham khảo

- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
- Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." 2019.
- Sanh et al. "DistilBERT, a distilled version of BERT." 2019.
- Maas et al. "Learning Word Vectors for Sentiment Analysis." ACL 2011 (IMDb dataset).
- HuggingFace Transformers, Datasets, PyTorch documentation.

---

## Phụ lục

### A. Cách chạy lại

```bash
conda run -n base python -m src.data.prepare          # ~1 phút
conda run -n base python -m src.training.run_baseline --config configs/baseline.yaml   # ~20s
conda run -n base python -m src.training.train --config configs/bert_finetune.yaml      # ~17 phút (GPU)
conda run -n base python -m src.training.train --config configs/roberta_finetune.yaml   # ~18 phút
conda run -n base python -m src.training.train --config configs/distilbert_finetune.yaml # ~10 phút
conda run -n base python -m pytest tests/ -v            # ~7s, 17 tests
# API / UI
conda run -n base uvicorn src.api:app --reload
conda run -n base streamlit run src/ui/app.py
```

### B. Artifacts (local, gitignored)

```
artifacts/
├── baseline_imdb_seed42/         — TF-IDF MAE 0.251
├── bert_frozen_imdb_seed42/      — MAE 0.455
├── bert_finetune_imdb_seed42/    — MAE 0.088
├── distilbert_finetune_imdb_seed42/ — MAE 0.100
├── roberta_finetune_imdb_seed42/ — MAE 0.070 (best)
├── summary.csv                   — bảng tổng hợp
├── comparison_mae.png, tradeoff_*.png
├── bert_finetune_top_*.csv, roberta_top_*.csv
└── (mỗi exp: config.yaml, metrics.json, training_history.csv, predictions.csv, model_metadata.json, plots/)
```

### C. Thông tin nhóm

- Branch: `feature/evaluation` (5 commits) → PR vào `main`
- Repo: `HuynhPhong385/text-regression-transformers`
- Progress: `progress_status.md` — 11 phases, 100% DONE

---

*Báo cáo này tuân thủ R01 (không fake metrics), R05 (diễn giải trung thực), và quy tắc "Chỉ đưa actual experimental results" trong `docs/REPORT.md`.*

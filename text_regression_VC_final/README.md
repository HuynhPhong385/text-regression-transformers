# Text Regression with Transformers — AI Agent + Vibe Coding

## 1. Project overview

Đề tài: **Xây dựng hệ thống Text Regression bằng Transformer Encoder để dự đoán sentiment score từ review**.

Project được thiết kế theo workflow mà giảng viên yêu cầu:

```text
Business Understanding
        ↓
Features / Tech Solution / AI Logic
        ↓
Plan
        ↓
Knowledge Base
        ↓
AI Agent
   ├── Prompt
   ├── Rules
   ├── Knowledge Base
   └── Skills
        ↓
Vibe Coding
        ↓
Implementation
        ↓
Model-level Testing
        ↓
Full-pipeline Testing
        ↓
Progress Tracking
        ↓
Codebase Audit
        ↓
Iteration
```

## 2. ML problem

Input:

```text
"The movie was excellent and very entertaining."
```

Output:

```text
sentiment_score = 0.87
```

Target được chuẩn hóa về `[0, 1]`.

Model chính:

- BERT + Regression Head
- RoBERTa + Regression Head
- DistilBERT + Regression Head

Strategy:

- Frozen encoder
- Full fine-tuning

Baseline:

- Mean predictor
- TF-IDF + Linear Regression

## 3. Project goals

### Technical goals

- Xây dựng pipeline NLP regression hoàn chỉnh.
- So sánh Transformer với traditional ML.
- So sánh frozen encoder và fine-tuning.
- So sánh BERT, RoBERTa và DistilBERT.
- Xây dựng inference API và demo UI.
- Đảm bảo reproducibility.

### AI Engineering goals

- Xây dựng Knowledge Base cho AI Agent.
- Dùng Prompt + Rules + Skills để điều khiển Vibe Coding.
- Theo dõi tiến độ bằng `progress_status.md`.
- Audit codebase định kỳ.
- Test cả model-level và full pipeline.

## 4. Repository structure

```text
text-regression-transformers/
├── README.md
├── AGENTS.md
├── progress_status.md
├── codebase_audit.md
├── naming_convention.md
├── folder_structure.md
│
├── knowledge_base/
│   ├── business_understanding.md
│   ├── tech_solution.md
│   ├── project_context.md
│   ├── pipeline.md
│   ├── data_prep.md
│   ├── training_info.md
│   ├── model.md
│   ├── eval.md
│   └── experiment_plan.md
│
├── .ai/
│   ├── agent.md
│   ├── prompts/
│   │   ├── planning.md
│   │   ├── implementation.md
│   │   ├── testing.md
│   │   ├── debugging.md
│   │   └── audit.md
│   ├── rules/
│   │   ├── working_rules.md
│   │   ├── coding_rules.md
│   │   └── research_rules.md
│   └── skills/
│       ├── data_preparation.md
│       ├── model_training.md
│       ├── evaluation.md
│       ├── api_ui.md
│       └── codebase_audit.md
│
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── RESEARCH_QUESTIONS.md
│   ├── DATASET.md
│   ├── DATA_PIPELINE.md
│   ├── MODEL.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   ├── EXPERIMENTS.md
│   ├── ERROR_ANALYSIS.md
│   ├── API.md
│   ├── UI.md
│   ├── REPORT.md
│   ├── ROADMAP.md
│   ├── DECISIONS.md
│   ├── COMMANDS.md
│   └── FINAL_CHECKLIST.md
│
├── configs/
├── data/
├── src/
├── tests/
├── scripts/
├── notebooks/
└── artifacts/
```

## 5. How to use the repository

### Step 1

AI Agent đọc:

```text
AGENTS.md
knowledge_base/
.ai/rules/
progress_status.md
```

### Step 2

Agent xác định task hiện tại.

### Step 3

Agent chọn skill + prompt phù hợp.

### Step 4

Agent lập plan ngắn.

### Step 5

Agent code.

### Step 6

Agent test.

### Step 7

Agent cập nhật progress.

### Step 8

Định kỳ chạy codebase audit.

## 6. Development order

```text
Phase 0 — Business Understanding
Phase 1 — Project bootstrap
Phase 2 — Data preparation
Phase 3 — Baselines
Phase 4 — BERT MVP
Phase 5 — Frozen vs Fine-tuning
Phase 6 — RoBERTa / DistilBERT
Phase 7 — Model-level evaluation
Phase 8 — Full pipeline
Phase 9 — API + UI
Phase 10 — Audit + report
```

## 7. Academic rule

Không được tạo hoặc ghi số liệu thực nghiệm giả.

Mọi metric trong báo cáo phải đến từ experiment thực tế.

## 8. Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
```

Các command chi tiết nằm trong `docs/COMMANDS.md`.

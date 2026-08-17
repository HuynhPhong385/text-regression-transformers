# Business Understanding

## Description

Chuyển user requirements thành yêu cầu sản phẩm có thể triển khai và kiểm thử.

## Goal

Xây dựng hệ thống nhận review tiếng Anh và dự đoán sentiment score liên tục.

## 1. Features

### F01 — Review input

User nhập một đoạn review.

### F02 — Score prediction

System trả về score `[0,1]`.

### F03 — Model selection

Demo có thể chọn model đã train.

### F04 — API inference

Có endpoint để prediction.

### F05 — Model comparison

Có bảng so sánh các experiment.

### F06 — Error analysis

Có phân tích những prediction sai.

## 2. Tech solution

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas/numpy
- FastAPI
- Streamlit
- pytest

## 3. Logic + AI solution

```text
Review
 ↓
Validation
 ↓
Tokenizer
 ↓
Transformer Encoder
 ↓
Regression Head
 ↓
Score [0,1]
```

Models:

```text
BERT
RoBERTa
DistilBERT
```

Strategies:

```text
Frozen encoder
Fine-tuning
```

## 4. Implementation

```text
Requirement
→ Plan
→ AI Agent
→ Vibe Coding
→ Test
→ Review
→ Progress update
```

## 5. Testing

### Model level

- predictive metrics;
- training time;
- inference time;
- parameter count;
- memory.

### Full pipeline

```text
Input
→ preprocessing
→ model
→ inference
→ API
→ output
```

## Acceptance criteria

MVP phải:
- train được;
- evaluate được;
- load checkpoint được;
- predict một review được;
- metrics được lưu;
- có reproducible config.

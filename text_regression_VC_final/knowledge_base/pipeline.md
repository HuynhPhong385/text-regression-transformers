# System Pipeline

## Training pipeline

```text
Raw Dataset
 ↓
Schema Validation
 ↓
Cleaning
 ↓
Target Conversion
 ↓
Train/Validation/Test
 ↓
Tokenizer
 ↓
DataLoader
 ↓
Transformer
 ↓
Regression Head
 ↓
Loss
 ↓
Backpropagation
 ↓
Checkpoint
```

## Evaluation pipeline

```text
Checkpoint
 ↓
Test Data
 ↓
Inference
 ↓
Prediction
 ↓
Metrics
 ↓
Plots
 ↓
Report
```

## Full application pipeline

```text
User
 ↓
UI/API
 ↓
Input Validation
 ↓
Tokenizer
 ↓
Loaded Transformer
 ↓
Regression Head
 ↓
Post-processing
 ↓
Score
 ↓
UI/API response
```

## Failure points

Agent phải xử lý:
- missing input;
- tokenizer/model mismatch;
- missing checkpoint;
- invalid config;
- corrupted dataset;
- incompatible device;
- out-of-memory.

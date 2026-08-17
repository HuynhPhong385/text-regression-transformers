# Skill — API/UI

## API

FastAPI:

```text
GET /health
POST /predict
GET /model-info
```

## UI

Streamlit:

- text input;
- model selector;
- prediction;
- model information;
- error handling.

## Full-pipeline test

Input → tokenizer → model → output.

Không fake prediction nếu checkpoint không tồn tại.

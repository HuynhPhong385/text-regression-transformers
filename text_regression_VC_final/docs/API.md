# API

## Health

```http
GET /health
```

## Prediction

```http
POST /predict
```

Request:

```json
{
  "text": "The movie was excellent."
}
```

Response:

```json
{
  "score": 0.87,
  "model": "bert",
  "strategy": "finetune"
}
```

## Model info

```http
GET /model-info
```

## Rules

Model load một lần lúc startup.

Input rỗng → validation error.

Missing checkpoint → explicit error, không fake output.

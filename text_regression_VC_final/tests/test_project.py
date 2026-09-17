"""Tests — model factory and data validation."""
import pytest
from src.models.factory import create_model


def test_factory_creates_bert():
    m = create_model("bert-base-uncased", strategy="finetune")
    assert m is not None


def test_factory_frozen_has_few_trainable():
    m = create_model("bert-base-uncased", strategy="frozen")
    enc_params = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
    assert enc_params == 0


def test_factory_finetune_has_many_trainable():
    m = create_model("bert-base-uncased", strategy="finetune")
    enc_params = sum(p.numel() for p in m.encoder.parameters() if p.requires_grad)
    assert enc_params > 0


def test_target_conversion():
    from src.data.prepare import rating_to_target
    assert rating_to_target(1) == 0.0
    assert rating_to_target(3) == 0.5
    assert rating_to_target(5) == 1.0


def test_validate_schema_detects_null():
    import pandas as pd
    from src.data.validate import validate_schema
    df = pd.DataFrame({"id": ["a"], "text": [None], "rating": [5], "target": [1.0], "split": ["train"]})
    errors = validate_schema(df)
    assert len(errors) > 0


def test_metrics_compute():
    from src.evaluation.metrics import compute_all_metrics
    r = compute_all_metrics([0, 0.5, 1], [0.1, 0.4, 1.0])
    assert "mae" in r and "mse" in r and "r2" in r


def test_api_health():
    from fastapi.testclient import TestClient
    from src.api.app import app
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_predict_empty():
    from fastapi.testclient import TestClient
    from src.api.app import app
    c = TestClient(app)
    r = c.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predictor_missing_checkpoint():
    from src.inference.predictor import Predictor
    import pytest
    with pytest.raises(Exception):
        Predictor("artifacts/nonexistent_ckpt")

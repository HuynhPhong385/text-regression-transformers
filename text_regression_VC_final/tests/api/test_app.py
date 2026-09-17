import pytest
from fastapi.testclient import TestClient

from src.api import app as api_app


class FakeRegistry:
    def __init__(self, loaded: bool = True, fixed_score: float = 0.73):
        self.loaded = loaded
        self.fixed_score = fixed_score
        self.default_model_key = "distilbert"
        self._models = {
            "distilbert": {
                "model_key": "distilbert",
                "display_name": "DistilBERT",
                "model_name": "distilbert-base-uncased",
                "strategy": "fine_tune",
                "loaded": loaded,
                "error": None if loaded else "checkpoint chưa có",
            },
            "bert": {
                "model_key": "bert",
                "display_name": "BERT",
                "model_name": "bert-base-uncased",
                "strategy": "fine_tune",
                "loaded": False,
                "error": "checkpoint chưa có",
            },
        }

    def model_info(self, model_key=None):
        if model_key is not None:
            return self._models[model_key]
        return self._models

    def predict(self, text, model_key=None):
        key = model_key or self.default_model_key
        if not self._models[key]["loaded"]:
            raise RuntimeError(f"Model '{key}' chưa sẵn sàng (fake).")
        if not text or not text.strip():
            raise ValueError("Input rỗng (fake).")
        return self.fixed_score


@pytest.fixture
def client_loaded(monkeypatch):
    fake = FakeRegistry(loaded=True)
    monkeypatch.setattr(api_app, "get_registry", lambda: fake)
    return TestClient(api_app.app)


@pytest.fixture
def client_not_loaded(monkeypatch):
    fake = FakeRegistry(loaded=False)
    monkeypatch.setattr(api_app, "get_registry", lambda: fake)
    return TestClient(api_app.app)


class TestHealth:
    def test_health_ok(self, client_loaded):
        resp = client_loaded.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["default_model_loaded"] is True

    def test_health_model_not_loaded(self, client_not_loaded):
        resp = client_not_loaded.get("/health")
        assert resp.status_code == 200
        assert resp.json()["default_model_loaded"] is False


class TestModelInfo:
    def test_model_info_lists_all_models(self, client_loaded):
        resp = client_loaded.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "distilbert" in data["models"]
        assert "bert" in data["models"]
        assert data["default_model"] == "distilbert"


class TestPredict:
    def test_predict_with_default_model(self, client_loaded):
        resp = client_loaded.post("/predict", json={"text": "Sản phẩm rất tốt"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "distilbert"
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_with_explicit_model_selector(self, client_loaded):
       
        resp = client_loaded.post("/predict", json={"text": "abc", "model_name": "bert"})
        assert resp.status_code == 503 

    def test_predict_returns_503_when_default_not_loaded(self, client_not_loaded):
        resp = client_not_loaded.post("/predict", json={"text": "abc"})
        assert resp.status_code == 503

    def test_predict_empty_text_returns_422(self, client_loaded):
        resp = client_loaded.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_missing_field_returns_422(self, client_loaded):
        resp = client_loaded.post("/predict", json={})
        assert resp.status_code == 422


class TestFullPipeline:
    """End-to-end giống UI thật: health -> model-info -> predict."""

    def test_full_flow(self, client_loaded):
        health = client_loaded.get("/health").json()
        assert health["status"] == "ok"

        models = client_loaded.get("/model-info").json()
        assert models["default_model"] in models["models"]

        result = client_loaded.post(
            "/predict", json={"text": "Kiểm tra pipeline", "model_name": models["default_model"]}
        ).json()
        assert 0.0 <= result["score"] <= 1.0

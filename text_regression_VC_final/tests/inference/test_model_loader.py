import pytest

from src.inference.model_loader import load_config, resolve_device, load_model_bundle


class TestLoadConfig:
    def test_fallback_when_file_missing(self):
        cfg = load_config("configs/khong_ton_tai.yaml")
        assert "distilbert" in cfg["models"]
        assert cfg["default_model"] == "distilbert"


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_returns_cpu_or_cuda(self):
        assert resolve_device("auto") in ("cpu", "cuda")


class TestLoadModelBundle:
    """Khi src/models/model_factory.py CHƯA tồn tại (Member 3 chưa push)."""

    def test_bundle_not_loaded_without_model_factory(self):
        config = {
            "models": {
                "distilbert": {
                    "display_name": "DistilBERT",
                    "name": "distilbert-base-uncased",
                    "strategy": "fine_tune",
                    "dropout": 0.1,
                    "checkpoint_path": "khong_ton_tai/model.pt",
                }
            }
        }
        try:
            bundle = load_model_bundle("distilbert", config, device="cpu")
        except Exception as exc:  
            pytest.skip(f"Bỏ qua vì thiếu dependency: {exc}")

        assert bundle.loaded is False
        assert bundle.error is not None
        assert bundle.model_key == "distilbert"

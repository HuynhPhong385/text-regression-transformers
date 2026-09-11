import pytest
from unittest.mock import MagicMock

from src.inference.predictor import (
    PredictorRegistry,
    validate_text,
    postprocess_score,
    reset_registry,
)
from src.inference.model_loader import ModelBundle


@pytest.fixture(autouse=True)
def _reset():
    reset_registry()
    yield
    reset_registry()


# Validation
class TestValidateText:
    def test_valid_text_passes(self):
        assert validate_text("  hello world  ") == "hello world"

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_text(None)

    def test_empty_after_strip_raises(self):
        with pytest.raises(ValueError):
            validate_text("   ")


# Post-processing
class TestPostprocessScore:
    def test_clips_above_one(self):
        assert postprocess_score(1.2) == 1.0

    def test_clips_below_zero(self):
        assert postprocess_score(-0.3) == 0.0

    def test_passthrough_in_range(self):
        assert postprocess_score(0.55) == 0.55


# Registry với model đã "loaded" 
def _make_fake_bundle(model_key: str, raw_score: float = 0.8) -> ModelBundle:
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value.to.return_value = {"input_ids": [[0, 1, 2]]}

    fake_output = MagicMock()
    fake_output.squeeze.return_value.item.return_value = raw_score
    fake_model = MagicMock(return_value=fake_output)

    return ModelBundle(
        model_key=model_key,
        display_name=model_key,
        model_name=f"{model_key}-base",
        strategy="fine_tune",
        tokenizer=fake_tokenizer,
        model=fake_model,
        loaded=True,
        error=None,
    )


class TestPredictorRegistry:
    def test_predict_full_flow_with_mocked_model(self, monkeypatch):
        registry = PredictorRegistry.__new__(PredictorRegistry)  
        registry.config = {"models": {"distilbert": {}}, "default_model": "distilbert", "max_length": 64}
        registry.device = "cpu"
        registry.max_length = 64
        registry.default_model_key = "distilbert"
        registry._bundles = {"distilbert": _make_fake_bundle("distilbert", raw_score=0.8)}

        score = registry.predict("Sản phẩm rất tốt", model_key="distilbert")
        assert 0.0 <= score <= 1.0
        assert score == 0.8

    def test_predict_raises_when_model_not_loaded(self):
        registry = PredictorRegistry.__new__(PredictorRegistry)
        registry.config = {"models": {"distilbert": {}}, "default_model": "distilbert", "max_length": 64}
        registry.device = "cpu"
        registry.max_length = 64
        registry.default_model_key = "distilbert"
        not_loaded_bundle = _make_fake_bundle("distilbert")
        not_loaded_bundle.loaded = False
        not_loaded_bundle.error = "checkpoint không tồn tại"
        registry._bundles = {"distilbert": not_loaded_bundle}

        with pytest.raises(RuntimeError):
            registry.predict("abc", model_key="distilbert")

    def test_predict_raises_valueerror_on_invalid_input(self):
        registry = PredictorRegistry.__new__(PredictorRegistry)
        registry.config = {"models": {"distilbert": {}}, "default_model": "distilbert", "max_length": 64}
        registry.device = "cpu"
        registry.max_length = 64
        registry.default_model_key = "distilbert"
        registry._bundles = {"distilbert": _make_fake_bundle("distilbert")}

        with pytest.raises(ValueError):
            registry.predict("   ", model_key="distilbert")

    def test_model_info_reports_loaded_state(self):
        registry = PredictorRegistry.__new__(PredictorRegistry)
        registry.config = {"models": {"distilbert": {}, "bert": {}}, "default_model": "distilbert", "max_length": 64}
        registry.device = "cpu"
        registry.max_length = 64
        registry.default_model_key = "distilbert"
        registry._bundles = {
            "distilbert": _make_fake_bundle("distilbert"),
        }

        info = registry._bundle_to_info(registry._bundles["distilbert"])
        assert info["loaded"] is True
        assert info["model_key"] == "distilbert"

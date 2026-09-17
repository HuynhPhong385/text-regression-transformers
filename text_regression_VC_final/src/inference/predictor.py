import logging
from typing import Any, Dict, List, Optional

import torch

from src.inference.model_loader import (
    load_config,
    resolve_device,
    load_model_bundle,
    ModelBundle,
    DEFAULT_CONFIG_PATH,
)

logger = logging.getLogger("predictor")


def validate_text(text: str) -> str:
    """Bước Validation trong pipeline. Raise ValueError nếu input không hợp lệ."""
    if text is None:
        raise ValueError("Input text không được None.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text rỗng sau khi strip().")
    return cleaned


def postprocess_score(raw_score: float) -> float:
    """Bước Post-processing: đảm bảo score luôn nằm trong [0,1] dù model có lệch nhỏ."""
    return max(0.0, min(1.0, float(raw_score)))


class PredictorRegistry:
    """
    Quản lý nhiều ModelBundle (bert/roberta/distilbert...) để UI có thể chọn
    model qua "Model selector". Load lười (lazy) — chỉ load model khi lần đầu
    được yêu cầu, tránh tốn RAM/thời gian khởi động nếu không dùng tới.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.device = resolve_device(self.config.get("inference", {}).get("device", "auto"))
        self.max_length = self.config.get("max_length", 256)
        self.default_model_key = self.config.get("default_model")
        self._bundles: Dict[str, ModelBundle] = {}

    def available_model_keys(self) -> List[str]:
        return list(self.config.get("models", {}).keys())

    def _get_bundle(self, model_key: Optional[str] = None) -> ModelBundle:
        key = model_key or self.default_model_key
        if key not in self.config.get("models", {}):
            raise ValueError(f"model_key '{key}' không tồn tại trong config.")
        if key not in self._bundles:
            self._bundles[key] = load_model_bundle(key, self.config, self.device)
        return self._bundles[key]

    def model_info(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """Dữ liệu cho endpoint GET /model-info và cho UI dựng Model selector."""
        if model_key is not None:
            bundle = self._get_bundle(model_key)
            return self._bundle_to_info(bundle)
        return {k: self._bundle_to_info(self._get_bundle(k)) for k in self.available_model_keys()}

    @staticmethod
    def _bundle_to_info(bundle: ModelBundle) -> Dict[str, Any]:
        return {
            "model_key": bundle.model_key,
            "display_name": bundle.display_name,
            "model_name": bundle.model_name,
            "strategy": bundle.strategy,
            "loaded": bundle.loaded,
            "error": bundle.error,
        }

    @torch.inference_mode()
    def predict(self, text: str, model_key: Optional[str] = None) -> float:
        """
        Full flow: Validation -> Tokenizer -> Model -> Prediction -> Post-processing.
        Trả về 1 score trong [0,1].
        """
        clean_text = validate_text(text)

        bundle = self._get_bundle(model_key)
        if not bundle.loaded:
            raise RuntimeError(
                f"Model '{bundle.model_key}' chưa sẵn sàng: {bundle.error}"
            )

        inputs = bundle.tokenizer(
            clean_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        raw_output = bundle.model(**inputs)
        raw_score = raw_output.squeeze().item()

        return postprocess_score(raw_score)  


_registry_instance: Optional[PredictorRegistry] = None


def get_registry() -> PredictorRegistry:
    """Singleton dùng chung cho API/UI."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PredictorRegistry()
    return _registry_instance


def reset_registry() -> None:
    """Dùng trong test để buộc tạo lại registry với config/mock khác."""
    global _registry_instance
    _registry_instance = None

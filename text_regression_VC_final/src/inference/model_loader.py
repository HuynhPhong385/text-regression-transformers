import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoTokenizer

logger = logging.getLogger("model_loader")

DEFAULT_CONFIG_PATH = os.getenv("API_CONFIG_PATH", "configs/api_config.yaml")

_FALLBACK_CONFIG: Dict[str, Any] = {
    "models": {
        "distilbert": {
            "display_name": "DistilBERT (fallback)",
            "name": "distilbert-base-uncased",
            "strategy": "fine_tune",
            "dropout": 0.1,
            "checkpoint_path": "artifacts/checkpoints/distilbert_finetune_imdb_seed42/best_model.pt",
        }
    },
    "default_model": "distilbert",
    "max_length": 256,
    "inference": {"device": "auto"},
    "api": {"host": "0.0.0.0", "port": 8000, "cors_origins": ["*"]},
    "ui": {"api_url": "http://localhost:8000"},
}


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Đọc config YAML. Nếu không tìm thấy, dùng fallback để dev không bị chặn."""
    if not os.path.exists(config_path):
        logger.warning(f"Không tìm thấy config tại '{config_path}', dùng giá trị mặc định.")
        return _FALLBACK_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@dataclass
class ModelBundle:
    """Gói tokenizer + model + metadata cho MỘT model_key, đã sẵn sàng để predict."""

    model_key: str
    display_name: str
    model_name: str
    strategy: str
    tokenizer: Any = None
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None


def load_model_bundle(model_key: str, config: Dict[str, Any], device: str) -> ModelBundle:
    """
    Load 1 model theo model_key trong config["models"].

    Thiết kế resilient: nếu model_factory (Member 3) hoặc checkpoint chưa có,
    trả về ModelBundle với loaded=False + error, KHÔNG raise, để API/UI vẫn
    chạy được và hiển thị trạng thái rõ ràng qua /model-info.
    """
    model_cfg = config["models"][model_key]
    bundle = ModelBundle(
        model_key=model_key,
        display_name=model_cfg.get("display_name", model_key),
        model_name=model_cfg["name"],
        strategy=model_cfg.get("strategy", "fine_tune"),
    )

    try:
        bundle.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    except Exception as exc:
        bundle.error = f"Không load được tokenizer '{model_cfg['name']}': {exc}"
        logger.warning(bundle.error)
        return bundle

    try:

        from src.models.model_factory import create_model

        bundle.model = create_model(
            model_name=model_cfg["name"],
            strategy=model_cfg.get("strategy", "fine_tune"),
            dropout=model_cfg.get("dropout", 0.1),
        )
    except ImportError as exc:
        bundle.error = f"Chưa có src/models/model_factory.py (Member 3 chưa push): {exc}"
        logger.warning(bundle.error)
        return bundle
    except Exception as exc:
        bundle.error = f"Lỗi khi tạo model '{model_key}' từ model_factory: {exc}"
        logger.warning(bundle.error)
        return bundle

    checkpoint_path = model_cfg.get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        bundle.error = f"Chưa tìm thấy checkpoint tại '{checkpoint_path}'."
        logger.warning(bundle.error)
        return bundle

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        bundle.model.load_state_dict(state_dict)
    except Exception as exc:
        bundle.error = f"Không load được checkpoint '{checkpoint_path}': {exc}"
        logger.warning(bundle.error)
        return bundle

    bundle.model.to(device)
    bundle.model.eval()
    bundle.loaded = True
    logger.info(f"Model '{model_key}' ({bundle.model_name}) sẵn sàng trên {device}")
    return bundle

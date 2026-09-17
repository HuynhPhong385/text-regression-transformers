"""Inference predictor — loads checkpoint + tokenizer."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.models.factory import create_model


class Predictor:
    def __init__(self, checkpoint_dir: str | Path):
        ckpt_dir = Path(checkpoint_dir)
        ckpt_path = ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else ckpt_dir
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.model_name: str = cfg.get("model_name", "bert-base-uncased")
        self.strategy: str = cfg.get("strategy", "finetune")
        self.experiment: str = cfg.get("experiment_name", ckpt_dir.name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = create_model(self.model_name, strategy=self.strategy)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        # Tokenizer saved alongside checkpoint
        tok_dir = ckpt_dir if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "vocab.txt").exists() else ckpt_dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = int(cfg.get("max_length", 256))

    @torch.no_grad()
    def predict(self, text: str) -> float:
        if not text or not text.strip():
            raise ValueError("Input text is empty")
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        score = self.model(input_ids, attention_mask).item()
        return float(max(0.0, min(1.0, score)))

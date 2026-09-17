"""FastAPI inference service."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Text Regression API")

_predictor = None
_experiment = os.environ.get("EXPERIMENT", "")


def get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    # Resolve checkpoint dir: env or latest artifact with checkpoint
    ckpt_dir = None
    if _experiment:
        ckpt_dir = Path(f"artifacts/{_experiment}/checkpoints")
    if ckpt_dir is None or not ckpt_dir.exists():
        # Fallback: find any artifact with best.pt
        for p in sorted(Path("artifacts").glob("*/checkpoints/best.pt")):
            ckpt_dir = p.parent
            break
    if ckpt_dir is None or not ckpt_dir.exists():
        return None
    from src.inference.predictor import Predictor
    _predictor = Predictor(ckpt_dir)
    return _predictor


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    score: float
    model: str
    strategy: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    pred = get_predictor()
    if pred is None:
        raise HTTPException(status_code=503, detail="No checkpoint loaded")
    return {"model": pred.model_name, "strategy": pred.strategy, "experiment": pred.experiment}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="Input text is empty")
    pred = get_predictor()
    if pred is None:
        raise HTTPException(status_code=503, detail="No checkpoint available - run training first")
    score = pred.predict(req.text)
    return PredictResponse(score=score, model=pred.model_name, strategy=pred.strategy)

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Văn bản đầu vào cần dự đoán")
    model_name: Optional[str] = Field(
        None, description="model_key để chọn model (xem GET /model-info). Bỏ trống dùng default_model."
    )


class PredictResponse(BaseModel):
    text: str
    model_name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Score đã hậu xử lý, trong [0,1]")


class ModelInfo(BaseModel):
    model_key: str
    display_name: str
    model_name: str
    strategy: str
    loaded: bool
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    default_model: str
    models: Dict[str, ModelInfo]


class HealthResponse(BaseModel):
    status: str
    default_model: str
    default_model_loaded: bool

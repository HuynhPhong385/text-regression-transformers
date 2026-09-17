import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    ModelInfoResponse,
    ModelInfo,
    HealthResponse,
)
from src.inference.predictor import get_registry
from src.inference.model_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model mặc định ngay khi service khởi động
    registry = get_registry()
    try:
        registry.model_info(registry.default_model_key)
    except Exception as exc:
        logger.warning(f"Không thể preload default model lúc startup: {exc}")
    yield


app = FastAPI(
    title="Text Regression API",
    description="API dự đoán score [0,1] từ văn bản, hỗ trợ chọn model (bert/roberta/distilbert).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    registry = get_registry()
    info = registry.model_info(registry.default_model_key)
    return HealthResponse(
        status="ok",
        default_model=registry.default_model_key,
        default_model_loaded=info["loaded"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    registry = get_registry()
    model_key = request.model_name or registry.default_model_key
    start = time.time()
    try:
        score = registry.predict(request.text, model_key=model_key)
    except ValueError as exc:

        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:

        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Lỗi khi predict")
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ khi predict: {exc}")

    logger.info(f"/predict [{model_key}] xử lý trong {time.time() - start:.3f}s")
    return PredictResponse(text=request.text, model_name=model_key, score=score)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Danh sách model khả dụng — UI dùng endpoint này để dựng Model selector."""
    registry = get_registry()
    info = registry.model_info()
    return ModelInfoResponse(
        default_model=registry.default_model_key,
        models={k: ModelInfo(**v) for k, v in info.items()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=True,
    )
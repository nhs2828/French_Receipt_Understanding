from fastapi import APIRouter, Request, Response, status

from app.core.config import get_paddle_ocr_params, get_segmentation_params, get_settings
from app.schemas import (
    HealthResponse,
    LivenessResponse,
    ModelsStatusResponse,
    ModelStatus,
    ReadinessResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Quick health check (legacy compatibility)")
async def health():
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.APP_VERSION)


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe - checks if the process is alive (does not check models)",
)
async def liveness():
    return LivenessResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe - checks if all models are loaded and ready to accept traffic",
)
async def readiness(request: Request, response: Response):
    seg = request.app.state.segmentation_service
    ext = request.app.state.text_extraction_service

    all_loaded = seg.is_loaded and ext.is_loaded

    if not all_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", all_models_loaded=False)

    return ReadinessResponse(
        status="ready",
        all_models_loaded=all_loaded,
    )


@router.get(
    "/models/status",
    response_model=ModelsStatusResponse,
    summary="Status of each model in the pipeline (loaded status, device, load time)",
)
async def models_status(request: Request):
    seg_settings = get_segmentation_params()
    ext_settings = get_paddle_ocr_params()
    seg = request.app.state.segmentation_service
    ext = request.app.state.text_extraction_service

    return ModelsStatusResponse(
        segmentation=ModelStatus(
            name="YOLO-seg",
            loaded=seg.is_loaded,
            device=seg_settings.device,
            load_time_ms=seg.load_time_ms,
        ),
        text_extraction=ModelStatus(
            name="PaddleOCR TextProcessing + TextDetection + Extraction Wrapper",
            loaded=ext.is_loaded,
            device=ext_settings.device,
            load_time_ms=ext.load_time_ms,
        ),
    )
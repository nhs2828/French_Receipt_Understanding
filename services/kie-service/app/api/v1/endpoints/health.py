from fastapi import APIRouter, Request, Response, status

from app.core.config import get_settings
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
    kie = request.app.state.kie_service

    all_loaded = kie.is_loaded

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
    settings = get_settings()
    kie = request.app.state.kie_service


    return ModelsStatusResponse(
        kie=ModelStatus(
            name="LayoutLM",
            loaded=kie.is_loaded,
            device=settings.device,
            load_time_ms=kie.load_time_ms,
        )
    )
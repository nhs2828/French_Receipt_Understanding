from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from prometheus_fastapi_instrumentator import Instrumentator

from app.services.segmentation_service import SegmentationService
from app.services.text_extraction_service import TextExtractionService
from app.api.v1.routers import api_router
from app.middleware.request_tracing import RequestTracingMiddleware
from app.middleware.logging_middleware import AccessLogMiddleware
from app.middleware.upload_guard import MaxUploadSizeMiddleware
from app.middleware.error_handler import register_exception_handlers
from app.core.config import settings
from app.core.metrics import MODEL_LOADED
from app.core.logging import setup_logging, get_logger


setup_logging()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.segmentation_service = SegmentationService()
    app.state.text_extraction_service = TextExtractionService()

    await run_in_threadpool(app.state.segmentation_service.load)
    MODEL_LOADED.labels(model_name="segmentation").set(1)
    await run_in_threadpool(app.state.text_extraction_service.load)
    MODEL_LOADED.labels(model_name="text_extraction").set(1)

    #app.state.inference_limiter = anyio.CapacityLimiter(settings.INFERENCE_MAX_WORKERS)
    app.state.inference_executor = ThreadPoolExecutor(
        max_workers=settings.INFERENCE_MAX_WORKERS, thread_name_prefix="inference"
    )

    app.state.pipeline_in_flight = 0

    logger.info("vision-service ready")
    yield
    logger.info("Shutting down vision-service...")
    app.state.inference_executor.shutdown(wait=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "API extracts French receipts information\n\n"
            "Pipeline: Segmentation (YOLO-seg) -> OCR (PaddleOCR TextDetection + "
            "PaddleOCR recognition)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- Rate limiting ---
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --- Middleware ---
    # The order of add_middleware is reversed relative to execution order (the last added
    # middleware runs first). MaxUploadSize + TrustedHost + CORS should run earliest
    # (blocking bad requests before wasting compute), while RequestTracing runs early so
    # all subsequent logs and middlewares have access to request_id.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(MaxUploadSizeMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestTracingMiddleware)

    # --- Exception handlers ---
    register_exception_handlers(app)

    # --- Routers ---
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # --- Metrics ---
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()


# uvicorn app.main:app --reload --port 8001
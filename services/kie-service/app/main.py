from pathlib import Path
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from vision_client import VisionClient
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.metrics import MODEL_LOADED
from app.services import KIEService
from app.api.v1.routers import api_router
from app.middleware.request_tracing import RequestTracingMiddleware
from app.middleware.logging_middleware import AccessLogMiddleware
from app.middleware.upload_guard import MaxUploadSizeMiddleware
from app.middleware.error_handler import register_exception_handlers


settings = get_settings()

setup_logging()
logger = get_logger(__name__)

ROOT_DIR = Path(__file__).parents[1]
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kie_service = KIEService(
        config_path=ROOT_DIR / "app/configs/fr_sroie.yaml",
        checkpoint=ROOT_DIR / "models/layoutlm/"
    )
    await run_in_threadpool(app.state.kie_service.load)
    MODEL_LOADED.labels(model_name="kie-layoutlm").set(1)
    app.state.vision_client = VisionClient(settings.VISION_SERVICE_URL, settings.VISION_SERVICE_TIMEOUT)

    # Dedicated capacity limiter for CPU-bound inference work, scoped to this app only.
    # Do NOT mutate anyio.to_thread's global default limiter here — that changes the
    # process-wide thread pool size and silently affects unrelated run_in_threadpool
    # calls made by FastAPI/Starlette internals (e.g. large upload spooling).
    #app.state.inference_limiter = anyio.CapacityLimiter(settings.INFERENCE_MAX_WORKERS)
    app.state.inference_executor = ThreadPoolExecutor(
        max_workers=settings.INFERENCE_MAX_WORKERS, thread_name_prefix="inference"
    )

    app.state.pipeline_in_flight = 0
    logger.info("kie-service ready")
    yield
    logger.info("Shutting down kie-service")
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

# uvicorn app.main:app --reload --port 8002
"""
Global exception handlers: ensure ALL errors return a JSON response
adhering strictly to a single schema, including the request_id so clients
can report errors back for easier log tracing.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.exceptions import PipelineError
from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import PIPELINE_ERRORS

logger = get_logger("app.error")


def _error_body(error_code: str, message: str, stage: str | None = None) -> dict:
    return {
        "error_code": error_code,
        "message": message,
        "stage": stage,
        "request_id": request_id_ctx.get(),
    }


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError):
        error_code = getattr(exc, "error_code", exc.__class__.__name__)
        stage = getattr(exc, "stage", "unknown")
        logger.warning(f"PipelineError: {error_code} - {exc.message}")
        PIPELINE_ERRORS.labels(error_code=error_code, stage=stage).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.error_code, exc.message, exc.stage),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        PIPELINE_ERRORS.labels(error_code=exc.error_code, stage=exc.stage or "unknown").inc()
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body("VALIDATION_ERROR", str(exc.errors())),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        PIPELINE_ERRORS.labels(error_code=exc.error_code, stage=exc.stage or "unknown").inc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("INTERNAL_SERVER_ERROR", "Unhandled errors"),
        )

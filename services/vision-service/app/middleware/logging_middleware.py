"""
Middleware log access: method, path, status_code, latency_ms.
"""
import time
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import get_logger

logger = get_logger("app.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        span = trace.get_current_span()
        trace_id = format(span.get_span_context().trace_id, "032x") if span.get_span_context().is_valid else None

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code}",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                    "client_ip": request.client.host if request.client else None,
                    "trace_id": trace_id,
                }
            },
        )
        response.headers["X-Process-Time-Ms"] = str(latency_ms)
        return response

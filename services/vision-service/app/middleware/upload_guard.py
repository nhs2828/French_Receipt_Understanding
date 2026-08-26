"""
Middleware chặn sớm request có Content-Length vượt giới hạn, trước khi FastAPI
đọc/parse body - tránh tốn CPU/RAM decode ảnh quá lớn (rủi ro OOM/DoS).
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import request_id_ctx


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error_code": "FILE_TOO_LARGE",
                    "message": f"Request vượt quá giới hạn {settings.MAX_UPLOAD_SIZE_MB}MB.",
                    "stage": "input",
                    "request_id": request_id_ctx.get(),
                },
            )
        return await call_next(request)

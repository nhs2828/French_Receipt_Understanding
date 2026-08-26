"""
Middleware sinh/giữ request_id cho mỗi request, gắn vào contextvar để logging.py
tự động đính kèm vào mọi log line, trả lại trong response header.
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import request_id_ctx

REQUEST_ID_HEADER = "X-Request-ID"


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming_id = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming_id or str(uuid.uuid4())

        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        # Dùng request.state.request_id (không phải biến local request_id) vì
        # endpoint có thể đã override bằng request_id do client tự truyền
        # (xem app/api/v1/endpoints/extract.py) - đảm bảo response header khớp
        # với request_id thật sự được dùng trong log/response body.
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response
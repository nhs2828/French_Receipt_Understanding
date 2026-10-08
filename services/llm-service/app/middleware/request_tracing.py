"""
Middleware generates/retains a request_id for each request, 
attaches it to a contextvar so logging.py automatically includes it
in every log line, and returns it in the response header.
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

        # Uses request.state.request_id (instead of the local request_id variable)
        # because the endpoint may have overridden it with 
        # a client-supplied request_id (see app/api/v1/endpoints/extract.py)
        # —ensuring the response header matches the actual request_id 
        # used in logs and the response body.
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response
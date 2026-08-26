import httpx
from pydantic import BaseModel


class ReceiptResult(BaseModel):
    instance_id: int
    words: list[str]
    boxes: list[list[int]]
    width: int
    height: int
    processed_image_b64: str


class VisionResponse(BaseModel):
    request_id: str | None = None
    filename: str
    num_instances: int
    results: list[ReceiptResult]


class VisionClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout

    async def process(self, request_id: str, image_bytes: bytes, filename: str = "image.jpg") -> VisionResponse:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.base_url,
                files={"image": (filename, image_bytes, "image/jpeg")},
                headers={"X-Request-ID": request_id}
            )
            resp.raise_for_status()
            return VisionResponse.model_validate(resp.json())
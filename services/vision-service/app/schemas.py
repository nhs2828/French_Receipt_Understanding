from pydantic import BaseModel


class ReceiptResult(BaseModel):
    instance_id: int
    words: list[str]
    boxes: list[list[int]]
    width: int
    height: int
    processed_image_b64: str


class VisionResponse(BaseModel):
    filename: str
    num_instances: int
    results: list[ReceiptResult]
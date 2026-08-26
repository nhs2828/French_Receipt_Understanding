from pydantic import BaseModel, Field


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


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    stage: str | None = None
    request_id: str


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    device: str
    load_time_ms: float | None = None


class LivenessResponse(BaseModel):
    status: str = Field(examples=["alive"])


class ReadinessResponse(BaseModel):
    status: str = Field(examples=["ready", "not_ready"])
    all_models_loaded: bool


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str


class ModelsStatusResponse(BaseModel):
    segmentation: ModelStatus
    text_extraction: ModelStatus
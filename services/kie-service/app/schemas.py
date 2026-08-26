from pydantic import BaseModel, Field


class KIEEntity(BaseModel):
    text: str
    label: str
    box: list[int]


class KIEInstanceResult(BaseModel):
    instance_id: int
    entities: dict[str, list[str]]


class KIEResponse(BaseModel):
    request_id: str | None = None
    filename: str
    num_instances: int
    results: list[KIEInstanceResult]

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
    kie: ModelStatus
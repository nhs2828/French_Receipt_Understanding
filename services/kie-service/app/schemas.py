from pydantic import BaseModel


class KIEEntity(BaseModel):
    text: str
    label: str
    box: list[int]


class KIEInstanceResult(BaseModel):
    instance_id: int
    entities: dict[str, list[str]]


class KIEResponse(BaseModel):
    filename: str
    num_instances: int
    results: list[KIEInstanceResult]
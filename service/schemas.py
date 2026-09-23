from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Prompt text to classify")


class ClassifyResponse(BaseModel):
    label: str
    score: float
    is_attack: bool
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

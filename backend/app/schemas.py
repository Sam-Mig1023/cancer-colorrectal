from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    key: str
    name: str
    input_width: int
    input_height: int
    recommended: bool
    available: bool


class PredictionResponse(BaseModel):
    model: str
    model_key: str
    predicted_class: str
    confidence: float = Field(ge=0, le=100)
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models_available: int
    models_total: int


class LegacyReportRequest(BaseModel):
    language: str = "es"
    model: str
    model_key: str | None = None
    predicted_class: str
    confidence: float = Field(ge=0, le=100)
    probabilities: dict[str, float]
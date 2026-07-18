from pydantic import BaseModel, Field

from typing import Literal


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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatContext(BaseModel):
    active_tab: Literal["diagnosis", "analysis", "training", "dataset"] = "diagnosis"
    selected_model: str | None = Field(default=None, max_length=200)
    predicted_class: str | None = Field(default=None, max_length=50)
    confidence: float | None = Field(default=None, ge=0, le=100)


class ChatRequest(BaseModel):
    language: Literal["es", "en"] = "es"
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    context: ChatContext = Field(default_factory=ChatContext)


class ChatResponse(BaseModel):
    answer: str
    model: str

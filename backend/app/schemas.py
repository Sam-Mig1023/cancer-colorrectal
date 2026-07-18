from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import CLASS_NAMES, MODEL_SPECS


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
    language: Literal["es", "en"] = "es"
    model: str = Field(min_length=1, max_length=120)
    model_key: str | None = None
    predicted_class: str = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=100)
    probabilities: dict[str, float]

    @field_validator("model_key")
    @classmethod
    def validate_model_key(cls, value: str | None) -> str | None:
        if value is not None and value not in MODEL_SPECS:
            raise ValueError("model_key must be a model returned by GET /api/v1/models.")
        return value

    @field_validator("predicted_class")
    @classmethod
    def validate_predicted_class(cls, value: str) -> str:
        if value not in CLASS_NAMES:
            raise ValueError("predicted_class must be one of the configured histology classes.")
        return value

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(CLASS_NAMES):
            raise ValueError("probabilities must contain exactly the configured histology classes.")
        if any(not 0 <= probability <= 100 for probability in value.values()):
            raise ValueError("probabilities must be percentages between 0 and 100.")
        if abs(sum(value.values()) - 100) > 0.2:
            raise ValueError("probabilities must add up to 100 percent.")
        return value


class ScientificEvaluationRequest(BaseModel):
    model_key: str
    dataset_path: str | None = Field(default=None, max_length=500)
    partition: Literal["test"] = "test"
    max_samples: int = Field(default=1000, ge=1, le=10000)
    bootstrap_iterations: int = Field(default=500, ge=100, le=5000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    patient_id_regex: str | None = Field(default=None, max_length=300)
    experiment_id: str | None = Field(default=None, max_length=120)

    @field_validator("model_key")
    @classmethod
    def validate_scientific_model_key(cls, value: str) -> str:
        if value not in MODEL_SPECS:
            raise ValueError("model_key must be a model returned by GET /api/v1/models.")
        return value


class TrainingPlanRequest(BaseModel):
    dataset_path: str | None = Field(default=None, max_length=500)
    train_partition: str = Field(default="train", min_length=1, max_length=80)
    validation_partition: str = Field(default="validation", min_length=1, max_length=80)
    test_partition: str = Field(default="test", min_length=1, max_length=80)
    architecture: Literal["mobilenetv2_transfer"] = "mobilenetv2_transfer"
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=32, ge=1, le=128)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    experiment_name: str = Field(default="mobilenetv2_experiment", min_length=3, max_length=80)
    horizontal_flip: bool = True
    rotation_factor: float = Field(default=0.05, ge=0, le=0.5)
    early_stopping_patience: int = Field(default=3, ge=1, le=30)
    pretrained_weights: bool = True


class TrainingJobRequest(TrainingPlanRequest):
    pass

class StatisticalComparisonRequest(BaseModel):
    first_artifact_path: str = Field(min_length=1, max_length=600)
    second_artifact_path: str = Field(min_length=1, max_length=600)
    class_name: str = Field(min_length=1, max_length=20)
    bootstrap_iterations: int = Field(default=2000, ge=100, le=5000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("class_name")
    @classmethod
    def validate_comparison_class(cls, value: str) -> str:
        if value not in CLASS_NAMES:
            raise ValueError("class_name must be one of the configured histology classes.")
        return value
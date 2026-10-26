from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DISCLAIMER = ("Research prototype only. Not a medical device and not a diagnosis. The model was trained on a small, controlled "
              "research dataset and has no demonstrated clinical validity; results on other microphones, environments or tasks may be unreliable.")


class Message(BaseModel):
    code: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ValidationInfo(BaseModel):
    ok: bool
    info: dict[str, Any]
    warnings: list[Message]


class AnalysisResponse(BaseModel):
    validation: ValidationInfo
    sample_rate: int
    features: dict[str, float | None] = Field(description="Extracted features; null = undefined measurement")
    waveform: dict[str, list[float]] | None = None
    spectrogram: dict[str, Any] | None = None


class Explanation(BaseModel):
    method: str
    top_features: list[dict[str, Any]]
    note: str


class PredictionResult(BaseModel):
    prediction: str = Field(description="'above_threshold' or 'below_threshold' relative to the model's research operating threshold")
    prediction_label: str
    score: float = Field(description="Uncalibrated model score. NOT a probability.")
    threshold: float
    score_type: str | None
    model_version: str | None
    model_name: str | None
    warnings: list[Message]
    inference_ms: float
    explanation: Explanation | None = None


class PredictResponse(BaseModel):
    result: PredictionResult
    analysis: AnalysisResponse
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class DemoSample(BaseModel):
    sample_id: str
    task_code: str | None
    task_family: str | None
    partition: str = Field(description="'test' (held out) or 'train' (the model saw this subject) or 'unknown'")


class DemoPredictResponse(PredictResponse):
    dataset_label_not_a_model_output: str
    partition: str

"""
Response schema models for API endpoints.

NOTE: We avoid Field(...) (Ellipsis) in response models because FastAPI 0.104
+ Pydantic 2.5 has a known serialization bug with Ellipsis in OpenAPI schema
generation. Instead we use plain type annotations for required fields.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """Top-5 prediction result."""

    sign: str
    confidence: float

    class Config:
        json_schema_extra = {"example": {"sign": "hello", "confidence": 0.95}}


class PredictionResponse(BaseModel):
    """Single frame prediction response."""

    sign: str
    confidence: float
    top5: List[TopPrediction]
    processing_time_ms: float
    landmarks: Optional[List[List[float]]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "sign": "hello",
                "confidence": 0.95,
                "top5": [
                    {"sign": "hello", "confidence": 0.95},
                    {"sign": "hi", "confidence": 0.03},
                ],
                "processing_time_ms": 45.2,
                "landmarks": None,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    signs: List[str]
    confidences: List[float]
    sentence: Optional[str] = None
    top5_per_sign: Optional[List[List[TopPrediction]]] = None
    frames_processed: int
    processing_time_sec: float

    class Config:
        json_schema_extra = {
            "example": {
                "signs": ["hello", "world"],
                "confidences": [0.95, 0.87],
                "sentence": "Hello world",
                "frames_processed": 120,
                "processing_time_sec": 2.3,
            }
        }


class VideoPredictionResponse(BaseModel):
    """Video file prediction response."""

    signs: List[str]
    confidences: List[float]
    sentence: Optional[str] = None
    frames_processed: int
    processing_time_sec: float
    video_duration_sec: float
    fps: float


class TokenResponse(BaseModel):
    """Authentication token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    version: str
    timestamp: str

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:45.123Z",
            }
        }


class MetricsResponse(BaseModel):
    """Application metrics response."""

    predictions_count: int
    avg_latency_ms: float
    uptime_seconds: float
    model_loaded: bool
    device: str

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "predictions_count": 1523,
                "avg_latency_ms": 42.5,
                "uptime_seconds": 3600,
                "model_loaded": True,
                "device": "cuda",
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid frame format",
                "detail": "Frame must be valid JPEG/PNG in base64",
                "request_id": "req-123-abc",
            }
        }


class WebSocketResponse(BaseModel):
    """WebSocket response message."""

    type: str
    sign: Optional[str] = None
    confidence: Optional[float] = None
    frame_id: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "type": "prediction",
                "sign": "hello",
                "confidence": 0.95,
                "frame_id": 1,
                "processing_time_ms": 45.2,
            }
        }

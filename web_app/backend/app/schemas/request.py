"""
Request schema models for API endpoints.

NOTE: We avoid Field(...) (Ellipsis) in models because FastAPI 0.104
+ Pydantic 2.5 has a serialization bug with Ellipsis in OpenAPI generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """User login request."""

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)

    class Config:
        json_schema_extra = {
            "example": {"username": "testuser", "password": "password123"}
        }


class FramePredictRequest(BaseModel):
    """Single frame prediction request."""

    frame_base64: str = Field(alias="frame", description="Base64 encoded JPEG frame")
    extract_landmarks: bool = Field(
        default=False, description="Return landmarks in response"
    )

    class Config:
        populate_by_name = True  # Accept both 'frame' and 'frame_base64'
        json_schema_extra = {
            "example": {
                "frame": "/9j/4AAQSkZJRg...",
                "extract_landmarks": False,
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction from landmarks."""

    landmarks: List[List[List[float]]] = Field(
        description="Landmarks array [T, 543, 3]"
    )
    enable_gemini: bool = Field(
        default=True, description="Enable Gemini sentence generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [[[0.1, 0.2, 0.3]]],
                "enable_gemini": True,
            }
        }


class GenerateSentenceRequest(BaseModel):
    """Request for generating a sentence from recognized signs."""

    signs: List[str] = Field(description="List of recognized sign labels")

    class Config:
        json_schema_extra = {
            "example": {
                "signs": ["hello", "world"],
            }
        }


class VideoPredictRequest(BaseModel):
    """Video file upload for batch processing."""

    # This is handled via multipart form data, not JSON
    # See the route handler for details
    enable_gemini: bool = Field(
        default=True, description="Enable Gemini sentence generation"
    )


class WebSocketMessage(BaseModel):
    """WebSocket message schema."""

    type: str = Field(description="Message type (frame, ping, etc.)")
    data: Optional[dict] = Field(default=None, description="Message payload")
    frame_id: Optional[int] = Field(default=None, description="Frame ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "frame",
                "data": {"frame_base64": "/9j/4AAQSkZJRg..."},
                "frame_id": 1,
            }
        }


class TranslateGlossRequest(BaseModel):
    """Request to translate English text into ASL glosses."""

    text: str = Field(
        min_length=1,
        max_length=500,
        description="English sentence to translate into ASL glosses",
    )
    use_gemini: bool = Field(
        default=True,
        description=(
            "Use Gemini API for intelligent gloss extraction "
            "(falls back to rule-based if unavailable)"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, what is your name?",
                "use_gemini": True,
            }
        }

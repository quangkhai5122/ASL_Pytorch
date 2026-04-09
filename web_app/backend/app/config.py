"""
Configuration module for the ASL Recognition Web Application.
Loads environment variables and provides application-wide settings.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # ==========================================================================
    # Application Environment
    # ==========================================================================
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-key-change-in-production-must-be-32-chars-min"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sign Language Recognition API"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "ASL/GISLR (Vietnamese Sign Language) recognition and production"

    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False

    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    MODEL_PATH: str = "./models/model_best_full_training.pth"
    DEVICE: str = "auto"  # Options: auto, cuda, cpu
    NUM_CLASSES: int = 250
    INPUT_SIZE: int = 64
    N_ROWS: int = 543  # Total landmark points from MediaPipe
    N_DIMS: int = 3  # X, Y, Z coordinates

    # ==========================================================================
    # Model Inference Parameters
    # ==========================================================================
    WINDOW_SIZE: int = 32  # Frames in rolling window
    MIN_FRAMES_FOR_INFER: int = 8  # Minimum frames before inference
    INFER_STRIDE_FRAMES: int = 4  # Run inference every N frames
    CONFIDENCE_ON: float = 0.25  # Hysteresis ON threshold
    CONFIDENCE_OFF: float = 0.12  # Hysteresis OFF threshold
    CONFIDENCE_UNKNOWN: float = 0.08  # Below this = "Unknown"
    STABILITY_N: int = 3  # Stable predictions needed to commit
    PROB_EMA_ALPHA: float = 0.55  # Probability smoothing (0-1, higher=smoother)
    MOVEMENT_THRESHOLD: float = 0.003  # Motion gate threshold
    MIN_HAND_POINTS: int = 4  # Minimum detected hand points

    # ==========================================================================
    # Gemini API Configuration
    # ==========================================================================
    ENABLE_GEMINI: bool = True
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ==========================================================================
    # AWS S3 Configuration (Optional)
    # ==========================================================================
    USE_S3: bool = False
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    # ==========================================================================
    # CORS & Frontend Configuration
    # ==========================================================================
    CORS_ORIGINS: List[str] = [
        "https://signlanguagetrans.lovable.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]
    ALLOWED_HOSTS: List[str] = ["*"]

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    DATABASE_URL: str = "sqlite:///./gislr.db"
    DATABASE_ECHO: bool = False

    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # Options: json, text

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60

    # ==========================================================================
    # Video Processing Configuration
    # ==========================================================================
    MAX_VIDEO_FILE_SIZE_MB: int = 500
    SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "flv", "wmv"]
    VIDEO_FRAME_RATE: int = 30
    MAX_VIDEO_DURATION_SECONDS: int = 300

    # ==========================================================================
    # WebSocket Configuration
    # ==========================================================================
    WEBSOCKET_PING_INTERVAL: int = 25
    WEBSOCKET_PING_TIMEOUT: int = 10

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================
    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        """Pydantic config for settings."""

        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Loads settings from environment on first call, then returns cached instance.

    Returns:
        Settings: Application configuration instance
    """
    return Settings()


# Export singleton instance
settings = get_settings()

# =============================================================================
# Computed Properties
# =============================================================================

# Validate model path on startup
if settings.MODEL_PATH and not os.path.exists(settings.MODEL_PATH):
    import warnings

    warnings.warn(
        f"Model path does not exist: {settings.MODEL_PATH}. "
        f"This may cause errors during initialization.",
        UserWarning,
    )

# Log configuration on startup
if settings.DEBUG:
    print(f"Application Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Model Device: {settings.DEVICE}")
    print(f"CORS Origins: {settings.CORS_ORIGINS}")

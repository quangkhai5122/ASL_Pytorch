"""
Dependency injection module for FastAPI.
Handles JWT authentication, current user resolution, and service access.
"""

from typing import Optional, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

from app.config import settings
from app.core.auth import decode_token


# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: Any = Depends(security),
) -> str:
    """
    Verify JWT token and return current user.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Username from token

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    username = decode_token(token)

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username


async def get_optional_user(
    credentials: Optional[Any] = Depends(security),
) -> Optional[str]:
    """
    Try to get current user, but don't fail if token is missing.
    Used for optional authentication endpoints.

    Args:
        credentials: Optional HTTP Bearer credentials

    Returns:
        Username if token is valid, None otherwise
    """
    if credentials is None:
        return None

    token = credentials.credentials
    username = decode_token(token)
    return username


# Service imports (lazy-loaded by FastAPI)
def get_model_service():
    """Get model inference service (singleton)."""
    from app.services.model_inference import model_service

    return model_service


def get_landmark_service():
    """Get landmark extraction service (singleton)."""
    from app.services.landmark_extraction import landmark_service

    return landmark_service


def get_video_service():
    """Get video processing service (singleton)."""
    from app.services.video_processing import video_service

    return video_service


def get_gemini_service():
    """Get Gemini service (singleton)."""
    from app.services.gemini_service import gemini_service

    return gemini_service

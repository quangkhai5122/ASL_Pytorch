"""
Authentication API routes.
Handles login and token management.
"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.core.auth import create_access_token, verify_password, get_password_hash
from app.schemas.request import LoginRequest
from app.schemas.response import TokenResponse, ErrorResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Simple in-memory user storage for MVP
# In production, use a database
USERS_DB = {
    "testuser": {
        "username": "testuser",
        "hashed_password": get_password_hash("testpass123"),
        "full_name": "Test User",
    },
    "demo": {
        "username": "demo",
        "hashed_password": get_password_hash("demo123"),
        "full_name": "Demo User",
    },
}


@router.post("/login", response_model=TokenResponse, responses={401: {"model": ErrorResponse}})
async def login(credentials: LoginRequest) -> TokenResponse:
    """
    User login endpoint.
    Returns JWT access token valid for 30 minutes (configurable).

    Args:
        credentials: Login credentials (username, password)

    Returns:
        Access token response

    Raises:
        HTTPException: If credentials are invalid
    """
    # Check if user exists
    user = USERS_DB.get(credentials.username)
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials.username}, expires_delta=access_token_expires
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/verify", response_model=dict, responses={401: {"model": ErrorResponse}})
async def verify_token(token: str) -> dict:
    """
    Verify if a token is valid.

    Args:
        token: JWT token to verify

    Returns:
        IsValid status

    Raises:
        HTTPException: If token is invalid
    """
    from app.core.auth import decode_token

    username = decode_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return {"valid": True, "username": username}


# Default test credentials endpoint (for frontend reference)
@router.get("/test-credentials")
async def get_test_credentials() -> dict:
    """
    Get test credentials for development/testing.
    Remove this in production!
    """
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test credentials only available in development",
        )

    return {
        "credentials": [
            {"username": "testuser", "password": "testpass123"},
            {"username": "demo", "password": "demo123"},
        ],
        "note": "These are test credentials for development only",
    }

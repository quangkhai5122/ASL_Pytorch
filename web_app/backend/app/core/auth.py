"""
Authentication and JWT token management module.
Handles login, token generation, and token validation.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
import bcrypt

from app.config import settings

# Password hashing context - compatible with bcrypt 4.x
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)


class TokenData:
    """Token payload data."""

    def __init__(self, username: Optional[str] = None):
        self.username = username


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify plain password against hashed password.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a plain text password.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in token (typically {"sub": username})
        expires_delta: Optional token expiration time. If None, uses default.

    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )

    return encoded_jwt


def decode_token(token: str) -> Optional[str]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        Optional[str]: Username if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")

        if username is None:
            return None

        return username

    except JWTError:
        return None


def verify_token(token: str) -> bool:
    """
    Verify if a token is valid.

    Args:
        token: JWT token string

    Returns:
        bool: True if token is valid, False otherwise
    """
    username = decode_token(token)
    return username is not None

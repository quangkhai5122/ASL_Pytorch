"""
Health check and metrics API routes.
"""

from datetime import datetime, timezone
from fastapi import APIRouter

from app.config import settings
from app.schemas.response import HealthResponse, MetricsResponse
from app.services.model_inference import model_service
from app.services.landmark_extraction import landmark_service
from app.services.video_processing import video_service

router = APIRouter(prefix="/api/v1", tags=["health"])

# Track startup time for uptime calculation
import time

_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns status of API and loaded models.

    Returns:
        Health status with model status
    """
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "degraded",
        model_loaded=model_service.is_loaded(),
        device=model_service.get_device(),
        version=settings.PROJECT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Get API metrics and inference statistics.

    Returns:
        Metrics including prediction count, latency, uptime
    """
    stats = model_service.get_stats()
    uptime_sec = time.time() - _startup_time

    return MetricsResponse(
        predictions_count=stats["total_predictions"],
        avg_latency_ms=stats["avg_latency_ms"],
        uptime_seconds=uptime_sec,
        model_loaded=model_service.is_loaded(),
        device=model_service.get_device(),
    )


@router.get("/info")
async def get_info() -> dict:
    """
    Get API information and configuration.

    Returns:
        API metadata
    """
    return {
        "project_name": settings.PROJECT_NAME,
        "project_version": settings.PROJECT_VERSION,
        "project_description": settings.PROJECT_DESCRIPTION,
        "environment": settings.ENVIRONMENT,
        "device": model_service.get_device(),
        "is_model_loaded": model_service.is_loaded(),
        "num_classes": settings.NUM_CLASSES,
        "gemini_enabled": settings.ENABLE_GEMINI,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

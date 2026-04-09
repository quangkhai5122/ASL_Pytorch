"""
Main FastAPI Application
Sign Language Recognition Web Backend
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from app.config import settings
from app.api.routes import auth, health, predict
from app.api.websocket import routes as websocket_routes

# Request tracking
request_id_counter = 0


def custom_openapi():
    """Generate custom OpenAPI documentation."""
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title=settings.PROJECT_NAME,
            version=settings.PROJECT_VERSION,
            description=settings.PROJECT_DESCRIPTION,
            routes=app.routes,
        )
    return app.openapi_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("\n" + "=" * 80)
    print(">>> Starting Sign Language Recognition API")
    print("=" * 80)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug: {settings.DEBUG}")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Device: {settings.DEVICE}")
    print(f"CORS Origins: {settings.CORS_ORIGINS}")

    # Initialize services
    try:
        from app.services.model_inference import model_service
        from app.services.landmark_extraction import landmark_service
        from app.services.gemini_service import gemini_service

        print(f"[OK] Model service initialized: {model_service.is_loaded()}")
        print(f"[OK] Landmark service initialized")
        print(f"[OK] Gemini service enabled: {gemini_service.is_enabled()}")

    except Exception as e:
        print(f"[ERROR] Error initializing services: {str(e)}")

    print("=" * 80 + "\n")

    yield

    # Shutdown
    print("\n" + "=" * 80)
    print("<<< Shutting down Sign Language Recognition API")
    print("=" * 80 + "\n")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Set custom OpenAPI schema
app.openapi = custom_openapi

# =============================================================================
# CORS Middleware
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request Logging Middleware
# =============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    global request_id_counter
    request_id_counter += 1
    request_id = f"req-{request_id_counter}"

    start_time = time.time()
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    if settings.DEBUG:
        print(
            f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({process_time*1000:.2f}ms)"
        )

    return response


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An error occurred",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# =============================================================================
# Include Routers
# =============================================================================
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(websocket_routes.router)

# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sign Language Recognition API",
        "version": settings.PROJECT_VERSION,
        "docs": f"{settings.API_V1_STR}/docs",
        "redoc": f"{settings.API_V1_STR}/redoc",
        "health": f"{settings.API_V1_STR}/health",
        "websocket": f"{settings.API_V1_STR}/ws/stream",
        "websocket_stats": f"{settings.API_V1_STR}/ws/stats",
        "test_webcam": "/test-webcam",
    }


@app.get("/test-webcam")
async def test_webcam():
    """Serve the webcam test page for debugging the inference pipeline."""
    import os
    from fastapi.responses import HTMLResponse

    test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_webcam.html")
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>test_webcam.html not found</h1>", status_code=404)


@app.get(f"{settings.API_V1_STR}/status")
async def status():
    """Detailed status endpoint."""
    from app.services.model_inference import model_service

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "model_loaded": model_service.is_loaded(),
        "device": model_service.get_device(),
        "version": settings.PROJECT_VERSION,
    }


# =============================================================================
# WebSocket (will be added in next phase)
# =============================================================================
# WebSocket endpoints can be added here


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )

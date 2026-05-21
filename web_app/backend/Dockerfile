# Stage 1: Builder
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create a pip cache directory
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="Sign Language Recognition Team"
LABEL description="ASL/GISLR Sign Language Recognition - Web Backend API"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy source code
COPY app/ /app/app/
COPY models/ /app/models/
COPY data/ /app/data/
COPY scripts/ /app/scripts/
COPY .env.example /app/.env.example

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=auto
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Default command: run with uvicorn (used in docker-compose)
# For production, use gunicorn instead:
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
